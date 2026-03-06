#!/usr/bin/env python
"""Extract per-review, per-user, and per-item features for top users (step 3).

For each category:
- Load ``top_users.parquet`` from Drive or local cache.
- Stream the category review ``.jsonl.gz`` and keep only top-user reviews.
- Produce per-review, per-user, and per-item feature tables.
- Upload results to Drive and optionally clean up local files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from amazon_next_category.io.data_io import (
    delete_remote_by_rel_path,
    ensure_local,
    ensure_local_path,
    remote_file_exists_by_rel_path,
    resync_registry,
    upload_to_drive,
)
from amazon_next_category.utils.config import (
    CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    META_URL_TEMPLATE,
    REVIEW_URL_TEMPLATE,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_rating_hist_plot(rating_hist: Counter, out_path: Path, title: str) -> None:
    if not rating_hist:
        logger.warning("No ratings to plot.")
        return
    items = sorted(rating_hist.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved rating hist: %s", out_path)


def save_helpful_hist_plot(helpful_hist: Counter, out_path: Path, title: str) -> None:
    if not helpful_hist:
        logger.warning("No helpful_votes to plot.")
        return
    items = sorted(helpful_hist.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Helpful votes (clipped)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved helpful hist: %s", out_path)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_if_needed(url: str, dest: Path, force: bool = False) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.debug("File already exists: %s", dest)
        return

    logger.info("Downloading%s: %s -> %s", " (force)" if force else "", url, dest)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    logger.info("Download complete: %s", dest)


def ensure_raw_gzip_or_download(
    path: Path, url: str, allow_download: bool, repo_root: Path
) -> None:
    if path.exists():
        return
    rel = str(path.relative_to(repo_root))
    try:
        logger.info("Trying Drive for %s", rel)
        ensure_local_path(rel)
        if path.exists():
            return
    except Exception:
        pass
    if allow_download:
        download_if_needed(url, path, force=False)
    else:
        raise FileNotFoundError(
            f"Missing {path} and --no-download is set; "
            "no Drive entry or HTTP download attempted."
        )


def ensure_outputs_from_drive(paths: List[Path], repo_root: Path) -> None:
    for p in paths:
        if p.exists():
            continue
        rel = str(p.relative_to(repo_root))
        try:
            ensure_local_path(rel)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Top-users + meta helpers
# ---------------------------------------------------------------------------


def load_top_users(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "user_id" in df.columns:
        return df
    df = df.reset_index()
    if "user_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "user_id"})
    return df


def load_item_meta(meta_gz: Path, category: str) -> Dict[str, Dict]:
    """Load ``parent_asin -> {item_avg_rating, item_categories}`` from meta gz."""
    item_meta: Dict[str, Dict] = {}
    if not meta_gz.exists():
        logger.warning("No meta file for %s at %s.", category, meta_gz)
        return item_meta

    logger.info("Loading item meta for %s from %s", category, meta_gz)
    first_bad = True
    with gzip.open(meta_gz, "rt", encoding="utf-8") as mf:
        for mline in mf:
            mline = mline.strip()
            if not mline:
                continue
            try:
                mobj = json.loads(mline)
            except json.JSONDecodeError as e:
                if first_bad:
                    logger.warning("Skipping malformed JSON in %s meta: %s", category, e)
                    first_bad = False
                continue

            a = mobj.get("parent_asin") or mobj.get("asin")
            if not a:
                continue

            item_avg = None
            for k in ("avg_rating", "average_rating", "rating"):
                v = mobj.get(k)
                if v is not None:
                    try:
                        item_avg = float(v)
                        break
                    except Exception:
                        pass

            cats = None
            raw_cats = mobj.get("categories") or mobj.get("category")
            if raw_cats:
                if isinstance(raw_cats, list):
                    flat: List[str] = []
                    for el in raw_cats:
                        if isinstance(el, list):
                            flat.extend([str(x) for x in el if x is not None])
                        else:
                            flat.append(str(el))
                    cats = list(dict.fromkeys(flat))
                else:
                    cats = [str(raw_cats)]

            item_meta[str(a)] = {"item_avg_rating": item_avg, "item_categories": cats}

    return item_meta


# ---------------------------------------------------------------------------
# Core streaming parser
# ---------------------------------------------------------------------------


def parse_reviews_for_top_users(
    review_gz: Path,
    top_users_set: set,
    item_meta: Dict[str, Dict],
    category: str,
    progress_interval: int,
    total_lines: Optional[int],
    out_reviews_parquet: Path,
    parquet_batch_size: int = 500_000,
) -> Tuple[Counter, Counter, int, int]:
    """Stream *review_gz*, filter to top users, write per-review Parquet in batches.

    Returns ``(rating_hist, helpful_hist, n_users_with_reviews, n_reviews_kept)``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out_reviews_parquet.exists():
        out_reviews_parquet.unlink()

    rating_hist: Counter = Counter()
    helpful_hist: Counter = Counter()
    seen_users: set = set()

    n_lines = 0
    n_kept = 0
    start_time = time.time()

    rows_batch: List[Dict] = []
    writer = None

    def flush_batch() -> None:
        nonlocal writer, rows_batch
        if not rows_batch:
            return
        keys = rows_batch[0].keys()
        cols = {k: [row[k] for row in rows_batch] for k in keys}
        table = pa.Table.from_pydict(cols)
        if writer is None:
            writer = pq.ParquetWriter(str(out_reviews_parquet), table.schema)
        writer.write_table(table)
        rows_batch = []

    with gzip.open(review_gz, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = obj.get("user_id")
            if user_id is None:
                continue
            user_id = str(user_id)
            if user_id not in top_users_set:
                continue

            product_id = (
                obj.get("asin") or obj.get("product_id") or obj.get("item_id") or obj.get("id")
            )
            rating = obj.get("rating")

            hv = obj.get("helpful_votes", obj.get("helpful_vote", 0))
            try:
                hv = int(hv)
            except (TypeError, ValueError):
                hv = 0
            hv_clipped = min(hv, 50)

            unix_time = (
                obj.get("timestamp")
                or obj.get("unixReviewTime")
                or obj.get("unix_review_time")
                or obj.get("unix_time")
            )

            verified = obj.get("verified_purchase") or obj.get("verified") or False
            im = item_meta.get(str(product_id), {}) if product_id is not None else {}

            rows_batch.append(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "unixReviewTime": int(unix_time) if unix_time is not None else None,
                    "rating": float(rating) if rating is not None else None,
                    "helpful_votes": hv,
                    "helpful_votes_clipped": hv_clipped,
                    "verified_purchase": bool(verified),
                    "item_avg_rating": im.get("item_avg_rating"),
                    "item_categories": im.get("item_categories"),
                }
            )
            seen_users.add(user_id)
            n_kept += 1

            if len(rows_batch) >= parquet_batch_size:
                flush_batch()

            if progress_interval > 0 and n_lines % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = n_lines / elapsed if elapsed > 0 else 0.0
                if total_lines:
                    frac = min(n_lines / total_lines, 1.0)
                    eta_str = "unknown"
                    if rate > 0:
                        remaining = max(total_lines - n_lines, 0)
                        eta_sec = remaining / rate
                        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                    logger.info(
                        "%s: %d/%d lines (%.1f%%), kept=%d, ETA=%s",
                        category,
                        n_lines,
                        total_lines,
                        frac * 100,
                        n_kept,
                        eta_str,
                    )
                else:
                    logger.info(
                        "%s: lines=%d, kept=%d, users=%d",
                        category,
                        n_lines,
                        n_kept,
                        len(seen_users),
                    )

            try:
                if rating is not None:
                    rating_hist[float(rating)] += 1
            except (TypeError, ValueError):
                pass
            helpful_hist[hv_clipped] += 1

    flush_batch()
    if writer is not None:
        writer.close()
    else:
        pd.DataFrame(
            columns=[
                "user_id",
                "product_id",
                "unixReviewTime",
                "rating",
                "helpful_votes",
                "helpful_votes_clipped",
                "verified_purchase",
                "item_avg_rating",
                "item_categories",
            ]
        ).to_parquet(out_reviews_parquet, index=False)

    logger.info(
        "Filtered reviews: rows=%d, users=%d, lines_seen=%d",
        n_kept,
        len(seen_users),
        n_lines,
    )
    return rating_hist, helpful_hist, len(seen_users), n_kept


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------


def process_category(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    top_users_df: pd.DataFrame,
    top_users_set: set,
    allow_download: bool,
    cleanup_raw: bool,
    cleanup_processed: str,
    repo_root: Path,
    progress_interval: int,
) -> None:
    logger.info("=== Category (top-users): %s ===", category)

    review_gz = raw_dir / "reviews" / f"{category}.jsonl.gz"
    meta_gz = raw_dir / "meta" / f"meta_{category}.jsonl.gz"
    review_url = REVIEW_URL_TEMPLATE.format(category=category)
    meta_url = META_URL_TEMPLATE.format(category=category)

    cat_proc_dir = processed_dir / category
    cat_proc_dir.mkdir(parents=True, exist_ok=True)

    out_reviews_parquet = cat_proc_dir / f"top_user_reviews_{category}.parquet"
    out_user_features = cat_proc_dir / f"top_user_features_{category}.parquet"
    out_item_features = cat_proc_dir / f"top_item_features_{category}.parquet"
    out_stats_json = cat_proc_dir / f"top_user_review_stats_{category}.json"
    rating_png = cat_proc_dir / f"top_users_rating_hist_{category}.png"
    helpful_png = cat_proc_dir / f"top_users_helpful_hist_{category}.png"

    expected_outputs = [
        out_reviews_parquet,
        out_user_features,
        out_item_features,
        out_stats_json,
        rating_png,
        helpful_png,
    ]

    # --- Lock ---
    lock_dir = repo_root / "data" / "locks" / "03_user_features"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{category}.lock"
    rel_lock = str(lock_path.relative_to(repo_root))

    try:
        if remote_file_exists_by_rel_path(rel_lock) and not lock_path.exists():
            ensure_local_path(rel_lock)
    except Exception:
        pass

    if lock_path.exists():
        logger.info("Lock exists for %s; checking completion...", category)
        ensure_outputs_from_drive(expected_outputs, repo_root)
        if all(p.exists() for p in expected_outputs):
            logger.info("All outputs exist; clearing lock for %s.", category)
            try:
                lock_path.unlink()
            except OSError:
                pass
            try:
                delete_remote_by_rel_path(rel_lock)
            except Exception as e:
                logger.warning("Failed to delete remote lock: %s", e)
            return
        else:
            logger.info("Stale lock for %s; taking over.", category)
            try:
                lock_path.unlink()
            except OSError:
                pass
            try:
                delete_remote_by_rel_path(rel_lock)
            except Exception as e:
                logger.warning("Failed to delete remote lock: %s", e)

    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write("locked\n")
    try:
        upload_to_drive(lock_path)
    except Exception as e:
        logger.warning("Failed to upload lock: %s", e)

    try:
        ensure_outputs_from_drive(expected_outputs, repo_root)

        if all(p.exists() for p in expected_outputs):
            logger.info("All outputs exist for %s; skipping.", category)
            if cleanup_raw:
                for p in (review_gz, meta_gz):
                    if p.exists():
                        p.unlink()
            return

        need_step1 = not (out_reviews_parquet.exists() and out_stats_json.exists())
        need_step2 = not out_user_features.exists()
        need_step3 = not out_item_features.exists()
        need_step4 = not (rating_png.exists() and helpful_png.exists())

        total_lines: Optional[int] = None

        if need_step1:
            ensure_raw_gzip_or_download(review_gz, review_url, allow_download, repo_root)

        need_meta = need_step1 or need_step3
        item_meta: Dict[str, Dict] = {}

        if need_meta:
            ensure_raw_gzip_or_download(meta_gz, meta_url, allow_download, repo_root)
            if meta_gz.exists():
                try:
                    item_meta = load_item_meta(meta_gz, category)
                except EOFError:
                    logger.warning("Truncated meta gz for %s; re-downloading.", category)
                    download_if_needed(meta_url, meta_gz, force=True)
                    item_meta = load_item_meta(meta_gz, category)

        logger.info("top_users: %d users; filtering reviews...", len(top_users_set))

        # Step 1
        if not need_step1:
            logger.info("Reviews + stats already exist for %s; loading.", category)
            with open(out_stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            rating_hist = Counter({float(k): v for k, v in stats.get("rating_hist", {}).items()})
            helpful_hist = Counter({int(k): v for k, v in stats.get("helpful_hist", {}).items()})
            n_reviews = int(stats.get("n_reviews", 0))
            n_users = int(stats.get("n_users", 0))
        else:
            logger.info("Parsing review gz for top users (%s)...", category)
            try:
                rating_hist, helpful_hist, n_users, n_reviews = parse_reviews_for_top_users(
                    review_gz,
                    top_users_set,
                    item_meta,
                    category,
                    progress_interval,
                    total_lines,
                    out_reviews_parquet,
                )
            except EOFError:
                logger.warning("Truncated review gz for %s; re-downloading.", category)
                download_if_needed(review_url, review_gz, force=True)
                rating_hist, helpful_hist, n_users, n_reviews = parse_reviews_for_top_users(
                    review_gz,
                    top_users_set,
                    item_meta,
                    category,
                    progress_interval,
                    total_lines,
                    out_reviews_parquet,
                )

            stats_out = {
                "category": category,
                "n_reviews": int(n_reviews),
                "n_users": int(n_users),
                "rating_hist": dict(rating_hist),
                "helpful_hist": dict(helpful_hist),
            }
            with open(out_stats_json, "w", encoding="utf-8") as f:
                json.dump(stats_out, f, indent=2)
            logger.info("Saved stats: %s", out_stats_json)

        if n_reviews == 0:
            logger.warning("No reviews for top users in %s.", category)
            for p, col in [(out_user_features, "user_id"), (out_item_features, "product_id")]:
                if not p.exists():
                    pd.DataFrame(columns=[col]).to_parquet(p, index=False)
            if need_step4:
                save_rating_hist_plot(
                    rating_hist,
                    rating_png,
                    title=f"Top users rating distribution ({category})",
                )
                save_helpful_hist_plot(
                    helpful_hist,
                    helpful_png,
                    title=f"Top users helpful votes ({category})",
                )
            if cleanup_raw:
                for p in (review_gz, meta_gz):
                    if p.exists():
                        p.unlink()
            return

        reviews_df: Optional[pd.DataFrame] = None
        if (need_step2 or need_step3) and n_reviews > 0:
            logger.info("Loading top-user reviews parquet for aggregation...")
            reviews_df = pd.read_parquet(out_reviews_parquet)

        # Step 2: per-user features
        if not need_step2:
            logger.info("Per-user features already exist: %s", out_user_features)
        else:
            logger.info("Computing per-user aggregated features...")
            agg = (
                reviews_df.groupby("user_id")
                .agg(
                    num_purchases_in_category=("product_id", "count"),
                    avg_rating_in_category=("rating", "mean"),
                    avg_helpful_vote_in_category=("helpful_votes", "mean"),
                    avg_item_avg_rating=("item_avg_rating", "mean"),
                    first_review_time=("unixReviewTime", "min"),
                    last_review_time=("unixReviewTime", "max"),
                )
                .reset_index()
            )
            tu = top_users_df.copy()
            if "user_id" not in tu.columns:
                tu = tu.reset_index()
            tu["user_id"] = tu["user_id"].astype(str)
            agg["user_id"] = agg["user_id"].astype(str)
            sel = agg.merge(tu, on="user_id", how="left")
            sel.to_parquet(out_user_features, index=False)
            logger.info("Saved per-user features: %s", out_user_features)

        # Step 3: per-item features
        if not need_step3:
            logger.info("Per-item features already exist: %s", out_item_features)
        else:
            logger.info("Computing per-item aggregated features...")
            item_agg = (
                reviews_df.groupby("product_id")
                .agg(
                    num_topuser_reviews=("user_id", "count"),
                    num_unique_topusers=("user_id", "nunique"),
                    avg_rating_topusers=("rating", "mean"),
                    avg_helpful_votes_topusers=("helpful_votes", "mean"),
                    first_review_time=("unixReviewTime", "min"),
                    last_review_time=("unixReviewTime", "max"),
                )
                .reset_index()
            )
            _avg_map = {k: v.get("item_avg_rating") for k, v in item_meta.items()}
            _cat_map = {k: v.get("item_categories") for k, v in item_meta.items()}
            item_agg["item_avg_rating_meta"] = item_agg["product_id"].map(
                lambda a: _avg_map.get(str(a))
            )
            item_agg["item_categories_meta"] = item_agg["product_id"].map(
                lambda a: _cat_map.get(str(a))
            )
            item_agg.to_parquet(out_item_features, index=False)
            logger.info("Saved per-item features: %s", out_item_features)

        # Step 4: plots
        if not rating_png.exists():
            save_rating_hist_plot(
                rating_hist,
                rating_png,
                title=f"Top users rating distribution ({category})",
            )
        if not helpful_png.exists():
            save_helpful_hist_plot(
                helpful_hist,
                helpful_png,
                title=f"Top users helpful votes ({category})",
            )

        # Upload
        upload_targets = [
            out_reviews_parquet,
            out_user_features,
            out_item_features,
            out_stats_json,
            rating_png,
            helpful_png,
        ]
        logger.info("Uploading outputs for %s to Drive...", category)
        for p in upload_targets:
            try:
                upload_to_drive(p)
            except Exception as e:
                logger.warning("Failed to upload %s: %s", p, e)

        # Cleanup
        if cleanup_processed != "none":
            for p in upload_targets:
                if cleanup_processed == "parquet" and p.suffix != ".parquet":
                    continue
                try:
                    if p.exists():
                        p.unlink()
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", p, e)

        if cleanup_raw:
            for p in (review_gz, meta_gz):
                if p.exists():
                    p.unlink()

        logger.info("=== Done: %s ===", category)

    finally:
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
        try:
            delete_remote_by_rel_path(rel_lock)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _process_category_worker(args: tuple) -> None:
    """Top-level worker so ProcessPoolExecutor can pickle it."""
    (
        category,
        raw_dir,
        processed_dir,
        top_users_df,
        top_users_set,
        allow_download,
        cleanup_raw,
        cleanup_processed,
        repo_root,
        progress_interval,
    ) = args
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    process_category(
        category,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        top_users_df=top_users_df,
        top_users_set=top_users_set,
        allow_download=allow_download,
        cleanup_raw=cleanup_raw,
        cleanup_processed=cleanup_processed,
        repo_root=repo_root,
        progress_interval=progress_interval,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract top-user filtered features per category.")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--categories-file", type=str, default=None)
    parser.add_argument("--top-users", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-cleanup-raw", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument(
        "--cleanup-processed", choices=["none", "parquet", "all"], default="parquet"
    )
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    return parser.parse_args()


def read_all_categories_from_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()
    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    cleanup_raw = not (args.no_cleanup_raw or args.no_cleanup)
    cleanup_processed = args.cleanup_processed
    allow_download = not args.no_download

    resync_registry()

    if args.top_users:
        top_users_path = Path(args.top_users)
        if not top_users_path.is_absolute():
            top_users_path = REPO_ROOT / top_users_path
        if not top_users_path.exists():
            raise FileNotFoundError(f"Top users file not found: {top_users_path}")
    else:
        top_users_path = ensure_local("processed", "top_users")

    top_users_df = load_top_users(top_users_path)
    top_users_set = set(top_users_df["user_id"].astype(str).tolist())

    if args.categories:
        categories = args.categories
        logger.info("Using categories from CLI: %s", categories)
    else:
        if args.categories_file:
            cat_file = Path(args.categories_file)
            if not cat_file.is_absolute():
                cat_file = REPO_ROOT / cat_file
            if not cat_file.exists():
                rel = str(cat_file.relative_to(REPO_ROOT))
                logger.info("categories file not local; trying Drive for %s", rel)
                cat_file = ensure_local_path(rel)
        else:
            cat_file = ensure_local("raw", "all_categories.txt")

        categories = read_all_categories_from_file(cat_file)
        logger.info("Loaded %d categories from %s", len(categories), cat_file)

    total = len(categories)
    worker_args = [
        (
            cat,
            raw_dir,
            processed_dir,
            top_users_df,
            top_users_set,
            allow_download,
            cleanup_raw,
            cleanup_processed,
            REPO_ROOT,
            args.progress_interval,
        )
        for cat in categories
    ]
    max_workers = min(os.cpu_count() or 4, total)
    logger.info("Processing %d categories with %d workers.", total, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_category_worker, wargs): wargs[0] for wargs in worker_args
        }
        for f in as_completed(futures):
            cat = futures[f]
            try:
                f.result()
            except Exception as e:
                logger.error("Error processing %s: %s", cat, e)


if __name__ == "__main__":
    main()
