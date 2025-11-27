#!/usr/bin/env python
"""
Extract per-review, per-user, and per-item features for top users (per-category).

For each category (or a list of categories):
  - Load global top_users (importance scores) from data/global/top_users.parquet
    via the data registry / Google Drive (unless overridden via CLI).
  - Stream the category review .jsonl.gz and keep only reviews by top users.
  - Produce per-review feature table:
      data/processed/<Category>/top_user_reviews_<Category>.parquet
  - Produce per-user aggregated features:
      data/processed/<Category>/top_user_features_<Category>.parquet
  - Produce per-item aggregated features (from top users only):
      data/processed/<Category>/top_item_features_<Category>.parquet
  - Produce EDA outputs:
      top_user_review_stats_<Category>.json
      top_users_rating_hist_<Category>.png
      top_users_helpful_hist_<Category>.png

Step-based skipping & Drive integration:
  - Resync registry at start of script.
  - For each category:
      - Use Drive-backed lockfiles under data/locks/03_user_features/.
      - Try to hydrate processed outputs from Drive to enable skipping.
      - Skip category if all outputs exist.
      - For needed steps, ensure raw gz exist via Drive or UCSD download.
      - Upload processed outputs + lockfiles to Drive.
      - Optional cleanup of raw gz and/or processed outputs locally.
"""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Make sure we can import siblings (data_io.py) no matter where we run it from
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import (  # noqa: E402
    ensure_local,
    ensure_local_path,
    resync_registry,
    upload_to_drive,
    delete_remote_by_rel_path,
    remote_file_exists_by_rel_path,
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------
def save_rating_hist_plot(
    rating_hist: Counter, out_path: Path, title: str
) -> None:
    if not rating_hist:
        print("  [plot] no ratings to plot")
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
    print(f"  [plot] saved {out_path}")


def save_helpful_hist_plot(
    helpful_hist: Counter, out_path: Path, title: str
) -> None:
    if not helpful_hist:
        print("  [plot] no helpful_votes to plot")
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
    print(f"  [plot] saved {out_path}")


# --------------------------------------------------------------------
# Raw data URLs
# --------------------------------------------------------------------
REVIEW_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/{category}.jsonl.gz"
)

META_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{category}.jsonl.gz"
)


def download_if_needed(url: str, dest: Path, force: bool = False) -> None:
    # Keep a small download helper to mirror 01 script behavior
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  [exists] {dest}")
        return

    print(f"  [download{' (force)' if force else ''}] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("  [ok] download complete")


# --------------------------------------------------------------------
# CLI handling
# --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top-user filtered features per category."
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="List of category names to process (default: all from data/raw/all_categories.txt).",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default=None,
        help=(
            "Path to all_categories.txt (default: data/raw/all_categories.txt under repo; "
            "will be fetched from Drive via registry if needed)."
        ),
    )
    parser.add_argument(
        "--top-users",
        type=str,
        default=None,
        help=(
            "Path to top_users.parquet. "
            "If omitted, will use data registry entry processed.top_users "
            "and download from Drive if needed."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't attempt to download missing raw files from UCSD; fail if not present.",
    )
    parser.add_argument(
        "--no-cleanup-raw",
        action="store_true",
        help="If set, keep the downloaded .jsonl.gz files instead of deleting them.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Alias for --no-cleanup-raw (backwards compatibility).",
    )
    parser.add_argument(
        "--cleanup-processed",
        choices=["none", "parquet", "all"],
        default="parquet",
        help=(
            "How to clean up processed outputs after uploading to Drive. "
            "'none' = keep everything; "
            "'parquet' = remove .parquet only (default); "
            "'all' = remove parquet, JSON, and PNGs."
        ),
    )
    return parser.parse_args()


def read_all_categories_from_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------------------------
# Helpers for top_users and meta
# --------------------------------------------------------------------
def load_top_users(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ensure we have a `user_id` column
    if "user_id" in df.columns:
        return df
    df = df.reset_index()
    if "user_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "user_id"})
    return df


def load_item_meta(meta_gz: Path, category: str) -> Dict[str, Dict]:
    """
    Load item-level metadata from meta_<Category>.jsonl.gz into a dict:
      parent_asin -> {item_avg_rating, item_categories}
    """
    item_meta: Dict[str, Dict] = {}
    if not meta_gz.exists():
        print(
            f"  [meta] no meta file found for {category} at {meta_gz}; "
            "continuing without item meta"
        )
        return item_meta

    print(
        f"  [meta] loading meta from {meta_gz} "
        "(to attach item-level avg rating + categories)"
    )

    with gzip.open(meta_gz, "rt", encoding="utf-8") as mf:
        for mline in mf:
            mline = mline.strip()
            if not mline:
                continue
            mobj = json.loads(mline)

            a = (
                mobj.get("parent_asin")
                or mobj.get("asin")
                or mobj.get("asin")
            )
            if not a:
                continue

            # possible fields for item average rating
            item_avg = None
            for k in ("avg_rating", "average_rating", "rating"):
                v = mobj.get(k)
                if v is not None:
                    try:
                        item_avg = float(v)
                        break
                    except Exception:
                        pass

            # Categories can be nested lists; try to flatten
            cats = None
            raw_cats = mobj.get("categories") or mobj.get("category")
            if raw_cats:
                if isinstance(raw_cats, list):
                    flat = []
                    for el in raw_cats:
                        if isinstance(el, list):
                            flat.extend(
                                [str(x) for x in el if x is not None]
                            )
                        else:
                            flat.append(str(el))
                    cats = list(dict.fromkeys(flat))
                else:
                    cats = [str(raw_cats)]

            item_meta[str(a)] = {
                "item_avg_rating": item_avg,
                "item_categories": cats,
            }

    return item_meta


# --------------------------------------------------------------------
# Raw gz helper (Drive + UCSD)
# --------------------------------------------------------------------
def ensure_raw_gzip_or_download(
    path: Path, url: str, allow_download: bool, repo_root: Path
) -> None:
    """
    Ensure a raw gzip exists locally by:
      1) Checking local path
      2) Trying Drive via ensure_local_path (using relative path)
      3) Falling back to direct HTTP download from UCSD if allowed
    """
    if path.exists():
        return

    rel = str(path.relative_to(repo_root))
    try:
        print(f"  [data_io] trying Drive for {rel}")
        ensure_local_path(rel)
        if path.exists():
            return
    except Exception:
        # No registry entry or download failed -> fall back
        pass

    if allow_download:
        download_if_needed(url, path, force=False)
    else:
        raise FileNotFoundError(
            f"Missing {path} and --no-download is set; "
            "no Drive entry or HTTP download attempted."
        )


def ensure_outputs_from_drive(paths: List[Path], repo_root: Path) -> None:
    """
    For each expected output, if it's missing locally but exists in Drive
    (according to registry), pull it down so we can skip work.
    """
    for p in paths:
        if p.exists():
            continue
        rel = str(p.relative_to(repo_root))
        try:
            ensure_local_path(rel)
        except Exception:
            # No registry entry or download failed – ignore.
            pass


# --------------------------------------------------------------------
# Core per-category processing
# --------------------------------------------------------------------
def parse_reviews_for_top_users(
    review_gz: Path,
    top_users_set,
    item_meta: Dict[str, Dict],
    category: str,
) -> Tuple[pd.DataFrame, Counter, Counter, int]:
    """
    Stream the review JSONL.gz, filter to top users, attach item meta,
    and return:
      reviews_df, rating_hist, helpful_hist, n_users
    """
    rows = []
    per_user_counts = defaultdict(int)
    rating_hist = Counter()
    helpful_hist = Counter()

    with gzip.open(review_gz, "rt", encoding="utf-8") as f:
        for line in f:
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
                obj.get("asin")
                or obj.get("product_id")
                or obj.get("item_id")
                or obj.get("id")
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

            verified = (
                obj.get("verified_purchase") or obj.get("verified") or False
            )

            im = (
                item_meta.get(str(product_id), {})
                if product_id is not None
                else {}
            )

            row = {
                "user_id": user_id,
                "product_id": product_id,
                "unixReviewTime": int(unix_time)
                if unix_time is not None
                else None,
                "rating": float(rating) if rating is not None else None,
                "helpful_votes": hv,
                "helpful_votes_clipped": hv_clipped,
                "verified_purchase": bool(verified),
                "item_avg_rating": im.get("item_avg_rating"),
                "item_categories": im.get("item_categories"),
            }

            rows.append(row)
            per_user_counts[user_id] += 1
            try:
                if rating is not None:
                    rating_hist[float(rating)] += 1
            except (TypeError, ValueError):
                pass
            helpful_hist[hv_clipped] += 1

    print(
        f"  [done] filtered reviews rows={len(rows):,}, "
        f"users_with_reviews={len(per_user_counts):,}"
    )

    reviews_df = pd.DataFrame(rows)
    return reviews_df, rating_hist, helpful_hist, len(per_user_counts)


def process_category(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    top_users_df: pd.DataFrame,
    allow_download: bool,
    cleanup_raw: bool,
    cleanup_processed: str,
    repo_root: Path,
):
    print(f"\n=== Category (top-users filtered): {category} ===")

    # Paths and expected outputs (used for lock + skip decisions)
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

    # ----------------- Locking (Drive-aware, per-category) -------------------
    lock_dir = repo_root / "data" / "locks" / "03_user_features"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{category}.lock"
    rel_lock = str(lock_path.relative_to(repo_root))

    # If a remote lock exists, hydrate it locally (if needed)
    try:
        if remote_file_exists_by_rel_path(rel_lock) and not lock_path.exists():
            ensure_local_path(rel_lock)
    except Exception:
        # If Drive is temporarily unavailable, we'll fall back to local state
        pass

    if lock_path.exists():
        print(
            f"  [lock] Detected existing lock for {category} at {lock_path}. "
            "Checking whether category is fully completed..."
        )
        # Pull any missing outputs from Drive so we can make a real decision
        ensure_outputs_from_drive(expected_outputs, repo_root)

        if all(p.exists() for p in expected_outputs):
            # Category is fully done; clear lock and skip
            print(
                "  [lock] All expected outputs exist locally; "
                "treating category as complete and clearing lock."
            )
            try:
                lock_path.unlink()
            except OSError:
                pass
            try:
                delete_remote_by_rel_path(rel_lock)
            except Exception as e:
                print(f"  [lock] WARNING: failed to delete remote lock: {e}")
            print(f"=== Done category (completed; lock cleared): {category} ===")
            return
        else:
            # Stale lock: outputs missing, so we take over
            print(
                "  [lock] Lock exists but outputs are incomplete; "
                "assuming stale lock and taking over."
            )
            try:
                lock_path.unlink()
            except OSError:
                pass
            try:
                delete_remote_by_rel_path(rel_lock)
            except Exception as e:
                print(f"  [lock] WARNING: failed to delete remote lock: {e}")

    # If we get here, either there was no lock or we just cleared a stale one.
    # Create a fresh lock for this run and push it to Drive.
    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write("locked\n")
    try:
        upload_to_drive(lock_path)
    except Exception as e:
        print(f"  [lock] WARNING: failed to upload lock to Drive: {e}")

    try:
        review_gz = raw_dir / "reviews" / f"{category}.jsonl.gz"
        meta_gz = raw_dir / "meta" / f"meta_{category}.jsonl.gz"
        review_url = REVIEW_URL_TEMPLATE.format(category=category)
        meta_url = META_URL_TEMPLATE.format(category=category)

        cat_proc_dir = processed_dir / category
        cat_proc_dir.mkdir(parents=True, exist_ok=True)

        # Outputs
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

        # Try to pull processed outputs from Drive if they exist there
        ensure_outputs_from_drive(expected_outputs, repo_root)

        # Category-level skip if EVERYTHING exists
        if all(p.exists() for p in expected_outputs):
            print("  [skip] all processed outputs exist for this category")
            # Optional cleanup of raw gz if requested
            if cleanup_raw:
                for p in (review_gz, meta_gz):
                    if p.exists():
                        print(f"  [cleanup-raw] removing {p}")
                        p.unlink()
            print(f"=== Done category (skipped): {category} ===")
            return

        # Determine which steps we actually need
        need_step1 = not (
            out_reviews_parquet.exists() and out_stats_json.exists()
        )
        need_step2 = not out_user_features.exists()
        need_step3 = not out_item_features.exists()
        need_step4 = not (rating_png.exists() and helpful_png.exists())

        # We need the review gzip if we are going to re-parse
        if need_step1:
            ensure_raw_gzip_or_download(
                review_gz, review_url, allow_download, repo_root
            )

        # We need meta if we are going to parse or create item-level aggregates
        need_meta = need_step1 or need_step3
        item_meta: Dict[str, Dict] = {}

        if need_meta:
            ensure_raw_gzip_or_download(
                meta_gz, meta_url, allow_download, repo_root
            )
            if meta_gz.exists():
                try:
                    item_meta = load_item_meta(meta_gz, category)
                except EOFError:
                    print(
                        f"  [warn] truncated meta gzip for {category}; "
                        "re-downloading and retrying..."
                    )
                    download_if_needed(meta_url, meta_gz, force=True)
                    item_meta = load_item_meta(meta_gz, category)

        # Build top_users set
        top_users_set = set(top_users_df["user_id"].astype(str).tolist())
        print(
            f"  [info] top_users provided: {len(top_users_set)} users; filtering reviews..."
        )

        # ------------------------------------------------------------------
        # Step 1: filtered reviews + stats
        # ------------------------------------------------------------------
        if not need_step1:
            print("  [skip] filtered reviews + stats already exist; loading from disk.")
            reviews_df = pd.read_parquet(out_reviews_parquet)
            with open(out_stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            rating_hist = Counter(
                {float(k): v for k, v in stats.get("rating_hist", {}).items()}
            )
            helpful_hist = Counter(
                {int(k): v for k, v in stats.get("helpful_hist", {}).items()}
            )
        else:
            print("  [run] parsing review gz for top users...")
            try:
                reviews_df, rating_hist, helpful_hist, n_users = (
                    parse_reviews_for_top_users(
                        review_gz, top_users_set, item_meta, category
                    )
                )
            except EOFError:
                print(
                    f"  [warn] truncated review gzip for {category}; "
                    "re-downloading and retrying..."
                )
                download_if_needed(review_url, review_gz, force=True)
                reviews_df, rating_hist, helpful_hist, n_users = (
                    parse_reviews_for_top_users(
                        review_gz, top_users_set, item_meta, category
                    )
                )

            # Save even if empty to signal that this category is processed
            reviews_df.to_parquet(out_reviews_parquet, index=False)
            print(f"  [save] top-user reviews -> {out_reviews_parquet}")

            stats_out = {
                "category": category,
                "n_reviews": int(len(reviews_df)),
                "n_users": int(n_users),
                "rating_hist": dict(rating_hist),
                "helpful_hist": dict(helpful_hist),
            }
            with open(out_stats_json, "w", encoding="utf-8") as f:
                json.dump(stats_out, f, indent=2)
            print(f"  [save] stats -> {out_stats_json}")

        # If there are zero reviews, short-circuit steps 2–4
        if reviews_df.empty:
            print("  [warn] No reviews for top users in this category.")
            if not out_user_features.exists():
                pd.DataFrame(columns=["user_id"]).to_parquet(
                    out_user_features, index=False
                )
                print(
                    f"  [save] empty per-user features -> {out_user_features}"
                )
            if not out_item_features.exists():
                pd.DataFrame(columns=["product_id"]).to_parquet(
                    out_item_features, index=False
                )
                print(
                    f"  [save] empty per-item features -> {out_item_features}"
                )
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
            # Cleanup raw gz if requested
            if cleanup_raw:
                for p in (review_gz, meta_gz):
                    if p.exists():
                        print(f"  [cleanup-raw] removing {p}")
                        p.unlink()
            print(f"=== Done category (no top-user reviews): {category} ===")
            return

        # ------------------------------------------------------------------
        # Step 2: per-user aggregated features
        # ------------------------------------------------------------------
        if not need_step2:
            print(
                f"  [skip] per-user features already exist: {out_user_features}"
            )
        else:
            print("  [run] computing per-user aggregated features...")
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

            top_users_df_local = top_users_df.copy()
            if "user_id" not in top_users_df_local.columns:
                top_users_df_local = top_users_df_local.reset_index()

            top_users_df_local["user_id"] = top_users_df_local[
                "user_id"
            ].astype(str)
            agg["user_id"] = agg["user_id"].astype(str)

            sel = agg.merge(top_users_df_local, on="user_id", how="left")

            sel.to_parquet(out_user_features, index=False)
            print(f"  [save] per-user features -> {out_user_features}")

        # ------------------------------------------------------------------
        # Step 3: per-item aggregated features
        # ------------------------------------------------------------------
        if not need_step3:
            print(
                f"  [skip] per-item features already exist: {out_item_features}"
            )
        else:
            print("  [run] computing per-item aggregated features...")
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

            def _get_meta_avg(asin):
                return item_meta.get(str(asin), {}).get("item_avg_rating")

            def _get_meta_cats(asin):
                return item_meta.get(str(asin), {}).get("item_categories")

            item_agg["item_avg_rating_meta"] = item_agg["product_id"].map(
                _get_meta_avg
            )
            item_agg["item_categories_meta"] = item_agg["product_id"].map(
                _get_meta_cats
            )

            item_agg.to_parquet(out_item_features, index=False)
            print(f"  [save] per-item features -> {out_item_features}")

        # ------------------------------------------------------------------
        # Step 4: EDA plots from rating/helpful hist
        # ------------------------------------------------------------------
        if not rating_png.exists() or not helpful_png.exists():
            print("  [run] creating EDA plots for top-user reviews...")
        if not rating_png.exists():
            save_rating_hist_plot(
                rating_hist,
                rating_png,
                title=f"Top users rating distribution ({category})",
            )
        else:
            print(f"  [skip] rating hist plot already exists: {rating_png}")

        if not helpful_png.exists():
            save_helpful_hist_plot(
                helpful_hist,
                helpful_png,
                title=f"Top users helpful votes ({category})",
            )
        else:
            print(f"  [skip] helpful votes plot already exists: {helpful_png}")

        # ------------------------------------------------------------------
        # Upload processed artifacts to Drive
        # ------------------------------------------------------------------
        upload_targets = [
            out_reviews_parquet,
            out_user_features,
            out_item_features,
            out_stats_json,
            rating_png,
            helpful_png,
        ]
        print("  [drive] uploading processed outputs to Drive...")
        for p in upload_targets:
            try:
                upload_to_drive(p)
            except Exception as e:
                print(f"  [drive] WARNING: failed to upload {p}: {e}")

        # ------------------------------------------------------------------
        # Cleanup processed outputs (optional)
        # ------------------------------------------------------------------
        if cleanup_processed != "none":
            for p in upload_targets:
                if cleanup_processed == "parquet" and p.suffix != ".parquet":
                    continue
                try:
                    if p.exists():
                        print(f"  [cleanup-processed] removing {p}")
                        p.unlink()
                except OSError as e:
                    print(f"  [cleanup-processed] WARNING: failed to remove {p}: {e}")

        # Cleanup raw gz if requested
        if cleanup_raw:
            for p in (review_gz, meta_gz):
                if p.exists():
                    print(f"  [cleanup-raw] removing {p}")
                    p.unlink()

        print(f"=== Done category: {category} ===")

    finally:
        # Always try to release lock
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

        try:
            rel_lock = str(lock_path.relative_to(repo_root))
            delete_remote_by_rel_path(rel_lock)
        except Exception:
            # If Drive is down or the file is already gone, ignore
            pass


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    cleanup_raw = not (args.no_cleanup_raw or args.no_cleanup)
    cleanup_processed = args.cleanup_processed
    allow_download = not args.no_download

    # Always resync registry at the start so Drive state is fresh
    resync_registry()

    # Resolve top_users path
    if args.top_users:
        top_users_path = Path(args.top_users)
        if not top_users_path.is_absolute():
            top_users_path = repo_root / top_users_path
        if not top_users_path.exists():
            raise FileNotFoundError(
                f"Top users file not found: {top_users_path}"
            )
    else:
        # Use data registry (processed.top_users) and Google Drive if needed
        top_users_path = ensure_local("processed", "top_users")

    top_users_df = load_top_users(top_users_path)

    # Determine categories
    if args.categories:
        categories = args.categories
        print(f"Using categories from CLI: {categories}")
    else:
        if args.categories_file:
            cat_file = Path(args.categories_file)
            if not cat_file.is_absolute():
                cat_file = repo_root / cat_file
            if not cat_file.exists():
                # Try drive registry lookup by relative path
                rel = str(cat_file.relative_to(repo_root))
                print(
                    f"[info] categories file not local; trying Drive for {rel}"
                )
                cat_file = ensure_local_path(rel)
        else:
            # Default: use registry entry raw.all_categories.txt
            cat_file = ensure_local("raw", "all_categories.txt")

        categories = read_all_categories_from_file(cat_file)
        print(f"No categories specified; using all from {cat_file}")
        print(f"{len(categories)} categories: {categories}")

    for cat in categories:
        try:
            process_category(
                cat,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                top_users_df=top_users_df,
                allow_download=allow_download,
                cleanup_raw=cleanup_raw,
                cleanup_processed=cleanup_processed,
                repo_root=repo_root,
            )
        except Exception as e:
            print(f"  [error] processing {cat}: {e}")


if __name__ == "__main__":
    main()

# usage examples:
# python common_scripts/03_user_features.py
# python common_scripts/03_user_features.py --categories Movies_and_TV Books
# python common_scripts/03_user_features.py --no-cleanup-raw
# python common_scripts/03_user_features.py --cleanup-processed none
# python common_scripts/03_user_features.py --cleanup-processed all