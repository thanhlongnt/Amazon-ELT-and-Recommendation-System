#!/usr/bin/env python
"""Preprocess Amazon-Reviews-2023 categories.

For each category:
- Download / fetch review + meta ``.jsonl.gz`` files (Drive or UCSD)
- Stream and parse the review file
- Compute per-user purchase counts
- Compute basic EDA stats (ratings, helpful votes, user purchase counts)
- Stream meta file for simple item-level stats
- Save results under ``data/processed/<Category>/``

By default, local raw gz files are deleted after processing (``cleanup_raw=True``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from amazon_next_category.io.data_io import (
    ensure_local,
    ensure_local_path,
    resync_registry,
    upload_to_drive,
)
from amazon_next_category.pipeline.pipeline_utils import (
    download_if_needed,
    ensure_outputs_from_drive,
    ensure_raw_gzip_or_download,
    read_all_categories_from_file,
    save_helpful_hist_plot,
    save_rating_hist_plot,
)
from amazon_next_category.utils.config import (
    META_URL_TEMPLATE,
    REVIEW_URL_TEMPLATE,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# EDA & processing
# ---------------------------------------------------------------------------


def process_review_file(gz_path: Path, category: str, max_helpful_bucket: int = 10) -> Dict:
    """Stream-parse *gz_path* and return counts / histograms."""
    logger.info("Parsing reviews: %s", gz_path)

    user_counts: Dict[str, int] = defaultdict(int)
    rating_hist: Counter = Counter()
    helpful_hist: Counter = Counter()
    n_reviews = 0
    n_verified = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            user_id = obj.get("user_id")
            if user_id is None:
                continue

            user_counts[user_id] += 1
            n_reviews += 1

            rating = obj.get("rating")
            try:
                if rating is not None:
                    rating_hist[float(rating)] += 1
            except (TypeError, ValueError):
                pass

            hv = obj.get("helpful_votes", 0)
            try:
                hv = int(hv)
            except (TypeError, ValueError):
                hv = 0
            helpful_hist[min(hv, max_helpful_bucket)] += 1

            if obj.get("verified_purchase"):
                n_verified += 1

    logger.info("Reviews parsed: n_reviews=%d, n_users=%d", n_reviews, len(user_counts))
    return {
        "user_counts": user_counts,
        "rating_hist": rating_hist,
        "helpful_hist": helpful_hist,
        "n_reviews": n_reviews,
        "n_verified": n_verified,
    }


def process_meta_file(gz_path: Path) -> Dict:
    """Stream-parse meta gzip and return item-level stats."""
    logger.info("Parsing meta: %s", gz_path)

    n_items = 0
    rating_number_hist: Counter = Counter()
    price_count = 0
    price_sum = 0.0
    price_sum_sq = 0.0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_items += 1

            rn = obj.get("rating_number")
            try:
                if rn is not None:
                    rating_number_hist[int(rn)] += 1
            except (TypeError, ValueError):
                pass

            price = obj.get("price")
            try:
                if price is not None and price != "None":
                    p = float(price)
                    price_count += 1
                    price_sum += p
                    price_sum_sq += p * p
            except (TypeError, ValueError):
                pass

    logger.info("Meta parsed: n_items=%d", n_items)
    return {
        "n_items": n_items,
        "rating_number_hist": rating_number_hist,
        "price_count": price_count,
        "price_sum": price_sum,
        "price_sum_sq": price_sum_sq,
    }


def save_user_purchases_hist_plot(user_counts: Dict[str, int], out_path: Path, title: str) -> None:
    if not user_counts:
        logger.warning("No user counts to plot.")
        return
    vals = list(user_counts.values())
    clipped = [min(v, 50) for v in vals]

    plt.figure()
    plt.hist(clipped, bins=50)
    plt.xlabel("Purchases per user (clipped at 50)")
    plt.ylabel("Number of users")
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved user-purchases hist: %s", out_path)


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------


def process_category(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    cleanup_raw: bool,
    cleanup_processed: str,
    allow_download: bool,
    repo_root: Path,
) -> None:
    """Process a single Amazon category end-to-end."""
    logger.info("=== Category: %s ===", category)

    lock_dir = repo_root / "data" / "locks" / "01_build_user_purchase_counts"
    lock_path = lock_dir / f"{category}.lock"
    rel_lock = str(lock_path.relative_to(repo_root))

    try:
        ensure_local_path(rel_lock)
    except Exception:
        pass

    lock_dir.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        logger.info("Lock exists for %s; skipping.", category)
        return

    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write("locked\n")
    try:
        upload_to_drive(lock_path)
    except Exception as e:
        logger.warning("Failed to upload lock to Drive: %s", e)

    try:
        review_url = REVIEW_URL_TEMPLATE.format(category=category)
        meta_url = META_URL_TEMPLATE.format(category=category)

        review_gz_path = raw_dir / "reviews" / f"{category}.jsonl.gz"
        meta_gz_path = raw_dir / "meta" / f"meta_{category}.jsonl.gz"

        cat_proc_dir = processed_dir / category
        cat_proc_dir.mkdir(parents=True, exist_ok=True)

        user_counts_path = cat_proc_dir / f"user_counts_{category}.parquet"
        review_stats_path = cat_proc_dir / f"review_stats_{category}.json"
        rating_hist_png = cat_proc_dir / f"rating_hist_{category}.png"
        helpful_hist_png = cat_proc_dir / f"helpful_votes_hist_{category}.png"
        user_purchases_png = cat_proc_dir / f"user_purchases_hist_{category}.png"
        meta_stats_path = cat_proc_dir / f"meta_stats_{category}.json"

        expected_outputs = [
            user_counts_path,
            review_stats_path,
            rating_hist_png,
            helpful_hist_png,
            user_purchases_png,
            meta_stats_path,
        ]

        ensure_outputs_from_drive(expected_outputs, repo_root)

        if all(p.exists() for p in expected_outputs):
            logger.info("All outputs exist for %s; skipping.", category)
            if cleanup_raw:
                for p in (review_gz_path, meta_gz_path):
                    if p.exists():
                        logger.info("Removing raw file: %s", p)
                        p.unlink()
            return

        ensure_raw_gzip_or_download(review_gz_path, review_url, allow_download, repo_root)
        ensure_raw_gzip_or_download(meta_gz_path, meta_url, allow_download, repo_root)

        # Step 1: review stats + user counts
        user_counts_df = None
        rating_hist = None
        helpful_hist = None

        if user_counts_path.exists() and review_stats_path.exists():
            logger.info("Loading existing user_counts + review_stats for %s.", category)
            user_counts_df = pd.read_parquet(user_counts_path)
            with open(review_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            rating_hist = Counter({float(k): v for k, v in stats["rating_hist"].items()})
            helpful_hist = Counter({int(k): v for k, v in stats["helpful_hist"].items()})
        else:
            logger.info("Computing review_stats + user_counts for %s.", category)
            try:
                review_stats_raw = process_review_file(review_gz_path, category)
            except EOFError:
                logger.warning("Truncated gzip for %s; re-downloading.", category)
                download_if_needed(review_url, review_gz_path, force=True)
                review_stats_raw = process_review_file(review_gz_path, category)

            user_counts = review_stats_raw["user_counts"]
            rating_hist = review_stats_raw["rating_hist"]
            helpful_hist = review_stats_raw["helpful_hist"]

            user_counts_rows = [
                {"user_id": uid, "num_purchases": cnt, "category": category}
                for uid, cnt in user_counts.items()
            ]
            user_counts_df = pd.DataFrame(user_counts_rows)
            user_counts_df.to_parquet(user_counts_path, index=False)
            logger.info("Saved user_counts: %s", user_counts_path)

            review_stats_out = {
                "category": category,
                "n_reviews": review_stats_raw["n_reviews"],
                "n_users": len(user_counts),
                "n_verified": review_stats_raw["n_verified"],
                "rating_hist": dict(rating_hist),
                "helpful_hist": dict(helpful_hist),
            }
            with open(review_stats_path, "w", encoding="utf-8") as f:
                json.dump(review_stats_out, f, indent=2)
            logger.info("Saved review_stats: %s", review_stats_path)

        # Step 2: plots
        if not rating_hist_png.exists():
            save_rating_hist_plot(
                rating_hist, rating_hist_png, title=f"Rating distribution ({category})"
            )
        if not helpful_hist_png.exists():
            save_helpful_hist_plot(
                helpful_hist, helpful_hist_png, title=f"Helpful votes ({category})"
            )
        if not user_purchases_png.exists():
            save_user_purchases_hist_plot(
                dict(zip(user_counts_df["user_id"], user_counts_df["num_purchases"])),
                user_purchases_png,
                title=f"Purchases per user ({category})",
            )

        # Step 3: meta stats
        if meta_stats_path.exists():
            logger.info("meta_stats already exists: %s", meta_stats_path)
        else:
            logger.info("Computing meta_stats for %s.", category)
            try:
                meta_stats_raw = process_meta_file(meta_gz_path)
            except EOFError:
                logger.warning("Truncated meta gzip for %s; re-downloading.", category)
                download_if_needed(meta_url, meta_gz_path, force=True)
                meta_stats_raw = process_meta_file(meta_gz_path)

            meta_stats_out = {
                "category": category,
                "n_items": meta_stats_raw["n_items"],
                "rating_number_hist": dict(meta_stats_raw["rating_number_hist"]),
                "price_count": meta_stats_raw["price_count"],
                "price_mean": (
                    meta_stats_raw["price_sum"] / meta_stats_raw["price_count"]
                    if meta_stats_raw["price_count"] > 0
                    else None
                ),
            }
            with open(meta_stats_path, "w", encoding="utf-8") as f:
                json.dump(meta_stats_out, f, indent=2)
            logger.info("Saved meta_stats: %s", meta_stats_path)

        # Upload
        upload_targets = [
            user_counts_path,
            review_stats_path,
            rating_hist_png,
            helpful_hist_png,
            user_purchases_png,
            meta_stats_path,
        ]
        logger.info("Uploading processed outputs for %s to Drive...", category)
        for p in upload_targets:
            try:
                upload_to_drive(p)
            except Exception as e:
                logger.warning("Failed to upload %s: %s", p, e)

        # Optional cleanup
        if cleanup_processed != "none":
            for p in upload_targets:
                if cleanup_processed == "parquet" and p.suffix != ".parquet":
                    continue
                try:
                    if p.exists():
                        logger.info("Removing processed file: %s", p)
                        p.unlink()
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", p, e)

        if cleanup_raw:
            for p in (review_gz_path, meta_gz_path):
                if p.exists():
                    logger.info("Removing raw file: %s", p)
                    p.unlink()

        logger.info("=== Done: %s ===", category)

    finally:
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
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
        cleanup_raw,
        cleanup_processed,
        allow_download,
        repo_root,
    ) = args
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    process_category(
        category,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        cleanup_raw=cleanup_raw,
        cleanup_processed=cleanup_processed,
        allow_download=allow_download,
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon-Reviews-2023 categories (user counts + EDA)."
    )
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--categories-file", type=str, default=None)
    parser.add_argument("--no-cleanup-raw", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument(
        "--cleanup-processed",
        choices=["none", "parquet", "all"],
        default="parquet",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()
    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    cleanup_raw = not (args.no_cleanup_raw or args.no_cleanup)
    cleanup_processed = args.cleanup_processed
    allow_download = not args.no_download

    resync_registry()

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

    worker_args = [
        (cat, raw_dir, processed_dir, cleanup_raw, cleanup_processed, allow_download, REPO_ROOT)
        for cat in categories
    ]
    max_workers = min(os.cpu_count() or 4, len(categories))
    logger.info("Processing %d categories with %d workers.", len(categories), max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_category_worker, wargs): wargs[0] for wargs in worker_args
        }
        for f in tqdm(as_completed(futures), total=len(categories), desc="Categories"):
            cat = futures[f]
            try:
                f.result()
            except Exception as e:
                logger.error("Error processing %s: %s", cat, e)


if __name__ == "__main__":
    main()
