#!/usr/bin/env python
"""
Preprocess Amazon-Reviews-2023 categories.

For each category:
  - Download review + meta .jsonl.gz files
  - Stream and parse the review file
  - Compute per-user purchase counts
  - Compute basic EDA stats (ratings, helpful votes, user purchase counts)
  - Streams meta file for simple item-level stats
  - Saves:
      data/processed/<Category>/user_counts_<Category>.parquet
      data/processed/<Category>/review_stats_<Category>.json
      data/processed/<Category>/meta_stats_<Category>.json
      data/processed/<Category>/*_hist_<Category>.png
  - Leaves only the gzipped raw files in data/raw
"""

# pip install requests pandas matplotlib pyarrow

import argparse
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests


# Config / paths

REVIEW_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/{category}.jsonl.gz"
)
META_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{category}.jsonl.gz"
)


def get_repo_root() -> Path:
    """Return repository root (one level above common_scripts)."""
    return Path(__file__).resolve().parents[1]


# Download helpers

def download_if_needed(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [exists] {dest}")
        return

    print(f"  [download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("  [ok] download complete")


# ---------------------------------------------------------------------------
# EDA & processing
# ---------------------------------------------------------------------------

def process_review_file(
    gz_path: Path,
    category: str,
    max_helpful_bucket: int = 10,
) -> Dict:
    print(f"  [reviews] parsing {gz_path}")

    user_counts = defaultdict(int)
    rating_hist = Counter()
    helpful_hist = Counter()
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

            # Helpful votes
            hv = obj.get("helpful_votes", 0)
            try:
                hv = int(hv)
            except (TypeError, ValueError):
                hv = 0
            hv_clipped = min(hv, max_helpful_bucket)
            helpful_hist[hv_clipped] += 1

            # Verified purchase
            if obj.get("verified_purchase"):
                n_verified += 1

    print(f"  [reviews] n_reviews={n_reviews:,}, n_users={len(user_counts):,}")
    return {
        "user_counts": user_counts,
        "rating_hist": rating_hist,
        "helpful_hist": helpful_hist,
        "n_reviews": n_reviews,
        "n_verified": n_verified,
    }


def process_meta_file(gz_path: Path) -> Dict:
    print(f"  [meta] parsing {gz_path}")

    n_items = 0
    rating_number_hist = Counter()
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

            # rating_number
            rn = obj.get("rating_number")
            try:
                if rn is not None:
                    rn = int(rn)
                    rating_number_hist[rn] += 1
            except (TypeError, ValueError):
                pass

            # price
            price = obj.get("price")
            try:
                if price is not None and price != "None":
                    p = float(price)
                    price_count += 1
                    price_sum += p
                    price_sum_sq += p * p
            except (TypeError, ValueError):
                pass

    print(f"  [meta] n_items={n_items:,}")
    return {
        "n_items": n_items,
        "rating_number_hist": rating_number_hist,
        "price_count": price_count,
        "price_sum": price_sum,
        "price_sum_sq": price_sum_sq,
    }


# Plot helpers

def save_rating_hist_plot(rating_hist: Counter, out_path: Path, title: str) -> None:
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


def save_helpful_hist_plot(helpful_hist: Counter, out_path: Path, title: str) -> None:
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


def save_user_purchases_hist_plot(user_counts: Dict[str, int], out_path: Path, title: str) -> None:
    if not user_counts:
        print("  [plot] no user counts to plot")
        return
    vals = list(user_counts.values())
    # For readability, we can clip to, say, 50 purchases
    clipped = [min(v, 50) for v in vals]

    plt.figure()
    plt.hist(clipped, bins=50)
    plt.xlabel("Purchases per user (clipped at 50)")
    plt.ylabel("Number of users")
    plt.title(title)
    plt.yscale("log")  # user counts are very skewed
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"  [plot] saved {out_path}")


# Main per-category processing

def process_category(category: str, raw_dir: Path, processed_dir: Path) -> None:
    """
    Process a single category:
      - download raw .jsonl.gz files
      - compute user_counts & EDA stats
      - save user_counts df + stats + plots
    """
    print(f"\n=== Category: {category} ===")
    # Paths
    review_url = REVIEW_URL_TEMPLATE.format(category=category)
    meta_url = META_URL_TEMPLATE.format(category=category)

    review_gz_path = raw_dir / "reviews" / f"{category}.jsonl.gz"
    meta_gz_path = raw_dir / "meta" / f"meta_{category}.jsonl.gz"

    # Download raw files if needed
    download_if_needed(review_url, review_gz_path)
    download_if_needed(meta_url, meta_gz_path)

    # Process review file
    review_stats = process_review_file(review_gz_path, category)
    user_counts = review_stats["user_counts"]

    # Build user_counts DataFrame (ALL users, no filtering)
    user_counts_rows = [
        {"user_id": uid, "num_purchases": cnt, "category": category}
        for uid, cnt in user_counts.items()
    ]
    user_counts_df = pd.DataFrame(user_counts_rows)

    # Category-specific processed dir
    cat_proc_dir = processed_dir / category
    cat_proc_dir.mkdir(parents=True, exist_ok=True)

    # Save user_counts
    user_counts_path = cat_proc_dir / f"user_counts_{category}.parquet"
    user_counts_df.to_parquet(user_counts_path, index=False)
    print(f"  [save] user_counts -> {user_counts_path}")

    # Save review stats JSON
    review_stats_out = {
        "category": category,
        "n_reviews": review_stats["n_reviews"],
        "n_users": len(user_counts),
        "n_verified": review_stats["n_verified"],
        "rating_hist": dict(review_stats["rating_hist"]),
        "helpful_hist": dict(review_stats["helpful_hist"]),
    }
    review_stats_path = cat_proc_dir / f"review_stats_{category}.json"
    with open(review_stats_path, "w", encoding="utf-8") as f:
        json.dump(review_stats_out, f, indent=2)
    print(f"  [save] review_stats -> {review_stats_path}")

    # Plots for reviews
    save_rating_hist_plot(
        review_stats["rating_hist"],
        cat_proc_dir / f"rating_hist_{category}.png",
        title=f"Rating distribution ({category})",
    )
    save_helpful_hist_plot(
        review_stats["helpful_hist"],
        cat_proc_dir / f"helpful_votes_hist_{category}.png",
        title=f"Helpful votes distribution ({category})",
    )
    save_user_purchases_hist_plot(
        user_counts,
        cat_proc_dir / f"user_purchases_hist_{category}.png",
        title=f"Purchases per user ({category})",
    )

    # Process meta file (basic stats)
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
    meta_stats_path = cat_proc_dir / f"meta_stats_{category}.json"
    with open(meta_stats_path, "w", encoding="utf-8") as f:
        json.dump(meta_stats_out, f, indent=2)
    print(f"  [save] meta_stats -> {meta_stats_path}")

    print(f"=== Done category: {category} ===")


# Category selection

def read_all_categories_from_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon-Reviews-2023 categories (user counts + EDA)."
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="List of category names to process (default: all from all_categories.txt).",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default=None,
        help="Path to all_categories.txt (default: data/raw/all_categories.txt under repo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"

    # Determine categories
    if args.categories:
        categories = args.categories
        print(f"Using categories from CLI: {categories}")
    else:
        cat_file = (
            Path(args.categories_file)
            if args.categories_file
            else (raw_dir / "all_categories.txt")
        )
        categories = read_all_categories_from_file(cat_file)
        print(f"No categories specified; using all from {cat_file}")
        print(f"{len(categories)} categories: {categories}")

    for cat in categories:
        process_category(cat, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    main()

# python common_scripts/01_build_user_purchase_counts.py --categories All_Beauty Toys_and_Games etc