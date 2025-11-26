#!/usr/bin/env python
"""
Extract per-review and per-user features for top users (per-category).

For each category (or a list of categories):
  - Load `data/global/top_users.parquet` and build the set of top users
  - Stream the category review .jsonl.gz and keep only reviews by top users
  - Produce per-review feature table:
      `data/processed/<Category>/top_user_reviews_<Category>.parquet`
    Columns include: `user_id`, `product_id` (asin), `unixReviewTime`, `reviewTime`,
    `rating`, `helpful_votes`, `verified_purchase`, `review_id`
  - Produce per-user aggregated features for top users relevant to the category:
      `data/processed/<Category>/top_user_features_<Category>.parquet`
    Columns include: `user_id`, `num_purchases_in_category`, plus any global
    stats present in the supplied `top_users.parquet` (e.g. `total_purchases`).
  - Save basic EDA outputs (rating/helpful histograms) for the filtered reviews.

This script is intentionally similar in structure to `01_build_user_purchase_counts.py`.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Plot helpers (simple copies tailored for filtered data)
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
        help="Path to all_categories.txt (default: data/raw/all_categories.txt under repo).",
    )
    parser.add_argument(
        "--top-users",
        type=str,
        default=None,
        help="Path to top_users.parquet (default: data/global/top_users.parquet).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't attempt to download missing raw files; fail if not present.",
    )
    return parser.parse_args()


def read_all_categories_from_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_top_users(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ensure we have a `user_id` column
    if "user_id" in df.columns:
        return df
    # sometimes parquet preserves index as the index; reset it
    df = df.reset_index()
    if "user_id" not in df.columns:
        # fall back to first column as user id
        df = df.rename(columns={df.columns[0]: "user_id"})
    return df


def process_category(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    top_users_df: pd.DataFrame,
    allow_download: bool,
):
    print(f"\n=== Category (top-users filtered): {category} ===")

    review_gz = raw_dir / "reviews" / f"{category}.jsonl.gz"
    review_url = REVIEW_URL_TEMPLATE.format(category=category)

    if not review_gz.exists():
        if allow_download:
            download_if_needed(review_url, review_gz)
        else:
            raise FileNotFoundError(
                f"Missing review gz for {category}: {review_gz}"
            )

    cat_proc_dir = processed_dir / category
    cat_proc_dir.mkdir(parents=True, exist_ok=True)

    out_reviews_parquet = cat_proc_dir / f"top_user_reviews_{category}.parquet"
    out_user_features = cat_proc_dir / f"top_user_features_{category}.parquet"
    out_item_features = cat_proc_dir / f"top_item_features_{category}.parquet"
    out_stats_json = cat_proc_dir / f"top_user_review_stats_{category}.json"
    rating_png = cat_proc_dir / f"top_users_rating_hist_{category}.png"
    helpful_png = cat_proc_dir / f"top_users_helpful_hist_{category}.png"

    top_users_set = set(top_users_df["user_id"].astype(str).tolist())
    print(
        f"  [info] top_users provided: {len(top_users_set)} users; filtering reviews..."
    )

    # If outputs exist, skip heavy work (but still report)
    if (
        out_reviews_parquet.exists()
        and out_user_features.exists()
        and out_stats_json.exists()
    ):
        print("  [skip] All outputs exist; loading and skipping parsing.")
        return

    rows = []
    per_user_counts = defaultdict(int)
    rating_hist = Counter()
    helpful_hist = Counter()

    # Try to load meta file for this category so we can attach per-item metadata
    item_meta = {}
    meta_gz = raw_dir / "meta" / f"meta_{category}.jsonl.gz"
    meta_url = META_URL_TEMPLATE.format(category=category)
    if not meta_gz.exists():
        if allow_download:
            # mirror behavior from script 01: attempt to download the meta file
            try:
                download_if_needed(meta_url, meta_gz)
            except Exception as e:
                print(f"  [warn] failed to download meta for {category}: {e}")

    if meta_gz.exists():
        print(
            f"  [meta] loading meta from {meta_gz} (to attach item-level avg rating + categories)"
        )
        try:
            with gzip.open(meta_gz, "rt", encoding="utf-8") as mf:
                for mline in mf:
                    mline = mline.strip()
                    if not mline:
                        continue
                    try:
                        mobj = json.loads(mline)
                    except Exception:
                        continue
                    # meta uses `parent_asin` for the canonical product id in this dataset
                    a = (
                        mobj.get("parent_asin")
                        or mobj.get("asin")
                        or mobj.get("asin")
                    )
                    if not a:
                        continue
                    # possible fields for item average rating
                    item_avg = None
                    for k in (
                        "avg_rating",
                        "average_rating",
                        "rating",
                        "rating_number",
                    ):
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
                        # raw_cats might be list of lists or list of strings
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
        except EOFError:
            print(
                f"  [warn] truncated meta gzip for {category}; skipping meta load"
            )
    else:
        print(
            f"  [meta] no meta file found for {category} at {meta_gz}; continuing without item meta"
        )

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

            # Extract fields defensively (use observed field names from Movies_and_TV)
            product_id = (
                obj.get("asin")
                or obj.get("product_id")
                or obj.get("item_id")
                or obj.get("id")
            )
            rating = obj.get("rating")
            # helpful votes: dataset uses `helpful_vote` in this category; accept both
            hv = obj.get("helpful_votes", obj.get("helpful_vote", 0))
            try:
                hv = int(hv)
            except (TypeError, ValueError):
                hv = 0
            hv_clipped = min(hv, 50)

            # timestamp candidates: Movies_and_TV uses `timestamp` (milliseconds)
            unix_time = (
                obj.get("timestamp")
                or obj.get("unixReviewTime")
                or obj.get("unix_review_time")
                or obj.get("unix_time")
            )

            verified = (
                obj.get("verified_purchase") or obj.get("verified") or False
            )

            # attach item meta if available
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
        f"  [done] filtered reviews rows={len(rows):,}, users_with_reviews={len(per_user_counts):,}"
    )

    if not rows:
        print("  [warn] No reviews for top users in this category.")
        # save empty artifacts to signal processed
        pd.DataFrame(columns=["user_id"]).to_parquet(out_reviews_parquet)
        pd.DataFrame(columns=["user_id"]).to_parquet(out_user_features)
        with open(out_stats_json, "w", encoding="utf-8") as f:
            json.dump({"n_reviews": 0, "n_users": 0}, f)
        return

    reviews_df = pd.DataFrame(rows)
    reviews_df.to_parquet(out_reviews_parquet, index=False)
    print(f"  [save] top-user reviews -> {out_reviews_parquet}")

    # Build per-user aggregated features for users present in this category
    # Compute per-user aggregated features from the filtered reviews_df
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

    # Merge with global top_users_df to include global stats
    top_users_df_local = top_users_df.copy()
    if "user_id" not in top_users_df_local.columns:
        top_users_df_local = top_users_df_local.reset_index()

    sel = top_users_df_local.merge(
        agg,
        how="right",
        left_on=top_users_df_local["user_id"].astype(str),
        right_on=agg["user_id"].astype(str),
    )
    # `merge` may produce duplicate columns; keep the `user_id` from agg and drop the synthetic key
    if "user_id_x" in sel.columns and "user_id_y" in sel.columns:
        sel = sel.drop(
            columns=[
                c for c in sel.columns if c.endswith("_x") or c.endswith("_y")
            ]
        )
    # ensure canonical user_id
    if "user_id" not in sel.columns and "key_0" in sel.columns:
        sel = sel.rename(columns={"key_0": "user_id"})

    # If merge didn't include some columns, fall back to aggregations only
    if sel.empty:
        sel = agg

    sel.to_parquet(out_user_features, index=False)
    print(f"  [save] per-user features -> {out_user_features}")

    # ------------------------------------------------------------------
    # Per-item aggregated features (from reviews by top users)
    # ------------------------------------------------------------------
    # Compute aggregates per product_id (asin)
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

    # Attach meta info where available
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

    stats_out = {
        "category": category,
        "n_reviews": len(reviews_df),
        "n_users": len(per_user_counts),
        "rating_hist": dict(rating_hist),
        "helpful_hist": dict(helpful_hist),
    }
    with open(out_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  [save] stats -> {out_stats_json}")

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


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"

    top_users_path = (
        Path(args.top_users)
        if args.top_users
        else (repo_root / "data" / "global" / "top_users.parquet")
    )
    if not top_users_path.exists():
        raise FileNotFoundError(f"Top users file not found: {top_users_path}")

    top_users_df = load_top_users(top_users_path)

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
        try:
            process_category(
                cat,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                top_users_df=top_users_df,
                allow_download=not args.no_download,
            )
        except Exception as e:
            print(f"  [error] processing {cat}: {e}")


if __name__ == "__main__":
    main()

# usage
# python common_scripts/03_user_features.py
# python common_scripts/03_user_features.py --categories Movies_and_TV Books