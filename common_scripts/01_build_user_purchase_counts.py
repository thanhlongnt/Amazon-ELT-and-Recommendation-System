#!/usr/bin/env python
"""
Preprocess Amazon-Reviews-2023 categories.

For each category:
  - Download / fetch review + meta .jsonl.gz files (Drive or UCSD)
  - Stream and parse the review file
  - Compute per-user purchase counts
  - Compute basic EDA stats (ratings, helpful votes, user purchase counts)
  - Stream meta file for simple item-level stats
  - Saves:
      data/processed/<Category>/user_counts_<Category>.parquet
      data/processed/<Category>/review_stats_<Category>.json
      data/processed/<Category>/meta_stats_<Category>.json
      data/processed/<Category>/*_hist_<Category>.png
  - By default, deletes local raw gz files when done (cleanup_raw=True).
"""

# pip install requests pandas matplotlib pyarrow

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests

# Make sure we can import siblings (data_io.py)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import (  # noqa: E402
    ensure_local,
    ensure_local_path,
    resync_registry,
    upload_to_drive,
)


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

def download_if_needed(url: str, dest: Path, force: bool = False) -> None:
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

def process_category(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    cleanup_raw: bool,
    cleanup_processed: str,
    allow_download: bool,
    repo_root: Path,
) -> None:
    """
    Process a single category with granular skipping and Drive-backed locking:
      - Skip whole category if ALL expected outputs exist (locally or via Drive).
      - Otherwise, skip individual steps if their outputs exist.
      - Upload processed outputs + lockfiles to Drive.
      - Optionally delete raw gz and/or processed outputs locally.
    """
    print(f"\n=== Category: {category} ===")

    # ------------- Lock (per-script, per-category, Drive-aware) -------------
    lock_dir = repo_root / "data" / "locks" / "01_build_user_purchase_counts"
    lock_path = lock_dir / f"{category}.lock"

    # Try to hydrate remote lock from Drive (if it exists) before checking
    rel_lock = str(lock_path.relative_to(repo_root))
    try:
        ensure_local_path(rel_lock)
    except Exception:
        pass

    lock_dir.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        print(
            f"  [lock] Detected existing lock for {category} at {lock_path}. "
            "Skipping this category."
        )
        return

    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write("locked\n")
    try:
        upload_to_drive(lock_path)
    except Exception as e:
        print(f"  [lock] WARNING: failed to upload lock to Drive: {e}")

    try:
        # Paths
        review_url = REVIEW_URL_TEMPLATE.format(category=category)
        meta_url = META_URL_TEMPLATE.format(category=category)

        review_gz_path = raw_dir / "reviews" / f"{category}.jsonl.gz"
        meta_gz_path = raw_dir / "meta" / f"meta_{category}.jsonl.gz"

        cat_proc_dir = processed_dir / category
        cat_proc_dir.mkdir(parents=True, exist_ok=True)

        # Expected processed outputs
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

        # Try to hydrate processed outputs from Drive
        ensure_outputs_from_drive(expected_outputs, repo_root)

        # --- Category-level skip if everything is already there ---
        if all(p.exists() for p in expected_outputs):
            print("  [skip] all processed outputs exist for this category; skipping heavy work.")
            if cleanup_raw:
                for p in (review_gz_path, meta_gz_path):
                    if p.exists():
                        print(f"  [cleanup] removing {p}")
                        p.unlink()
            print(f"=== Done category (skipped): {category} ===")
            return

        # Always ensure gz files are present before any step that might need them
        ensure_raw_gzip_or_download(
            review_gz_path, review_url, allow_download, repo_root
        )
        ensure_raw_gzip_or_download(
            meta_gz_path, meta_url, allow_download, repo_root
        )

        # -------------------------------------------------------------------
        # Step 1: ensure review_stats + user_counts exist
        # -------------------------------------------------------------------
        user_counts_df = None
        rating_hist = None
        helpful_hist = None

        if user_counts_path.exists() and review_stats_path.exists():
            print("  [skip] user_counts + review_stats already exist; loading from disk.")
            # Load from existing artifacts
            user_counts_df = pd.read_parquet(user_counts_path)
            with open(review_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            rating_hist = Counter(
                {float(k): v for k, v in stats["rating_hist"].items()}
            )
            helpful_hist = Counter(
                {int(k): v for k, v in stats["helpful_hist"].items()}
            )
        else:
            print("  [run] computing review_stats + user_counts from gz.")
            # (Re)compute from gz, with re-download protection for truncated file
            try:
                review_stats_raw = process_review_file(
                    review_gz_path, category
                )
            except EOFError:
                print(
                    f"  [warn] Detected truncated gzip for {category}, re-downloading..."
                )
                download_if_needed(review_url, review_gz_path, force=True)
                review_stats_raw = process_review_file(
                    review_gz_path, category
                )

            user_counts = review_stats_raw["user_counts"]
            rating_hist = review_stats_raw["rating_hist"]
            helpful_hist = review_stats_raw["helpful_hist"]

            # Build DataFrame for ALL users, no filtering
            user_counts_rows = [
                {"user_id": uid, "num_purchases": cnt, "category": category}
                for uid, cnt in user_counts.items()
            ]
            user_counts_df = pd.DataFrame(user_counts_rows)
            user_counts_df.to_parquet(user_counts_path, index=False)
            print(f"  [save] user_counts -> {user_counts_path}")

            # Save review stats JSON
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
            print(f"  [save] review_stats -> {review_stats_path}")

        # -------------------------------------------------------------------
        # Step 2: ensure plots exist (rating, helpful, user_purchases)
        # -------------------------------------------------------------------
        if not rating_hist_png.exists():
            save_rating_hist_plot(
                rating_hist,
                rating_hist_png,
                title=f"Rating distribution ({category})",
            )
        else:
            print(f"  [skip] rating hist plot already exists: {rating_hist_png}")

        if not helpful_hist_png.exists():
            save_helpful_hist_plot(
                helpful_hist,
                helpful_hist_png,
                title=f"Helpful votes distribution ({category})",
            )
        else:
            print(
                f"  [skip] helpful votes plot already exists: {helpful_hist_png}"
            )

        if not user_purchases_png.exists():
            # user_counts_df is guaranteed to be loaded at this point
            save_user_purchases_hist_plot(
                dict(zip(user_counts_df["user_id"], user_counts_df["num_purchases"])),
                user_purchases_png,
                title=f"Purchases per user ({category})",
            )
        else:
            print(
                f"  [skip] user purchases plot already exists: {user_purchases_png}"
            )

        # -------------------------------------------------------------------
        # Step 3: ensure meta_stats exist
        # -------------------------------------------------------------------
        if meta_stats_path.exists():
            print(f"  [skip] meta_stats already exist: {meta_stats_path}")
        else:
            print("  [run] computing meta_stats from gz.")
            try:
                meta_stats_raw = process_meta_file(meta_gz_path)
            except EOFError:
                print(
                    f"  [warn] Detected truncated meta gzip for {category}, re-downloading..."
                )
                download_if_needed(meta_url, meta_gz_path, force=True)
                meta_stats_raw = process_meta_file(meta_gz_path)

            meta_stats_out = {
                "category": category,
                "n_items": meta_stats_raw["n_items"],
                "rating_number_hist": dict(
                    meta_stats_raw["rating_number_hist"]
                ),
                "price_count": meta_stats_raw["price_count"],
                "price_mean": (
                    meta_stats_raw["price_sum"]
                    / meta_stats_raw["price_count"]
                    if meta_stats_raw["price_count"] > 0
                    else None
                ),
            }
            with open(meta_stats_path, "w", encoding="utf-8") as f:
                json.dump(meta_stats_out, f, indent=2)
            print(f"  [save] meta_stats -> {meta_stats_path}")

        # -------------------------------------------------------------------
        # Upload processed artifacts to Drive
        # -------------------------------------------------------------------
        upload_targets = [
            user_counts_path,
            review_stats_path,
            rating_hist_png,
            helpful_hist_png,
            user_purchases_png,
            meta_stats_path,
        ]
        print("  [drive] uploading processed outputs to Drive...")
        for p in upload_targets:
            try:
                upload_to_drive(p)
            except Exception as e:
                print(f"  [drive] WARNING: failed to upload {p}: {e}")

        # -------------------------------------------------------------------
        # Cleanup processed outputs (optional)
        # -------------------------------------------------------------------
        if cleanup_processed != "none":
            for p in upload_targets:
                if cleanup_processed == "parquet" and p.suffix != ".parquet":
                    continue
                # 'all' -> delete everything in upload_targets
                try:
                    if p.exists():
                        print(f"  [cleanup-processed] removing {p}")
                        p.unlink()
                except OSError as e:
                    print(f"  [cleanup-processed] WARNING: failed to remove {p}: {e}")

        # -------------------------------------------------------------------
        # Cleanup raw gz files if requested
        # -------------------------------------------------------------------
        if cleanup_raw:
            for p in (review_gz_path, meta_gz_path):
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
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't attempt to download missing raw files from UCSD; fail if not present.",
    )
    return parser.parse_args()


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
                rel = str(cat_file.relative_to(repo_root))
                print(
                    f"[info] categories file not local; trying Drive for {rel}"
                )
                cat_file = ensure_local_path(rel)
        else:
            # Use registry entry raw.all_categories.txt by default
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
                cleanup_raw=cleanup_raw,
                cleanup_processed=cleanup_processed,
                allow_download=allow_download,
                repo_root=repo_root,
            )
        except Exception as e:
            print(f"  [error] processing {cat}: {e}")


if __name__ == "__main__":
    main()

# examples:
# python common_scripts/01_build_user_purchase_counts.py --categories All_Beauty Toys_and_Games
# python common_scripts/01_build_user_purchase_counts.py --no-cleanup-raw
# python common_scripts/01_build_user_purchase_counts.py --cleanup-processed all