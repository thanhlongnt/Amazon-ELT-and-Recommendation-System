#!/usr/bin/env python
"""
04_create_train_data.py

Build a *temporal* training set for:
    Given a user and their purchase/review history up to time t,
    predict the CATEGORY of their next purchase at time t+1.

Pipeline:

1. Ensure per-category processed outputs from script 3 exist:
   - data/processed/<cat>/top_user_reviews_<cat>.parquet
   - data/processed/<cat>/top_user_features_<cat>.parquet
   - (top_item_features_<cat>.parquet is not strictly needed here)

2. Load:
   - Combined per-review data across categories:
       user_id, product_id, unixReviewTime, rating, helpful_votes,
       helpful_votes_clipped, verified_purchase, item_avg_rating,
       item_categories, category (= top-level Amazon category)
   - Global user-level features (importance, entropy, etc.)

3. For each user:
   - Sort their reviews by unixReviewTime.
   - For each time step i where prefix length >= min_prefix
     and there exists a purchase at i+1:
       - Build features from *prefix* (reviews 0..i):
           * static user stats (importance, entropy, ...)
           * prefix_length, prefix_timespan
           * prefix_avg_rating, prefix_avg_helpful, prefix_avg_item_avg_rating
           * last rating/helpful/item_avg_rating/verified
           * last N categories (as integer indices)
           * prefix category counts & most frequent category
       - Label = category of purchase at i+1

4. Compute baselines:
   - Global majority category
   - Per-user majority category baseline
   - "Last category" baseline

5. Save:
   - data/global/sequence_training_samples.parquet
     (contains all features + target_category + target_category_idx)
"""

import argparse
import os
import shutil
from collections import Counter, defaultdict
from typing import Dict, List

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import data_io


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------


class DataLoader:
    def ensure_categories_downloaded(self, categories: List[str]) -> List[str]:
        """
        Ensure that script-3 outputs exist locally (via Drive if needed).
        Returns the subset of categories for which all required files exist.
        Logs missing categories to data/processed/missing_categories.txt.
        """
        successful_categories = []
        missing_path = "data/processed/missing_categories.txt"
        os.makedirs(os.path.dirname(missing_path), exist_ok=True)

        for cat in categories:
            try:
                data_io.ensure_local_path(
                    f"data/processed/{cat}/top_user_reviews_{cat}.parquet"
                )
                data_io.ensure_local_path(
                    f"data/processed/{cat}/top_user_features_{cat}.parquet"
                )
                # Not strictly needed here, but we keep the check:
                data_io.ensure_local_path(
                    f"data/processed/{cat}/top_item_features_{cat}.parquet"
                )
                successful_categories.append(cat)
            except Exception:
                print(f"[warn] top-user outputs for category {cat} not fully available")
                with open(missing_path, "a", encoding="utf-8") as f:
                    f.write(f"{cat}\n")

        return successful_categories

    def load_categories_from_file(self, file_path: str) -> List[str]:
        """Load categories from a line-separated text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            categories = [line.strip() for line in f if line.strip()]
        return categories

    def build_category_index(self, categories: List[str]) -> Dict[str, int]:
        """
        Build a mapping category_name -> integer index, with 0 reserved for 'Unknown'.
        """
        cat_to_idx = {"Unknown": 0}
        idx = 1
        for cat in categories:
            if cat not in cat_to_idx:
                cat_to_idx[cat] = idx
                idx += 1
        return cat_to_idx


# ---------------------------------------------------------------------
# Phase 0: shard user features by user_id to disk
# ---------------------------------------------------------------------


def shard_user_features(
    categories: List[str],
    tmp_user_dir: str,
    n_shards: int = 256,
) -> None:
    """
    For each category, load top_user_features_<cat>.parquet and write it into a
    partitioned Parquet dataset under tmp_user_dir, partitioned by 'user_shard'.

    Logic:
      - user_shard = hash_pandas_object(user_id) % n_shards
      - We do NOT concatenate across categories in memory.
      - Duplicates for the same user across categories will land in the same shard
        and will be deduplicated later per shard.
    """
    from pandas.util import hash_pandas_object
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(tmp_user_dir, exist_ok=True)

    for cat in categories:
        path = f"data/processed/{cat}/top_user_features_{cat}.parquet"
        if not os.path.exists(path):
            print(f"[warn] missing user features parquet for {cat}: {path}")
            continue

        print(f"[user-shard] loading user features for {cat} from {path}")
        df = pd.read_parquet(path)

        if "user_id" not in df.columns:
            print(f"[warn] {cat} user features missing user_id; skipping")
            continue

        df["user_id"] = df["user_id"].astype(str)

        # Compute shard id from user_id (vectorized)
        shard_ids = hash_pandas_object(df["user_id"], index=False).values % n_shards
        df["user_shard"] = shard_ids.astype("int32")

        print(f"[user-shard] writing {len(df):,} rows for {cat} into user shards...")
        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=tmp_user_dir,
            partition_cols=["user_shard"],
        )

        del df, table
        gc.collect()

    print("[user-shard] done creating sharded user-features dataset.")


# ---------------------------------------------------------------------
# Phase 1: shard reviews by user_id to disk
# ---------------------------------------------------------------------


def shard_reviews_by_user(
    categories: List[str],
    category_index: Dict[str, int],
    tmp_dir: str,
    n_shards: int = 256,
) -> None:
    """
    For each category, load its top_user_reviews_<cat>.parquet and write it into a
    partitioned Parquet dataset under tmp_dir, partitioned by 'user_shard'.

    Logic:
      - For each review row, compute user_shard = hash(user_id) % n_shards
      - Write to Parquet dataset with partition_cols = ['user_shard'].

    Later we read one shard (partition) at a time to build sequences.
    """
    from pandas.util import hash_pandas_object
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(tmp_dir, exist_ok=True)

    for cat in categories:
        path = f"data/processed/{cat}/top_user_reviews_{cat}.parquet"
        if not os.path.exists(path):
            print(f"[warn] missing reviews parquet for {cat}: {path}")
            continue

        print(f"[review-shard] loading reviews for {cat} from {path}")
        df = pd.read_parquet(path)

        # We only need these columns; if any missing, create reasonable defaults.
        if "user_id" not in df.columns or "unixReviewTime" not in df.columns:
            print(f"[warn] {cat} missing user_id/unixReviewTime; skipping")
            continue

        df["user_id"] = df["user_id"].astype(str)
        df = df.dropna(subset=["unixReviewTime"])
        df["unixReviewTime"] = df["unixReviewTime"].astype("int64")

        if "rating" not in df.columns:
            df["rating"] = np.nan
        if "helpful_votes" not in df.columns:
            df["helpful_votes"] = 0
        if "item_avg_rating" not in df.columns:
            df["item_avg_rating"] = np.nan
        if "verified_purchase" not in df.columns:
            df["verified_purchase"] = False

        # Attach top-level category and its index
        df["category"] = cat
        df["category_idx"] = df["category"].map(
            lambda c: category_index.get(c, 0)
        )

        # Compute shard id from user_id (vectorized)
        shard_ids = hash_pandas_object(df["user_id"], index=False).values % n_shards
        df["user_shard"] = shard_ids.astype("int32")

        print(f"[review-shard] writing {len(df):,} rows for {cat} into review shards...")
        table = pa.Table.from_pandas(df)

        pq.write_to_dataset(
            table,
            root_path=tmp_dir,
            partition_cols=["user_shard"],
        )

        del df, table
        gc.collect()

    print("[review-shard] done creating sharded review dataset.")


def list_shard_dirs(tmp_dir: str) -> List[str]:
    """
    Return sorted list of shard partition directories under tmp_dir,
    e.g. tmp_dir/user_shard=0, tmp_dir/user_shard=1, ...
    """
    shard_dirs = []
    if not os.path.exists(tmp_dir):
        return shard_dirs

    for entry in os.scandir(tmp_dir):
        if entry.is_dir() and entry.name.startswith("user_shard="):
            shard_dirs.append(entry.path)

    shard_dirs.sort()
    return shard_dirs


# ---------------------------------------------------------------------
# Phase 2: build sequences per shard
# ---------------------------------------------------------------------


def build_sequence_dataset_for_shard(
    reviews_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    category_index: Dict[str, int],
    n_latest: int = 5,
    min_prefix: int = 3,
    disable_prefix_cat_counts: bool = False,
    sample_every_k_prefix: int = 1,
) -> pd.DataFrame:
    """
    Build sequence samples for a SINGLE shard:

      For each user in this shard and each time step i where prefix length >= min_prefix
      and there exists a purchase at i+1, create a training sample:

        - Features: prefix-based summary up to i (only past info)
        - Label: category at i+1

    Returns a DataFrame of samples for this shard.
    """

    if reviews_df.empty:
        return pd.DataFrame()

    reviews_df = reviews_df.copy()

    reviews_df["user_id"] = reviews_df["user_id"].astype(str)

    # Ensure required columns exist
    required_cols = ["user_id", "unixReviewTime", "category_idx"]
    for col in required_cols:
        if col not in reviews_df.columns:
            raise ValueError(f"shard reviews_df missing required column '{col}'")

    # Normalize and clean
    reviews_df = reviews_df.dropna(subset=["unixReviewTime"])
    reviews_df["unixReviewTime"] = reviews_df["unixReviewTime"].astype("int64")

    if "rating" not in reviews_df.columns:
        reviews_df["rating"] = np.nan
    if "helpful_votes" not in reviews_df.columns:
        reviews_df["helpful_votes"] = 0
    if "item_avg_rating" not in reviews_df.columns:
        reviews_df["item_avg_rating"] = np.nan
    if "verified_purchase" not in reviews_df.columns:
        reviews_df["verified_purchase"] = False
    if "category" not in reviews_df.columns:
        # Map category_idx back to name if needed
        idx_to_cat = {idx: name for name, idx in category_index.items()}
        reviews_df["category"] = reviews_df["category_idx"].map(
            lambda i: idx_to_cat.get(i, "Unknown")
        )

    # Static user features (per shard)
    uf = user_features_df.copy()
    if not uf.empty:
        uf["user_id"] = uf["user_id"].astype(str)
        uf = uf.drop_duplicates(subset=["user_id"], keep="first")
        uf = uf.set_index("user_id", drop=False)
        static_feature_cols = [c for c in uf.columns if c != "user_id"]
    else:
        uf = pd.DataFrame(columns=["user_id"]).set_index("user_id", drop=False)
        static_feature_cols = []

    # Group by user within this shard
    #reviews_df = reviews_df.sort_values(["user_id", "unixReviewTime"])
    # Group by user within this shard
    # NOTE: global sort is unnecessary (we sort per user below)
    # and was causing a Categorical bug on some shards.
    grouped = reviews_df.groupby("user_id", sort=False)

    num_cats = max(category_index.values()) if category_index else 0
    samples = []

    # Pull locals for a tiny speed boost
    np_isnan = np.isnan
    pd_isna = pd.isna
    category_items = list(category_index.items())

    for user_id, user_group in grouped:
        user_group = user_group.sort_values("unixReviewTime")
        n = len(user_group)
        if n <= min_prefix:
            continue

        times = user_group["unixReviewTime"].to_numpy()
        cat_idxs = user_group["category_idx"].to_numpy()
        ratings = user_group["rating"].to_numpy()
        helpful = user_group["helpful_votes"].to_numpy()
        item_avg = user_group["item_avg_rating"].to_numpy()
        verified = user_group["verified_purchase"].to_numpy()
        target_cats = user_group["category"].astype(str).to_numpy()

        if n == 0:
            continue

        # Prefix accumulators
        prefix_len = 0
        sum_rating = 0.0
        rating_count = 0
        sum_helpful = 0.0
        sum_item_avg = 0.0
        item_avg_count = 0
        cat_counts = np.zeros(num_cats + 1, dtype=np.int64)
        first_time = None

        # Static user features (if present)
        if user_id in uf.index:
            urow = uf.loc[user_id]
        else:
            urow = None

        for i in range(n):
            t = int(times[i])
            cidx = int(cat_idxs[i])
            r = ratings[i]
            h = helpful[i]
            ia = item_avg[i]
            v = bool(verified[i])

            prefix_len += 1
            if first_time is None:
                first_time = t
            cat_counts[cidx] += 1

            if not pd_isna(r):
                sum_rating += float(r)
                rating_count += 1
            if not pd_isna(h):
                sum_helpful += float(h)
            if not pd_isna(ia):
                sum_item_avg += float(ia)
                item_avg_count += 1

            # Need at least min_prefix purchases so far, and a future one at i+1
            if prefix_len < min_prefix or i >= n - 1:
                continue

            # Downsample prefixes: only keep every k-th prefix if k > 1
            if sample_every_k_prefix > 1:
                # Example: first eligible prefix (prefix_len == min_prefix) is kept,
                # then every k-th afterwards.
                if (prefix_len - min_prefix) % sample_every_k_prefix != 0:
                    continue

            feat = {}

            # 1) ID
            feat["user_id"] = user_id

            # 2) Static user features
            if urow is not None:
                for col in static_feature_cols:
                    feat[col] = urow.get(col, np.nan)
            else:
                for col in static_feature_cols:
                    feat[col] = np.nan

            # 3) Prefix-level summary
            feat["prefix_length"] = prefix_len
            feat["prefix_timespan"] = (
                t - first_time if first_time is not None else 0
            )
            feat["prefix_avg_rating"] = (
                sum_rating / rating_count if rating_count > 0 else np.nan
            )
            feat["prefix_avg_helpful"] = (
                sum_helpful / prefix_len if prefix_len > 0 else 0.0
            )
            feat["prefix_avg_item_avg_rating"] = (
                sum_item_avg / item_avg_count if item_avg_count > 0 else np.nan
            )

            # 4) Last purchase features at time i
            feat["last_category_idx"] = cidx
            feat["last_rating"] = float(r) if not pd_isna(r) else np.nan
            feat["last_helpful_votes"] = float(h) if not pd_isna(h) else 0.0
            feat["last_item_avg_rating"] = (
                float(ia) if not pd_isna(ia) else np.nan
            )
            feat["last_verified"] = int(v)

            # 5) Last n_latest categories (backwards from i)
            for k in range(n_latest):
                j = i - k
                if j >= 0:
                    feat[f"last_{k+1}_category_idx"] = int(cat_idxs[j])
                else:
                    feat[f"last_{k+1}_category_idx"] = 0

            # 6) Prefix category counts
            if not disable_prefix_cat_counts:
                for cat_name, idx in category_items:
                    if cat_name == "Unknown":
                        feat["prefix_cat_count_Unknown"] = int(cat_counts[idx])
                    else:
                        feat[f"prefix_cat_count_{cat_name}"] = int(cat_counts[idx])

            # 7) Most frequent category in prefix
            feat["prefix_most_freq_category_idx"] = int(cat_counts.argmax())

            # 8) Target: category at i+1
            next_idx = int(cat_idxs[i + 1])
            next_cat_name = target_cats[i + 1]

            feat["target_category_idx"] = next_idx
            feat["target_category"] = next_cat_name

            samples.append(feat)

    if not samples:
        return pd.DataFrame()

    return pd.DataFrame(samples)


# ---------------------------------------------------------------------
# Baseline stats (streaming)
# ---------------------------------------------------------------------


class BaselineStats:
    """
    Streaming baselines:

      - global majority category
      - per-user majority category
      - last-category baseline
    """

    def __init__(self):
        self.global_counts = Counter()                  # category_name -> count
        self.user_label_counts = defaultdict(Counter)   # user_id -> Counter(label_idx -> count)
        self.total_samples = 0
        self.last_equal_correct = 0

    def update_from_shard(self, shard_df: pd.DataFrame) -> None:
        if shard_df.empty:
            return

        self.total_samples += len(shard_df)

        # Global category counts (by name)
        self.global_counts.update(shard_df["target_category"].tolist())

        # Per-user label counts (by idx)
        for user_id, grp in shard_df.groupby("user_id"):
            counts = grp["target_category_idx"].value_counts()
            for label_idx, cnt in counts.items():
                self.user_label_counts[user_id][int(label_idx)] += int(cnt)

        # Last-category baseline (predict last_category_idx)
        self.last_equal_correct += int(
            (shard_df["last_category_idx"] == shard_df["target_category_idx"]).sum()
        )

    def log(self) -> None:
        if self.total_samples == 0:
            print("[baseline] no samples; cannot compute baselines.")
            return

        # Global majority
        majority_cat, majority_cnt = max(self.global_counts.items(), key=lambda kv: kv[1])
        majority_acc = majority_cnt / self.total_samples

        # Per-user majority
        correct_user_majority = 0
        for counts in self.user_label_counts.values():
            if counts:
                correct_user_majority += max(counts.values())
        user_majority_acc = correct_user_majority / self.total_samples

        # Last-category baseline
        last_cat_acc = self.last_equal_correct / self.total_samples

        print("\n[baseline] Global majority category:")
        print(f"  majority_category = {majority_cat}")
        print(f"  accuracy          = {majority_acc:.4f}")

        print("\n[baseline] Per-user majority category:")
        print(f"  accuracy          = {user_majority_acc:.4f}")

        print("\n[baseline] Last-category baseline:")
        print(f"  accuracy          = {last_cat_acc:.4f}")

        print("\n[baseline] summary:")
        print(f"  Global majority:   {majority_acc:.4f}")
        print(f"  Per-user majority: {user_majority_acc:.4f}")
        print(f"  Last-category:     {last_cat_acc:.4f}\n")


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build temporal training data (streaming) to predict next purchase category."
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default="data/raw/all_categories.txt",
        help="Path to all_categories.txt (one category name per line).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional explicit list of categories (overrides --categories-file).",
    )
    parser.add_argument(
        "--n-latest",
        type=int,
        default=5,
        help="Number of last purchases to encode as features.",
    )
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=3,
        help="Minimum prefix length to create a sample.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=256,
        help="Number of user shards for streaming processing.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="data/tmp/review_shards",
        help="Temporary directory to store sharded review data.",
    )
    parser.add_argument(
        "--tmp-user-dir",
        type=str,
        default="data/tmp/user_shards",
        help="Temporary directory to store sharded user feature data.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/global/sequence_training_samples.parquet",
        help="Where to write the final sequence dataset parquet.",
    )
    parser.add_argument(
        "--skip-user-sharding",
        action="store_true",
        help="Skip Phase 0 and reuse existing user_shards if present.",
    )
    parser.add_argument(
        "--skip-review-sharding",
        action="store_true",
        help="Skip Phase 1 and reuse existing review_shards if present.",
    )
    parser.add_argument(
        "--disable-prefix-cat-counts",
        action="store_true",
        help="If set, do NOT materialize per-category prefix counts (prefix_cat_count_*)."
    )
    parser.add_argument(
        "--sample-every-k-prefix",
        type=int,
        default=1,
        help="Only create a sequence sample for every k-th prefix (per user). "
             "k=1 means use every possible prefix; k=5 means keep ~1/5 of them."
    )
    parser.add_argument(
        "--disable-baselines",
        action="store_true",
        help="If set, skip computing streaming baselines on the full dataset."
    )
    parser.add_argument(
        "--per-shard-output-dir",
        type=str,
        default="data/global/sequence_samples_by_shard",
        help="Directory to write per-shard parquet outputs for Phase 2.",
    )
    parser.add_argument(
        "--resume-phase2",
        action="store_true",
        help="If set, Phase 2 will skip shards whose per-shard parquet already exists.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data_io.resync_registry()

    loader = DataLoader()

    # Determine categories
    if args.categories:
        categories = args.categories
        print(f"[info] using categories from CLI: {categories}")
    else:
        categories = loader.load_categories_from_file(args.categories_file)
        print(f"[info] loaded {len(categories)} categories from {args.categories_file}")

    # Ensure script-3 outputs exist locally / via Drive
    categories_ok = loader.ensure_categories_downloaded(categories)
    print(f"[info] categories with complete script-3 outputs: {len(categories_ok)}")
    if not categories_ok:
        raise RuntimeError("No categories have the required script-3 outputs.")

    # Build category index (for All_Beauty, Amazon_Fashion, etc.)
    category_index = loader.build_category_index(categories_ok)
    print(f"[info] category -> idx mapping: {category_index}")

    # Phase 0: shard user features
    if args.skip_user_sharding:
        if not os.path.exists(args.tmp_user_dir):
            raise RuntimeError(
                f"--skip-user-sharding was set but tmp_user_dir={args.tmp_user_dir} does not exist."
            )
        print(f"\n=== Phase 0: Skipping user sharding (reusing {args.tmp_user_dir}) ===")
    else:
        if os.path.exists(args.tmp_user_dir):
            print(f"[info] removing existing tmp_user_dir={args.tmp_user_dir} to avoid stale shards")
            shutil.rmtree(args.tmp_user_dir)

        print("\n=== Phase 0: Sharding user features by user_id ===")
        shard_user_features(
            categories=categories_ok,
            tmp_user_dir=args.tmp_user_dir,
            n_shards=args.n_shards,
        )

    # Phase 1: shard all reviews by user_id
    if args.skip_review_sharding:
        if not os.path.exists(args.tmp_dir):
            raise RuntimeError(
                f"--skip-review-sharding was set but tmp_dir={args.tmp_dir} does not exist."
            )
        print(f"\n=== Phase 1: Skipping review sharding (reusing {args.tmp_dir}) ===")
    else:
        if os.path.exists(args.tmp_dir):
            print(f"[info] removing existing tmp_dir={args.tmp_dir} to avoid stale shards")
            shutil.rmtree(args.tmp_dir)

        print("\n=== Phase 1: Sharding reviews by user_id ===")
        shard_reviews_by_user(
            categories=categories_ok,
            category_index=category_index,
            tmp_dir=args.tmp_dir,
            n_shards=args.n_shards,
        )

    shard_dirs = list_shard_dirs(args.tmp_dir)
    num_shards = len(shard_dirs)
    print(f"[info] found {num_shards} review shard partitions in {args.tmp_dir}")

    if not shard_dirs:
        raise RuntimeError("No shard directories found; sharding may have failed.")

    # Phase 2: For each shard, build sequence samples and stream to Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #writer = None
    baseline_stats = BaselineStats()

    print("\n=== Phase 2: Building sequence samples per shard ===")
    shard_idx = 0
    for shard_dir in tqdm(shard_dirs, desc="Shards"):
        shard_idx += 1
        shard_name = os.path.basename(shard_dir)  # e.g., "user_shard=123"
        print(f"\n[phase2] === Shard {shard_idx}/{num_shards}: {shard_name} ===")

        # Where we store this shard's sequence samples
        shard_output_path = os.path.join(
            args.per_shard_output_dir,
            f"sequence_{shard_name}.parquet",
        )

        # If resuming and this shard already has output, skip the heavy work
        if args.resume_phase2 and os.path.exists(shard_output_path):
            print(f"[phase2] shard {shard_name}: found existing {shard_output_path}, skipping sequence build.")

            # Optionally rebuild baselines from existing shard output
            if not args.disable_baselines:
                print(f"[phase2] shard {shard_name}: reloading shard for baselines...")
                existing_df = pd.read_parquet(shard_output_path)
                baseline_stats.update_from_shard(existing_df)
                print(f"[phase2] shard {shard_name}: reused {len(existing_df):,} samples for baselines.")
                del existing_df
                gc.collect()

            continue

        # Load review shard
        rfiles = [
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.endswith(".parquet")
        ]
        if not rfiles:
            print(f"[phase2] shard {shard_name}: no review parquet files, skipping.")
            continue

        print(f"[phase2] shard {shard_name}: loading {len(rfiles)} review files...")
        shard_dfs = [pd.read_parquet(p) for p in rfiles]
        shard_reviews = pd.concat(shard_dfs, ignore_index=True)
        del shard_dfs
        gc.collect()

        num_rows = len(shard_reviews)
        num_users = shard_reviews["user_id"].nunique() if "user_id" in shard_reviews.columns else 0
        print(f"[phase2] shard {shard_name}: {num_rows:,} rows, {num_users:,} users")

        # Load matching user-feature shard (if exists)
        user_shard_dir = os.path.join(args.tmp_user_dir, shard_name)
        if os.path.exists(user_shard_dir):
            ufiles = [
                os.path.join(user_shard_dir, f)
                for f in os.listdir(user_shard_dir)
                if f.endswith(".parquet")
            ]
            if ufiles:
                print(f"[phase2] shard {shard_name}: loading {len(ufiles)} user-feature files...")
                udfs = [pd.read_parquet(p) for p in ufiles]
                shard_user_features_df = pd.concat(udfs, ignore_index=True)
                del udfs
                gc.collect()
            else:
                print(f"[phase2] shard {shard_name}: no user-feature files (using empty).")
                shard_user_features_df = pd.DataFrame(columns=["user_id"])
        else:
            print(f"[phase2] shard {shard_name}: user-feature shard dir not found (using empty).")
            shard_user_features_df = pd.DataFrame(columns=["user_id"])

        print(f"[phase2] shard {shard_name}: building sequence dataset...")
        seq_df_shard = build_sequence_dataset_for_shard(
            reviews_df=shard_reviews,
            user_features_df=shard_user_features_df,
            category_index=category_index,
            n_latest=args.n_latest,
            min_prefix=args.min_prefix,
            disable_prefix_cat_counts=args.disable_prefix_cat_counts,
            sample_every_k_prefix=args.sample_every_k_prefix,
        )
        del shard_reviews, shard_user_features_df
        gc.collect()

        if seq_df_shard.empty:
            print(f"[phase2] shard {shard_name}: no valid sequence samples, skipping.")
            continue

        shard_samples = len(seq_df_shard)
        print(f"[phase2] shard {shard_name}: created {shard_samples:,} sequence samples")

        # Update baselines (optional)
        if not args.disable_baselines:
            baseline_stats.update_from_shard(seq_df_shard)
            print(f"[phase2] cumulative samples so far (for baselines): {baseline_stats.total_samples:,}")
        else:
            print(f"[phase2] cumulative samples so far (no baselines): "
                  f"{shard_samples:,} this shard")

        # Write per-shard output
        table = pa.Table.from_pandas(seq_df_shard)
        pq.write_table(table, shard_output_path)
        print(f"[phase2] shard {shard_name}: wrote per-shard output -> {shard_output_path}")

        del seq_df_shard, table
        gc.collect()

    # -----------------------------------------------------------------
    # Combine per-shard outputs into a single global Parquet file
    # -----------------------------------------------------------------
    import pyarrow.parquet as pq
    import pyarrow as pa

    shard_files = [
        os.path.join(args.per_shard_output_dir, f)
        for f in os.listdir(args.per_shard_output_dir)
        if f.endswith(".parquet")
    ]
    shard_files.sort()

    if not shard_files:
        print("[save] No per-shard outputs found; global output will not be written.")
    else:
        print(f"\n[save] combining {len(shard_files)} shard files into {args.output_path} ...")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        # If a previous global file exists (e.g., from a failed run), remove it
        if os.path.exists(args.output_path):
            print(f"[save] removing existing global output at {args.output_path} to avoid duplicates.")
            os.remove(args.output_path)

        writer = None
        total_rows = 0

        for fpath in shard_files:
            df = pd.read_parquet(fpath)
            total_rows += len(df)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(args.output_path, table.schema)
            writer.write_table(table)
            del df, table
            gc.collect()

        if writer is not None:
            writer.close()
            print(f"[save] sequence training samples -> {args.output_path} "
                  f"(total rows: {total_rows:,})")
            try:
                abs_path = os.path.abspath(args.output_path)
                data_io.upload_to_drive(abs_path)
                print(f"[drive] uploaded sequence dataset to Drive: {abs_path}")
            except Exception as e:
                print(f"[drive] WARNING: failed to upload sequence dataset: {e}")

    # Log baselines (global, over all shards)
    if not args.disable_baselines:
        baseline_stats.log()
    else:
        print("[baseline] skipped baseline computation (disable_baselines=True).")


if __name__ == "__main__":
    main()

# python common_scripts/04_create_train_data.py --resume-phase2
"""
python common_scripts/04_create_train_data.py \
--skip-user-sharding \
--skip-review-sharding \
--resume-phase2

python common_scripts/04_create_train_data.py --skip-user-sharding --skip-review-sharding --resume-phase2

python common_scripts/04_create_train_data.py --skip-user-sharding --skip-review-sharding --disable-prefix-cat-counts --sample-every-k-prefix 5 --resume-phase2
"""