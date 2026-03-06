#!/usr/bin/env python
"""Build a temporal training set for next-category prediction (step 4).

Given a user's purchase/review history up to time *t*, predict the category
of their next purchase at *t+1*.

Pipeline:
1. Ensure step-3 outputs exist for all categories.
2. Shard user features and reviews by user_id to disk.
3. For each shard, build sequence samples (prefix features + target label).
4. Combine per-shard Parquet files into a single global dataset.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import amazon_next_category.io.data_io as data_io
from amazon_next_category.utils.config import N_SHARDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


class DataLoader:
    def ensure_categories_downloaded(self, categories: List[str]) -> List[str]:
        """Return the subset of *categories* for which all step-3 outputs exist."""
        successful: List[str] = []
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
                data_io.ensure_local_path(
                    f"data/processed/{cat}/top_item_features_{cat}.parquet"
                )
                successful.append(cat)
            except Exception:
                logger.warning("Step-3 outputs for %s not fully available.", cat)
                with open(missing_path, "a", encoding="utf-8") as f:
                    f.write(f"{cat}\n")

        return successful

    def load_categories_from_file(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def build_category_index(self, categories: List[str]) -> Dict[str, int]:
        """Map category name -> integer index (0 reserved for 'Unknown')."""
        cat_to_idx: Dict[str, int] = {"Unknown": 0}
        idx = 1
        for cat in categories:
            if cat not in cat_to_idx:
                cat_to_idx[cat] = idx
                idx += 1
        return cat_to_idx


# ---------------------------------------------------------------------------
# Phase 0: shard user features
# ---------------------------------------------------------------------------


def shard_user_features(
    categories: List[str],
    tmp_user_dir: str,
    n_shards: int = N_SHARDS,
) -> None:
    """Write per-category user features into a hash-partitioned Parquet dataset."""
    from pandas.util import hash_pandas_object

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(tmp_user_dir, exist_ok=True)

    for cat in categories:
        path = f"data/processed/{cat}/top_user_features_{cat}.parquet"
        if not os.path.exists(path):
            logger.warning("Missing user features parquet for %s: %s", cat, path)
            continue

        logger.info("Sharding user features for %s...", cat)
        df = pd.read_parquet(path)

        if "user_id" not in df.columns:
            logger.warning("%s user features missing user_id; skipping.", cat)
            continue

        df["user_id"] = df["user_id"].astype(str)
        shard_ids = hash_pandas_object(df["user_id"], index=False).values % n_shards
        df["user_shard"] = shard_ids.astype("int32")

        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=tmp_user_dir, partition_cols=["user_shard"])
        del df, table
        gc.collect()

    logger.info("Done creating sharded user-features dataset.")


# ---------------------------------------------------------------------------
# Phase 1: shard reviews
# ---------------------------------------------------------------------------


def shard_reviews_by_user(
    categories: List[str],
    category_index: Dict[str, int],
    tmp_dir: str,
    n_shards: int = N_SHARDS,
) -> None:
    """Write per-category reviews into a hash-partitioned Parquet dataset."""
    from pandas.util import hash_pandas_object

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(tmp_dir, exist_ok=True)

    for cat in categories:
        path = f"data/processed/{cat}/top_user_reviews_{cat}.parquet"
        if not os.path.exists(path):
            logger.warning("Missing reviews parquet for %s: %s", cat, path)
            continue

        logger.info("Sharding reviews for %s...", cat)
        df = pd.read_parquet(path)

        if "user_id" not in df.columns or "unixReviewTime" not in df.columns:
            logger.warning("%s missing user_id/unixReviewTime; skipping.", cat)
            continue

        df["user_id"] = df["user_id"].astype(str)
        df = df.dropna(subset=["unixReviewTime"])
        df["unixReviewTime"] = df["unixReviewTime"].astype("int64")

        for col, default in [
            ("rating", np.nan),
            ("helpful_votes", 0),
            ("item_avg_rating", np.nan),
            ("verified_purchase", False),
        ]:
            if col not in df.columns:
                df[col] = default

        df["category"] = cat
        df["category_idx"] = df["category"].map(lambda c: category_index.get(c, 0))

        shard_ids = hash_pandas_object(df["user_id"], index=False).values % n_shards
        df["user_shard"] = shard_ids.astype("int32")

        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=tmp_dir, partition_cols=["user_shard"])
        del df, table
        gc.collect()

    logger.info("Done creating sharded review dataset.")


def list_shard_dirs(tmp_dir: str) -> List[str]:
    """Return sorted list of ``user_shard=*`` partition directories."""
    shard_dirs = []
    if not os.path.exists(tmp_dir):
        return shard_dirs
    for entry in os.scandir(tmp_dir):
        if entry.is_dir() and entry.name.startswith("user_shard="):
            shard_dirs.append(entry.path)
    shard_dirs.sort()
    return shard_dirs


# ---------------------------------------------------------------------------
# Phase 2: build sequences per shard
# ---------------------------------------------------------------------------


def build_sequence_dataset_for_shard(
    reviews_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    category_index: Dict[str, int],
    n_latest: int = 5,
    min_prefix: int = 3,
    disable_prefix_cat_counts: bool = False,
    sample_every_k_prefix: int = 1,
) -> pd.DataFrame:
    """Build sequence samples for a single shard.

    For each user and each time step *i* (prefix length >= *min_prefix* with a
    future purchase at *i+1*), emit one training sample containing:
    - Static user features
    - Prefix-level summary statistics
    - Last-N category indices
    - Optional per-category prefix counts
    - Target: category at *i+1*
    """
    if reviews_df.empty:
        return pd.DataFrame()

    reviews_df = reviews_df.copy()
    reviews_df["user_id"] = reviews_df["user_id"].astype(str)

    required_cols = ["user_id", "unixReviewTime", "category_idx"]
    for col in required_cols:
        if col not in reviews_df.columns:
            raise ValueError(f"shard reviews_df missing required column '{col}'")

    reviews_df = reviews_df.dropna(subset=["unixReviewTime"])
    reviews_df["unixReviewTime"] = reviews_df["unixReviewTime"].astype("int64")

    for col, default in [
        ("rating", np.nan),
        ("helpful_votes", 0),
        ("item_avg_rating", np.nan),
        ("verified_purchase", False),
    ]:
        if col not in reviews_df.columns:
            reviews_df[col] = default

    if "category" not in reviews_df.columns:
        idx_to_cat = {idx: name for name, idx in category_index.items()}
        reviews_df["category"] = reviews_df["category_idx"].map(
            lambda i: idx_to_cat.get(i, "Unknown")
        )

    uf = user_features_df.copy()
    if not uf.empty:
        uf["user_id"] = uf["user_id"].astype(str)
        uf = uf.drop_duplicates(subset=["user_id"], keep="first")
        uf = uf.set_index("user_id", drop=False)
        static_feature_cols = [c for c in uf.columns if c != "user_id"]
    else:
        uf = pd.DataFrame(columns=["user_id"]).set_index("user_id", drop=False)
        static_feature_cols = []

    grouped = reviews_df.groupby("user_id", sort=False)
    num_cats = max(category_index.values()) if category_index else 0
    samples = []

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

        prefix_len = 0
        sum_rating = 0.0
        rating_count = 0
        sum_helpful = 0.0
        sum_item_avg = 0.0
        item_avg_count = 0
        cat_counts = np.zeros(num_cats + 1, dtype=np.int64)
        first_time: Optional[int] = None

        urow = uf.loc[user_id] if user_id in uf.index else None

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

            if prefix_len < min_prefix or i >= n - 1:
                continue

            if sample_every_k_prefix > 1:
                if (prefix_len - min_prefix) % sample_every_k_prefix != 0:
                    continue

            feat: Dict = {}
            feat["user_id"] = user_id

            if urow is not None:
                for col in static_feature_cols:
                    feat[col] = urow.get(col, np.nan)
            else:
                for col in static_feature_cols:
                    feat[col] = np.nan

            feat["prefix_length"] = prefix_len
            feat["prefix_timespan"] = t - first_time if first_time is not None else 0
            feat["prefix_avg_rating"] = sum_rating / rating_count if rating_count > 0 else np.nan
            feat["prefix_avg_helpful"] = sum_helpful / prefix_len if prefix_len > 0 else 0.0
            feat["prefix_avg_item_avg_rating"] = (
                sum_item_avg / item_avg_count if item_avg_count > 0 else np.nan
            )

            feat["last_category_idx"] = cidx
            feat["last_rating"] = float(r) if not pd_isna(r) else np.nan
            feat["last_helpful_votes"] = float(h) if not pd_isna(h) else 0.0
            feat["last_item_avg_rating"] = float(ia) if not pd_isna(ia) else np.nan
            feat["last_verified"] = int(v)

            for k in range(n_latest):
                j = i - k
                feat[f"last_{k + 1}_category_idx"] = int(cat_idxs[j]) if j >= 0 else 0

            if not disable_prefix_cat_counts:
                for cat_name, idx in category_items:
                    key = "prefix_cat_count_Unknown" if cat_name == "Unknown" else f"prefix_cat_count_{cat_name}"
                    feat[key] = int(cat_counts[idx])

            feat["prefix_most_freq_category_idx"] = int(cat_counts.argmax())
            feat["target_category_idx"] = int(cat_idxs[i + 1])
            feat["target_category"] = target_cats[i + 1]

            samples.append(feat)

    if not samples:
        return pd.DataFrame()
    return pd.DataFrame(samples)


# ---------------------------------------------------------------------------
# Baseline stats
# ---------------------------------------------------------------------------


class BaselineStats:
    """Streaming accumulators for global/per-user/last-category baselines."""

    def __init__(self) -> None:
        self.global_counts: Counter = Counter()
        self.user_label_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_samples: int = 0
        self.last_equal_correct: int = 0

    def update_from_shard(self, shard_df: pd.DataFrame) -> None:
        if shard_df.empty:
            return
        self.total_samples += len(shard_df)
        self.global_counts.update(shard_df["target_category"].tolist())
        for user_id, grp in shard_df.groupby("user_id"):
            counts = grp["target_category_idx"].value_counts()
            for label_idx, cnt in counts.items():
                self.user_label_counts[user_id][int(label_idx)] += int(cnt)
        self.last_equal_correct += int(
            (shard_df["last_category_idx"] == shard_df["target_category_idx"]).sum()
        )

    def log(self) -> None:
        if self.total_samples == 0:
            logger.warning("No samples; cannot compute baselines.")
            return

        majority_cat, majority_cnt = max(self.global_counts.items(), key=lambda kv: kv[1])
        majority_acc = majority_cnt / self.total_samples

        correct_user_majority = sum(
            max(counts.values()) for counts in self.user_label_counts.values() if counts
        )
        user_majority_acc = correct_user_majority / self.total_samples
        last_cat_acc = self.last_equal_correct / self.total_samples

        logger.info("Baseline — global majority (%s): %.4f", majority_cat, majority_acc)
        logger.info("Baseline — per-user majority: %.4f", user_majority_acc)
        logger.info("Baseline — last-category: %.4f", last_cat_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build temporal training data to predict next purchase category."
    )
    parser.add_argument("--categories-file", type=str, default="data/raw/all_categories.txt")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--n-latest", type=int, default=5)
    parser.add_argument("--min-prefix", type=int, default=3)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--tmp-dir", type=str, default="data/tmp/review_shards")
    parser.add_argument("--tmp-user-dir", type=str, default="data/tmp/user_shards")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/global/sequence_training_samples.parquet",
    )
    parser.add_argument("--skip-user-sharding", action="store_true")
    parser.add_argument("--skip-review-sharding", action="store_true")
    parser.add_argument("--disable-prefix-cat-counts", action="store_true")
    parser.add_argument("--sample-every-k-prefix", type=int, default=1)
    parser.add_argument("--disable-baselines", action="store_true")
    parser.add_argument(
        "--per-shard-output-dir",
        type=str,
        default="data/global/sequence_samples_by_shard",
    )
    parser.add_argument("--resume-phase2", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = build_arg_parser()
    args = parser.parse_args()

    data_io.resync_registry()
    loader = DataLoader()

    if args.categories:
        categories = args.categories
        logger.info("Using categories from CLI: %s", categories)
    else:
        categories = loader.load_categories_from_file(args.categories_file)
        logger.info("Loaded %d categories from %s", len(categories), args.categories_file)

    categories_ok = loader.ensure_categories_downloaded(categories)
    logger.info("Categories with complete step-3 outputs: %d", len(categories_ok))
    if not categories_ok:
        raise RuntimeError("No categories have the required step-3 outputs.")

    category_index = loader.build_category_index(categories_ok)
    logger.info("category -> idx mapping: %s", category_index)

    # Phase 0: shard user features
    if args.skip_user_sharding:
        if not os.path.exists(args.tmp_user_dir):
            raise RuntimeError(
                f"--skip-user-sharding set but {args.tmp_user_dir} does not exist."
            )
        logger.info("Skipping user sharding (reusing %s).", args.tmp_user_dir)
    else:
        if os.path.exists(args.tmp_user_dir):
            logger.info("Removing existing tmp_user_dir to avoid stale shards.")
            shutil.rmtree(args.tmp_user_dir)
        logger.info("=== Phase 0: Sharding user features ===")
        shard_user_features(
            categories=categories_ok, tmp_user_dir=args.tmp_user_dir, n_shards=args.n_shards
        )

    # Phase 1: shard reviews
    if args.skip_review_sharding:
        if not os.path.exists(args.tmp_dir):
            raise RuntimeError(
                f"--skip-review-sharding set but {args.tmp_dir} does not exist."
            )
        logger.info("Skipping review sharding (reusing %s).", args.tmp_dir)
    else:
        if os.path.exists(args.tmp_dir):
            logger.info("Removing existing tmp_dir to avoid stale shards.")
            shutil.rmtree(args.tmp_dir)
        logger.info("=== Phase 1: Sharding reviews ===")
        shard_reviews_by_user(
            categories=categories_ok,
            category_index=category_index,
            tmp_dir=args.tmp_dir,
            n_shards=args.n_shards,
        )

    shard_dirs = list_shard_dirs(args.tmp_dir)
    num_shards = len(shard_dirs)
    logger.info("Found %d review shard partitions.", num_shards)
    if not shard_dirs:
        raise RuntimeError("No shard directories found; sharding may have failed.")

    os.makedirs(args.per_shard_output_dir, exist_ok=True)
    baseline_stats = BaselineStats()

    logger.info("=== Phase 2: Building sequence samples per shard ===")
    for shard_idx, shard_dir in enumerate(tqdm(shard_dirs, desc="Shards"), start=1):
        shard_name = os.path.basename(shard_dir)
        logger.info("Shard %d/%d: %s", shard_idx, num_shards, shard_name)

        shard_output_path = os.path.join(
            args.per_shard_output_dir, f"sequence_{shard_name}.parquet"
        )

        if os.path.exists(shard_output_path) and args.resume_phase2:
            pf = pq.ParquetFile(shard_output_path)
            col_names = pf.schema.names
            has_prefix_counts = any(c.startswith("prefix_cat_count_") for c in col_names)

            if args.disable_prefix_cat_counts or has_prefix_counts:
                if not args.disable_baselines:
                    existing_df = pf.read().to_pandas()
                    baseline_stats.update_from_shard(existing_df)
                    del existing_df
                    gc.collect()
                continue

            logger.info("Shard %s lacks prefix_cat_count_*; rebuilding.", shard_name)

        rfiles = [
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.endswith(".parquet")
        ]
        if not rfiles:
            logger.info("No review files in shard %s; skipping.", shard_name)
            continue

        shard_dfs = [pd.read_parquet(p) for p in rfiles]
        shard_reviews = pd.concat(shard_dfs, ignore_index=True)
        del shard_dfs
        gc.collect()

        user_shard_dir = os.path.join(args.tmp_user_dir, shard_name)
        if os.path.exists(user_shard_dir):
            ufiles = [
                os.path.join(user_shard_dir, f)
                for f in os.listdir(user_shard_dir)
                if f.endswith(".parquet")
            ]
            if ufiles:
                udfs = [pd.read_parquet(p) for p in ufiles]
                shard_user_features_df = pd.concat(udfs, ignore_index=True)
                del udfs
                gc.collect()
            else:
                shard_user_features_df = pd.DataFrame(columns=["user_id"])
        else:
            shard_user_features_df = pd.DataFrame(columns=["user_id"])

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
            logger.info("No valid sequence samples for shard %s; skipping.", shard_name)
            continue

        logger.info("Shard %s: %d sequence samples.", shard_name, len(seq_df_shard))

        if not args.disable_baselines:
            baseline_stats.update_from_shard(seq_df_shard)

        table = pa.Table.from_pandas(seq_df_shard)
        pq.write_table(table, shard_output_path)
        logger.info("Wrote shard output: %s", shard_output_path)
        del seq_df_shard, table
        gc.collect()

    # Combine
    shard_files = sorted(
        [
            os.path.join(args.per_shard_output_dir, f)
            for f in os.listdir(args.per_shard_output_dir)
            if f.endswith(".parquet")
        ]
    )

    if not shard_files:
        logger.warning("No per-shard outputs found; global output will not be written.")
    else:
        logger.info("Combining %d shard files into %s ...", len(shard_files), args.output_path)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        if os.path.exists(args.output_path):
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
            logger.info("Saved %d rows -> %s", total_rows, args.output_path)
            try:
                abs_path = os.path.abspath(args.output_path)
                data_io.upload_to_drive(abs_path)
                logger.info("Uploaded sequence dataset to Drive: %s", abs_path)
            except Exception as e:
                logger.warning("Failed to upload sequence dataset: %s", e)

    if not args.disable_baselines:
        baseline_stats.log()
    else:
        logger.info("Baseline computation skipped.")


if __name__ == "__main__":
    main()
