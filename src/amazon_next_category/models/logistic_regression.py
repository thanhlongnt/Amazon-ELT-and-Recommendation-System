#!/usr/bin/env python
"""Train multinomial logistic regression for next-category prediction.

Reads from the sharded sequence dataset produced by
:mod:`amazon_next_category.pipeline.create_sequences`.

Design:
- Shard-level train/val/test split (80/10/10).
- Cap rows per split to avoid OOM.
- Numeric features: ``SimpleImputer(median) + StandardScaler``.
- Categorical ``*_category_idx`` features: ``SimpleImputer + OneHotEncoder``.
- Derived features added before training (activity tempo, rating dynamics, etc.).
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import amazon_next_category.io.data_io as data_io
from amazon_next_category.utils.config import MAX_TRAIN_ROWS, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from amazon_next_category.utils.model_io import (
    list_shard_files,
    load_split_from_shards,
    log_baselines,
    select_feature_columns,
    split_shards,
    validate_split_columns,
)

logger = logging.getLogger(__name__)

SHARD_DIR = "data/global/sequence_samples_by_shard"
GLOBAL_OUT_PATH = "data/global/sequence_training_samples.parquet"

MAX_VAL_ROWS = 200_000
MAX_TEST_ROWS = 200_000

N_SHARDS_FOR_GLOBAL = 256


def shard_global_sequence_file(global_path: str, shard_dir: str, n_shards: int = 256) -> None:
    """Shard the global sequence Parquet by user_id in a streaming fashion."""
    import gc

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pandas.util import hash_pandas_object

    logger.info("Sharding global file '%s' into %d shards...", global_path, n_shards)

    if not os.path.exists(global_path):
        raise FileNotFoundError(f"Global sequence file not found at {global_path}")

    os.makedirs(shard_dir, exist_ok=True)
    pf = pq.ParquetFile(global_path)
    writers = {}
    total_rows = 0

    try:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            df = table.to_pandas()
            df["user_id"] = df["user_id"].astype(str)
            shard_ids = hash_pandas_object(df["user_id"], index=False).values % n_shards
            df["user_shard"] = shard_ids.astype("int32")

            for shard_id, shard_df in df.groupby("user_shard"):
                shard_df = shard_df.drop(columns=["user_shard"])
                shard_table = pa.Table.from_pandas(shard_df)
                shard_path = os.path.join(shard_dir, f"sequence_user_shard={int(shard_id)}.parquet")
                if shard_id not in writers:
                    writers[shard_id] = pq.ParquetWriter(shard_path, shard_table.schema)
                writers[shard_id].write_table(shard_table)

            total_rows += len(df)
            del df, table
            gc.collect()
    finally:
        for writer in writers.values():
            writer.close()

    logger.info("Sharding complete. Total rows: %d", total_rows)


def ensure_shards_from_global_if_needed() -> None:
    existing_files = []
    if os.path.exists(SHARD_DIR):
        existing_files = [
            f
            for f in os.listdir(SHARD_DIR)
            if f.endswith(".parquet") and f.startswith("sequence_user_shard=")
        ]
    if existing_files:
        logger.info("Found %d existing shard files in %s; reusing.", len(existing_files), SHARD_DIR)
        return
    if os.path.exists(GLOBAL_OUT_PATH):
        shard_global_sequence_file(GLOBAL_OUT_PATH, SHARD_DIR, n_shards=N_SHARDS_FOR_GLOBAL)
    else:
        raise FileNotFoundError(
            f"No shard files and global file '{GLOBAL_OUT_PATH}' does not exist. "
            "Run create_sequences.py first."
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features computed from existing columns."""
    df = df.copy()

    if {"prefix_length", "prefix_timespan"}.issubset(df.columns):
        span_days = df["prefix_timespan"].astype(float) / (3600.0 * 24.0)
        span_days = span_days.replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = df["prefix_length"].astype(float) / span_days
        df["prefix_reviews_per_day"] = rate.replace([np.inf, -np.inf], np.nan)

    if "last_rating" in df.columns and "prefix_avg_rating" in df.columns:
        df["last_minus_prefix_avg_rating"] = df["last_rating"] - df["prefix_avg_rating"]

    if "last_item_avg_rating" in df.columns and "prefix_avg_item_avg_rating" in df.columns:
        df["last_item_avg_minus_prefix_item_avg"] = (
            df["last_item_avg_rating"] - df["prefix_avg_item_avg_rating"]
        )

    cat_count_cols = [c for c in df.columns if c.startswith("prefix_cat_count_")]
    if cat_count_cols:
        counts = df[cat_count_cols].to_numpy(dtype=np.float32)
        total_counts = counts.sum(axis=1)
        max_counts = counts.max(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            dom_ratio = np.where(total_counts > 0, max_counts / total_counts, np.nan)
        df["prefix_cat_dom_ratio"] = dom_ratio

        total_safe = np.where(total_counts > 0, total_counts, 1.0)
        probs = counts / total_safe[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(probs, where=(probs > 0))
        entropy = -np.nansum(probs * log_probs, axis=1)
        df["prefix_cat_entropy_dynamic"] = entropy

        num_cats = counts.shape[1]
        if num_cats > 1:
            df["prefix_cat_norm_entropy_dynamic"] = entropy / np.log(num_cats)
        else:
            df["prefix_cat_norm_entropy_dynamic"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    np.random.seed(RANDOM_SEED)

    had_global_before = os.path.exists(GLOBAL_OUT_PATH)
    logger.info("Resyncing registry and ensuring global sequence dataset is local...")
    try:
        data_io.resync_registry()
        data_io.ensure_local_path(GLOBAL_OUT_PATH)
    except Exception as e:
        logger.warning("Could not ensure %s: %s", GLOBAL_OUT_PATH, e)

    downloaded_global = os.path.exists(GLOBAL_OUT_PATH) and not had_global_before
    if downloaded_global and os.path.exists(SHARD_DIR):
        logger.info("Freshly downloaded global file; removing stale shards.")
        shutil.rmtree(SHARD_DIR)

    ensure_shards_from_global_if_needed()

    shard_files = list_shard_files(SHARD_DIR)
    logger.info("Found %d shard files.", len(shard_files))
    train_files, val_files, test_files = split_shards(
        shard_files, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
    )
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")
    validate_split_columns([("train", df_train), ("val", df_val), ("test", df_test)])

    df_train = add_derived_features(df_train)
    df_val = add_derived_features(df_val)
    df_test = add_derived_features(df_test)

    log_baselines(df_train, df_val)
    feature_cols, cat_feature_cols, numeric_feature_cols = select_feature_columns(df_train)

    label_col = "target_category_idx"
    X_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].copy()
    X_val = df_val[feature_cols].copy()
    y_val = df_val[label_col].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[label_col].copy()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_feature_cols),
            ("cat", categorical_transformer, cat_feature_cols),
        ]
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=400,
        n_jobs=-1,
        verbose=1,
    )
    model = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    logger.info("Fitting logistic regression on train split...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    def evaluate_split(name: str, X: pd.DataFrame, y_true: np.ndarray) -> float:
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        logger.info("Accuracy (%s): %.4f", name, acc)
        try:
            y_proba = model.predict_proba(X)
            top3 = top_k_accuracy_score(y_true, y_proba, k=3)
            logger.info("Top-3 accuracy (%s): %.4f", name, top3)
        except Exception as e:
            logger.warning("Could not compute top-3 for %s: %s", name, e)
        return acc

    train_acc = evaluate_split("train", X_train, y_train.to_numpy())
    val_acc = evaluate_split("val", X_val, y_val.to_numpy())
    test_acc = evaluate_split("test", X_test, y_test.to_numpy())

    logger.info("Summary — train: %.4f, val: %.4f, test: %.4f", train_acc, val_acc, test_acc)

    y_test_pred = model.predict(X_test)
    logger.info(
        "Classification report (test):\n%s", classification_report(y_test, y_test_pred, digits=4)
    )


if __name__ == "__main__":
    main()
