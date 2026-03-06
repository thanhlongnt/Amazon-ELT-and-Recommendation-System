"""Shared shard-loading helpers used by all model training scripts."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score

from amazon_next_category.utils.config import RANDOM_SEED

logger = logging.getLogger(__name__)

_IO_WORKERS = 16


def list_shard_files(shard_dir: str) -> list[str]:
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found. "
            "Run create_sequences.py first."
        )
    shard_files = [
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.endswith(".parquet") and f.startswith("sequence_user_shard=")
    ]
    shard_files.sort()
    if not shard_files:
        raise RuntimeError(f"No shard files found under {shard_dir}.")
    return shard_files


def load_split_from_shards(
    files: list[str], max_rows: int, name: str
) -> pd.DataFrame:
    logger.info(
        "Loading %s split from %d shard files (max_rows=%s)...",
        name, len(files), max_rows,
    )

    # Determine which files to load in full and whether the boundary file needs sampling.
    # Read only parquet footer metadata (no column data) to get row counts cheaply.
    if max_rows is not None:
        n_workers = min(len(files), _IO_WORKERS)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            row_counts = list(ex.map(lambda f: pq.read_metadata(f).num_rows, files))

        full_files: list[str] = []
        partial_file: str | None = None
        partial_n: int = 0
        cumsum = 0
        for fpath, count in zip(files, row_counts):
            if cumsum >= max_rows:
                break
            remaining = max_rows - cumsum
            if count <= remaining:
                full_files.append(fpath)
                cumsum += count
            else:
                partial_file = fpath
                partial_n = remaining
                break
    else:
        full_files = list(files)
        partial_file = None
        partial_n = 0

    # Load full files in parallel.
    dfs: list[pd.DataFrame] = []
    if full_files:
        n_workers = min(len(full_files), _IO_WORKERS)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            dfs = list(ex.map(pd.read_parquet, full_files))

    # Load and sample the boundary file (if any) on the main thread.
    if partial_file is not None:
        df_partial = pd.read_parquet(partial_file)
        dfs.append(df_partial.sample(n=partial_n, random_state=RANDOM_SEED))

    if not dfs:
        raise RuntimeError(f"No rows loaded for split '{name}'.")
    df_out = pd.concat(dfs, ignore_index=True)
    logger.info("Final %s size: %d rows", name, len(df_out))
    return df_out


def split_shards(
    shard_files: list[str], train_split: float, val_split: float, random_seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle shards and split into train / val / test lists."""
    rng = np.random.RandomState(random_seed)
    arr = np.array(shard_files)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(train_split * n)
    n_val = int(val_split * n)
    train = list(arr[:n_train])
    val = list(arr[n_train:n_train + n_val])
    test = list(arr[n_train + n_val:])
    logger.info("Shard split — train: %d, val: %d, test: %d", len(train), len(val), len(test))
    return train, val, test


def validate_split_columns(
    splits: list[tuple[str, pd.DataFrame]],
    *,
    cast_target: bool = True,
) -> None:
    """Check required columns exist and optionally cast target to int."""
    required = ["user_id", "target_category_idx", "target_category"]
    for name, df in splits:
        for col in required:
            if col not in df.columns:
                raise ValueError(f"[{name}] Missing required column '{col}'")
        if cast_target:
            df["target_category_idx"] = df["target_category_idx"].astype(int)


def log_baselines(df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
    """Log majority-class and heuristic baselines on the val split."""
    y_val = df_val["target_category_idx"].astype(int).to_numpy()
    majority = df_train["target_category_idx"].value_counts().idxmax()
    acc_maj = accuracy_score(y_val, np.full_like(y_val, majority))
    logger.info("Val accuracy (global majority, idx=%d): %.4f", majority, acc_maj)
    if "last_category_idx" in df_val.columns:
        acc = accuracy_score(y_val, df_val["last_category_idx"].astype(int).to_numpy())
        logger.info("Val accuracy (last_category_idx): %.4f", acc)
    if "prefix_most_freq_category_idx" in df_val.columns:
        acc = accuracy_score(
            y_val, df_val["prefix_most_freq_category_idx"].astype(int).to_numpy()
        )
        logger.info("Val accuracy (prefix_most_freq_category_idx): %.4f", acc)


def select_feature_columns(
    df_train: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return (feature_cols, cat_feature_cols, numeric_feature_cols)."""
    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols + [label_col]]
    cat_cols = [c for c in feature_cols if "category_idx" in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    logger.info(
        "Features — total: %d, numeric: %d, categorical: %d",
        len(feature_cols), len(num_cols), len(cat_cols),
    )
    return feature_cols, cat_cols, num_cols
