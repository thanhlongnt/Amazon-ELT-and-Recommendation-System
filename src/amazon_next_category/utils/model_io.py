"""Shared shard-loading helpers used by all model training scripts."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq

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
