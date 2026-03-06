"""Global user collation and importance scoring (pipeline step 2).

Aggregates per-category user counts, computes entropy-based importance scores,
and extracts "top users" satisfying configurable thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import amazon_next_category.io.data_io as data_io
from amazon_next_category.utils.config import (
    IMPORTANCE_PERCENTILE,
    MIN_DISTINCT_CATEGORIES,
    MIN_TOTAL_PURCHASES,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "data" / "global"


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_entropy(counts: np.ndarray) -> float:
    """Return the Shannon entropy of a probability vector derived from *counts*."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def load_all_user_counts() -> pd.DataFrame:
    """Load all per-category ``user_counts_*.parquet`` files and concatenate.

    Downloads any missing files from Drive via the data registry before globbing.
    """
    data_io._load_registry()
    processed_ns = data_io._DATA_REGISTRY.get("processed", {})
    for key in processed_ns:
        if key.startswith("user_counts_"):
            try:
                data_io.ensure_local("processed", key)
            except Exception as e:
                logger.warning("Could not fetch %s from Drive: %s", key, e)

    files = list(BASE_DIR.glob("*/user_counts_*.parquet"))
    dfs = []
    for f in tqdm(files, desc="Loading user_counts"):
        dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True)


def aggregate_user_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-category rows into a user-category matrix with totals."""
    pivot = df.pivot_table(
        index="user_id",
        columns="category",
        values="num_purchases",
        aggfunc="sum",
        fill_value=0,
    )
    pivot["total_purchases"] = pivot.sum(axis=1)
    pivot["distinct_categories"] = (pivot.drop(columns="total_purchases") > 0).sum(axis=1)
    return pivot


def compute_user_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``entropy``, ``norm_entropy``, and ``importance`` columns to *df*."""
    category_cols = [c for c in df.columns if c not in ["total_purchases", "distinct_categories"]]

    df["entropy"] = df[category_cols].apply(lambda row: compute_entropy(row.values), axis=1)

    max_entropy = np.log(len(category_cols))
    df["norm_entropy"] = df["entropy"] / max_entropy
    df["importance"] = df["total_purchases"] * (1 + df["norm_entropy"])
    return df


def extract_top_users(
    df: pd.DataFrame,
    percentile: float = IMPORTANCE_PERCENTILE,
    min_purchases: int = MIN_TOTAL_PURCHASES,
    min_categories: int = MIN_DISTINCT_CATEGORIES,
) -> pd.DataFrame:
    """Return rows where *importance*, *total_purchases*, and *distinct_categories* pass thresholds."""
    threshold = df["importance"].quantile(percentile)
    return df[
        (df["importance"] >= threshold)
        & (df["total_purchases"] >= min_purchases)
        & (df["distinct_categories"] >= min_categories)
    ]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    """Save total-purchases and distinct-categories histograms to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    df["total_purchases"].clip(upper=50).hist(bins=50)
    plt.title("User Total Purchase Counts (clipped at 50)")
    plt.xlabel("Total Purchases")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "user_total_purchases_hist.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    df["distinct_categories"].hist(bins=50)
    plt.title("Distinct Categories per User")
    plt.xlabel("Num Categories")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "user_distinct_categories_hist.png")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading per-category user data...")
    raw = load_all_user_counts()

    logger.info("Aggregating global user table...")
    user_df = aggregate_user_data(raw)

    logger.info("Generating histograms...")
    save_histograms(user_df, OUT_DIR)

    logger.info("Computing user importance...")
    user_df = compute_user_importance(user_df)

    logger.info(
        "Extracting top users (min_categories=%d, min_purchases=%d)...",
        MIN_DISTINCT_CATEGORIES,
        MIN_TOTAL_PURCHASES,
    )
    top_users = extract_top_users(user_df)

    top_out = OUT_DIR / "top_users.parquet"
    top_users.to_parquet(top_out)
    logger.info("Saved top users -> %s", top_out)


if __name__ == "__main__":
    main()
