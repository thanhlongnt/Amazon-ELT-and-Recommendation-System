"""
Script 2: Global User Collation + User Importance Scoring
Updated per requirements:
- Save histograms for:
    * total purchases per user
    * distinct categories per user
- Extract top users that satisfy:
    * importance >= 95th percentile
    * distinct_categories >= 3
    * total_purchases >= 3
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = pathlib.Path("data/processed")
OUT_DIR = pathlib.Path("data/global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Compute entropy from category counts
# ---------------------------------------------------------
def compute_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]  # avoid log(0)
    return -(p * np.log(p)).sum()


# ---------------------------------------------------------
# Load all per-category user_count tables
# ---------------------------------------------------------
def load_all_user_counts():
    files = list(BASE_DIR.glob("*/user_counts_*.parquet"))

    dfs = []
    for f in tqdm(files, desc="Loading user_counts"):
        dfs.append(pd.read_parquet(f))

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# Aggregate into global user‐category matrix
# ---------------------------------------------------------
def aggregate_user_data(df:pd.DataFrame):
    pivot = df.pivot_table(
        index="user_id",
        columns="category",
        values="num_purchases",
        aggfunc="sum",
        fill_value=0
    )

    pivot["total_purchases"] = pivot.sum(axis=1)
    pivot["distinct_categories"] = (pivot.drop(columns="total_purchases") > 0).sum(axis=1)
    return pivot


# ---------------------------------------------------------
# Compute user diversity and importance
# ---------------------------------------------------------
def compute_user_importance(df:pd.DataFrame):
    category_cols = [c for c in df.columns if c not in ["total_purchases", "distinct_categories"]]

    df["entropy"] = df[category_cols].apply(lambda row: compute_entropy(row.values), axis=1)

    max_entropy = np.log(len(category_cols))
    df["norm_entropy"] = df["entropy"] / max_entropy

    df["importance"] = df["total_purchases"] * (1 + df["norm_entropy"])

    return df


# ---------------------------------------------------------
# Extract top users with constraints
# ---------------------------------------------------------
def extract_top_users(df:pd.DataFrame, percentile=0.95):
    importance_threshold = df["importance"].quantile(percentile)

    filtered = df[
        (df["importance"] >= importance_threshold) &
        (df["total_purchases"] >= 3) &
        (df["distinct_categories"] >= 3)
    ]

    return filtered


# ---------------------------------------------------------
# Plot histograms and save to disk
# ---------------------------------------------------------
def save_histograms(df:pd.DataFrame):
    # Total purchases hist
    plt.figure(figsize=(8, 5))
    df["total_purchases"].clip(upper=50).hist(bins=50)
    plt.title("User Total Purchase Counts (clipped at 50)")
    plt.xlabel("Total Purchases")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "user_total_purchases_hist.png")
    plt.close()

    # Distinct categories hist
    plt.figure(figsize=(8, 5))
    df["distinct_categories"].hist(bins=50)
    plt.title("Distinct Categories per User")
    plt.xlabel("Num Categories")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "user_distinct_categories_hist.png")
    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print(">>> Loading per-category user data...")
    raw = load_all_user_counts()

    print(">>> Aggregating global user table...")
    user_df = aggregate_user_data(raw)

    print(">>> Generating histograms...")
    save_histograms(user_df)

    print(">>> Computing user importance...")
    user_df = compute_user_importance(user_df)

    print(">>> Extracting top users (with min categories=3, min purchases=3)...")
    top_users = extract_top_users(user_df, percentile=0.95)

    top_out = OUT_DIR / "top_users.parquet"
    top_users.to_parquet(top_out)
    print(f"Saved top users → {top_out}")

    print("\n[ DONE ]\n")


if __name__ == "__main__":
    main()
