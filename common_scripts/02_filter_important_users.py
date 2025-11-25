"""
Script 2: Global User Collation + User Importance Scoring

Inputs:
    data/processed/<Category>/user_counts_<Category>.parquet

Outputs:
    data/global/user_importance.parquet
    data/global/top_users.parquet
"""

import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = pathlib.Path("data/processed")
OUT_DIR = pathlib.Path("data/global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Utility: Compute entropy given row of category counts
# ---------------------------------------------------------
def compute_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]   # avoid log(0)
    return -(p * np.log(p)).sum()


# ---------------------------------------------------------
# Step 1 — Load all user_counts parquet files from all categories
# ---------------------------------------------------------
def load_all_user_counts():
    all_files = list(BASE_DIR.glob("*/user_counts_*.parquet"))

    dfs = []
    for f in tqdm(all_files, desc="Loading per-category user_counts"):
        print(f.name)
        df = pd.read_parquet(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# Step 2 — Aggregate into global user/cat matrix
# ---------------------------------------------------------
def aggregate_user_data(df:pd.DataFrame):
    # Pivot to create user x category table
    pivot = df.pivot_table(
        index="user_id",
        columns="category",
        values="num_purchases",
        aggfunc="sum",
        fill_value=0
    )

    # Compute global stats
    pivot["total_purchases"] = pivot.sum(axis=1)
    pivot["distinct_categories"] = (pivot.drop(columns="total_purchases") > 0).sum(axis=1)

    return pivot


# ---------------------------------------------------------
# Step 3 — Compute entropy + importance score
# ---------------------------------------------------------
def compute_user_importance(df:pd.DataFrame):
    # Extract only category columns (exclude metrics appended later)
    category_cols = [c for c in df.columns if c not in ["total_purchases", "distinct_categories"]]

    entropies = df[category_cols].apply(
        lambda row: compute_entropy(row.values), axis=1
    )
    df["entropy"] = entropies

    max_entropy = np.log(len(category_cols))
    df["norm_entropy"] = df["entropy"] / max_entropy
    df["importance"] = df["total_purchases"] * (1 + df["norm_entropy"])

    return df


# ---------------------------------------------------------
# Step 4 — Extract top users
# ---------------------------------------------------------
def extract_top_users(df, percentile=0.95):
    threshold = df["importance"].quantile(percentile)
    top_users = df[df["importance"] >= threshold]
    return top_users


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print(">>> Loading category user data...")
    raw_df = load_all_user_counts()

    print(">>> Aggregating to global user table...")
    user_df = aggregate_user_data(raw_df)

    print(">>> Computing entropy and importance scores...")
    user_df = compute_user_importance(user_df)

    full_out = OUT_DIR / "user_importance.parquet"
    user_df.to_parquet(full_out)
    print(f"Saved full user table → {full_out}")

    print(">>> Extracting top users...")
    top_df = extract_top_users(user_df, percentile=0.95)

    top_out = OUT_DIR / "top_users.parquet"
    top_df.to_parquet(top_out)
    print(f"Saved top user list → {top_out}")
    print("\n[ DONE ] User collation + importance computation complete.\n")


if __name__ == "__main__":
    main()
