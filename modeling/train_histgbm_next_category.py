#!/usr/bin/env python
"""
train_histgbm_next_category.py

Train a HistGradientBoostingClassifier to predict target_category_idx
from the sharded sequence data produced by 04_create_train_data.py.

Assumptions:
- Per-shard sequence files exist at:
      data/global/sequence_samples_by_shard/sequence_user_shard=*.parquet
  (i.e., you've already run 04_create_train_data.py once).

We:
- Split shards into train/val/test (80/10/10) at shard level
  (shards are by user_id hash, so this is effectively a user-level split).
- Cap rows per split to avoid blowing up RAM.
- Use ALL feature columns except:
    * "user_id"
    * "target_category"
    * "target_category_idx" (label)
- Numeric pipeline:
    * Fill NaNs with training medians (per feature)
    * Convert to dense float32 numpy arrays (HistGBM requires dense)
- Model:
    * HistGradientBoostingClassifier with early stopping and verbose progress.
- Metrics:
    * Accuracy on train/val/test
    * Top-3 accuracy on train/val/test
    * Classification report on TEST.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SHARD_DIR = "data/global/sequence_samples_by_shard"

RANDOM_SEED = 42

# To keep runtime + memory reasonable; bump if your machine can handle more.
MAX_TRAIN_ROWS = 300_000
MAX_VAL_ROWS   = 100_000
MAX_TEST_ROWS  = 100_000

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1  # rest -> test


# ---------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------


def list_shard_files(shard_dir: str):
    """
    Return sorted list of per-shard parquet files like:
      data/global/sequence_samples_by_shard/sequence_user_shard=0.parquet
    """
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found.\n"
            f"Make sure 04_create_train_data.py has been run with per-shard outputs."
        )

    shard_files = [
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.endswith(".parquet") and f.startswith("sequence_user_shard=")
    ]
    shard_files.sort()

    if not shard_files:
        raise RuntimeError(
            f"No shard files found under {shard_dir}. "
            f"Expected files like 'sequence_user_shard=0.parquet'."
        )
    return shard_files


def load_split_from_shards(files, max_rows: int, name: str):
    """
    Load up to max_rows from the given list of shard Parquet files.
    Randomly subsamples from shards if necessary.

    Args:
        files: list of .parquet shard paths
        max_rows: cap on total rows to load (None -> load all)
        name: split name for logging ("train"/"val"/"test")

    Returns:
        pandas.DataFrame for the split
    """
    print(
        f"\n[load:{name}] Loading split from {len(files)} shard files "
        f"(max_rows={max_rows if max_rows is not None else 'ALL'}) ..."
    )

    dfs = []
    total = 0

    for fpath in files:
        df_shard = pd.read_parquet(fpath)

        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            if len(df_shard) > remaining:
                df_shard = df_shard.sample(n=remaining, random_state=RANDOM_SEED)

        dfs.append(df_shard)
        total += len(df_shard)
        print(
            f"[load:{name}]   +{len(df_shard):,} rows from {os.path.basename(fpath)} "
            f"(cumulative: {total:,})"
        )

        if max_rows is not None and total >= max_rows:
            break

    if not dfs:
        raise RuntimeError(f"[load:{name}] No rows loaded for split '{name}'.")

    df_out = pd.concat(dfs, ignore_index=True)
    print(f"[load:{name}] Final {name} size: {len(df_out):,} rows")
    return df_out


# ---------------------------------------------------------------------
# Feature prep for HistGBM
# ---------------------------------------------------------------------


def prepare_features_for_histgbm(df_train, df_val, df_test):
    """
    - Identify feature columns (everything except id + label columns).
    - For each feature, compute train median and fill NaNs in all splits.
    - Return X_train, X_val, X_test as dense float32 arrays, plus y's.
    """
    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]

    all_cols = df_train.columns.tolist()
    feature_cols = [c for c in all_cols if c not in drop_cols + [label_col]]

    # Ensure labels are int
    df_train[label_col] = df_train[label_col].astype(int)
    df_val[label_col] = df_val[label_col].astype(int)
    df_test[label_col] = df_test[label_col].astype(int)

    # Compute medians from train only
    medians = {}
    for col in feature_cols:
        col_median = df_train[col].median()
        if pd.isna(col_median):
            # if column is all-NaN in train, just use 0.0
            col_median = 0.0
        medians[col] = col_median

    # Fill NaNs in all splits using train medians
    for df_part, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        df_sub = df_part[feature_cols]
        df_part[feature_cols] = df_sub.fillna(medians)

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_val   = df_val[feature_cols].to_numpy(dtype=np.float32)
    X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)

    y_train = df_train[label_col].to_numpy()
    y_val   = df_val[label_col].to_numpy()
    y_test  = df_test[label_col].to_numpy()

    print("\n[fe] Shapes after prep for HistGBM:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------


def evaluate_histgbm(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit and evaluate a HistGradientBoostingClassifier on train/val/test.
    Returns dict of metrics (acc + top-3 for each split).
    """
    print("\n[histgbm] Training HistGradientBoostingClassifier ...")
    model.fit(X_train, y_train)
    print("[histgbm] Training done.")

    metrics = {}

    def _eval_split(split_name, X, y_true):
        print(f"[histgbm:{split_name}] Evaluating on {split_name} (n={len(y_true):,}) ...")
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"[histgbm:{split_name}] Accuracy = {acc:.4f}")

        top3 = None
        try:
            proba = model.predict_proba(X)
            top3 = top_k_accuracy_score(y_true, proba, k=3)
            print(f"[histgbm:{split_name}] Top-3 accuracy = {top3:.4f}")
        except Exception as e:
            print(f"[histgbm:{split_name}] Could not compute top-3 accuracy: {e}")

        return acc, top3, y_pred

    train_acc, train_top3, _ = _eval_split("train", X_train, y_train)
    val_acc,   val_top3,   _ = _eval_split("val",   X_val,   y_val)
    test_acc,  test_top3,  y_test_pred = _eval_split("test",  X_test,  y_test)

    print("\n[histgbm] Summary:")
    print(f"  Train acc: {train_acc:.4f}")
    print(f"  Val   acc: {val_acc:.4f}")
    print(f"  Test  acc: {test_acc:.4f}")
    if val_top3 is not None:
        print(f"  Val   top-3: {val_top3:.4f}")
    if test_top3 is not None:
        print(f"  Test  top-3: {test_top3:.4f}")

    metrics["train_acc"] = train_acc
    metrics["val_acc"] = val_acc
    metrics["test_acc"] = test_acc
    metrics["val_top3"] = val_top3
    metrics["test_top3"] = test_top3
    metrics["y_test_pred"] = y_test_pred

    # Detailed classification report on TEST
    print("\n[histgbm:report:test] Classification report on TEST split:")
    print(classification_report(y_test, y_test_pred, digits=4))

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    np.random.seed(RANDOM_SEED)

    # 1. Discover shard files & define splits
    print(f"[shards] Listing shard files under {SHARD_DIR} ...")
    shard_files = list_shard_files(SHARD_DIR)
    n_shards = len(shard_files)
    print(f"[shards] Found {n_shards} shard files.")

    # Shuffle shards for randomness
    rng = np.random.RandomState(RANDOM_SEED)
    shard_files = np.array(shard_files)
    rng.shuffle(shard_files)

    n_train = int(TRAIN_FRAC * n_shards)
    n_val = int(VAL_FRAC * n_shards)
    n_test = n_shards - n_train - n_val

    train_files = shard_files[:n_train]
    val_files   = shard_files[n_train : n_train + n_val]
    test_files  = shard_files[n_train + n_val :]

    print("[shards] Split into:")
    print(f"  train shards: {len(train_files)}")
    print(f"  val shards:   {len(val_files)}")
    print(f"  test shards:  {len(test_files)}")

    # 2. Load splits
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val   = load_split_from_shards(val_files,   MAX_VAL_ROWS,   "val")
    df_test  = load_split_from_shards(test_files,  MAX_TEST_ROWS,  "test")

    # Basic sanity checks
    for name, df_part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in ["user_id", "target_category_idx", "target_category"]:
            if col not in df_part.columns:
                raise ValueError(f"[{name}] Missing required column '{col}'")

    # 3. Baselines on VAL set
    print("\n[baseline] Computing baselines on VAL split ...")
    y_val_true = df_val["target_category_idx"].astype(int).to_numpy()

    # Global majority (from TRAIN)
    majority_class = df_train["target_category_idx"].value_counts().idxmax()
    y_val_majority = np.full_like(y_val_true, fill_value=majority_class)
    acc_majority = accuracy_score(y_val_true, y_val_majority)
    print(f"[baseline] Global majority class idx = {majority_class}")
    print(f"[baseline] Val accuracy (global majority) = {acc_majority:.4f}")

    # Last-category baseline
    if "last_category_idx" in df_val.columns:
        y_val_last = df_val["last_category_idx"].astype(int).to_numpy()
        acc_last = accuracy_score(y_val_true, y_val_last)
        print(f"[baseline] Val accuracy (last_category_idx) = {acc_last:.4f}")
    else:
        print("[baseline] last_category_idx not found; skipping last-category baseline.")

    # Prefix-most-frequent-category baseline
    if "prefix_most_freq_category_idx" in df_val.columns:
        y_val_most = df_val["prefix_most_freq_category_idx"].astype(int).to_numpy()
        acc_most = accuracy_score(y_val_true, y_val_most)
        print(f"[baseline] Val accuracy (prefix_most_freq_category_idx) = {acc_most:.4f}")
    else:
        print("[baseline] prefix_most_freq_category_idx not found; skipping that baseline.")

    # 4. Feature prep for HistGBM
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
        prepare_features_for_histgbm(df_train, df_val, df_test)

    # 5. Define HistGBM model
    # Note: early_stopping=True will use an internal validation split from X_train.
    # We still keep an external val/test for honest evaluation.
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=300,          # increase if underfitting; verbose shows progress
        max_leaf_nodes=63,
        max_depth=None,        # None => no explicit depth limit; max_leaf_nodes controls complexity
        min_samples_leaf=50,   # regularization
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=1,             # <-- this gives you per-iteration training progress
    )

    # 6. Train + evaluate
    metrics = evaluate_histgbm(
        clf,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
