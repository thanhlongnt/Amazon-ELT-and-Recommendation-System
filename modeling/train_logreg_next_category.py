#!/usr/bin/env python
"""
train_logreg_next_category.py

Train a multinomial logistic regression model to predict
target_category_idx from the sharded sequence data produced
by 04_create_train_data.py.

Instead of loading the huge global file
  data/global/sequence_training_samples.parquet
(which can blow up RAM), we work from the per-shard files:

  data/global/sequence_samples_by_shard/sequence_user_shard=*.parquet

Design:

- Shard-level split (equivalent to user-level split, since shards are
  defined by hash(user_id)):
    * 80% of shards -> train
    * 10% of shards -> val
    * 10% of shards -> test
- We cap the *total rows* for each split to stay within memory.
- Modeling approach:
    * Numeric features: SimpleImputer + StandardScaler
    * *_category_idx features: SimpleImputer + OneHotEncoder
    * Classifier: multinomial LogisticRegression (lbfgs)
- Baselines on the validation set:
    * Global majority class
    * last_category_idx
    * prefix_most_freq_category_idx
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
)

# ---------------------------------------------------------------------
# Wire up common_scripts.data_io regardless of where we run from
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common_scripts import data_io  # type: ignore

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

SHARD_DIR = "data/global/sequence_samples_by_shard"
GLOBAL_OUT_PATH = "data/global/sequence_training_samples.parquet"  # uploaded by 04_create_train_data.py

RANDOM_SEED = 42

# Target total rows per split (approx upper bound)
MAX_TRAIN_ROWS = 1_000_000
MAX_VAL_ROWS   =   200_000
MAX_TEST_ROWS  =   200_000

# Fractions of shards for each split
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1  # rest -> test

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def list_shard_files(shard_dir: str):
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found. "
            f"Make sure 04_create_train_data.py was run with per-shard outputs enabled."
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
    If max_rows is None, load all rows.
    """
    print(f"\n[load:{name}] Loading split from {len(files)} shard files "
          f"(max_rows={max_rows if max_rows is not None else 'ALL'}) ...")

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
        print(f"[load:{name}]   +{len(df_shard):,} rows from {os.path.basename(fpath)} "
              f"(cumulative: {total:,})")

        if max_rows is not None and total >= max_rows:
            break

    if not dfs:
        raise RuntimeError(f"[load:{name}] No rows loaded for split '{name}'.")

    df_out = pd.concat(dfs, ignore_index=True)
    print(f"[load:{name}] Final {name} size: {len(df_out):,} rows")
    return df_out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    np.random.seed(RANDOM_SEED)

    # 0. Resync data registry and ensure global sequence file is available (if registered)
    #    This makes it easy to run on a fresh machine: the global parquet will be
    #    downloaded from Drive if the registry knows about it.
    print("[data_io] Resyncing registry and trying to ensure global sequence dataset is local...")
    try:
        data_io.resync_registry()
        data_io.ensure_local_path(GLOBAL_OUT_PATH)
        print(f"[data_io] ensure_local_path OK for {GLOBAL_OUT_PATH}")
    except Exception as e:
        print(f"[data_io] WARNING: could not ensure {GLOBAL_OUT_PATH}: {e}")

    # 1. Discover shard files
    print(f"[shards] Listing shard files under {SHARD_DIR} ...")
    shard_files = list_shard_files(SHARD_DIR)
    n_shards = len(shard_files)
    print(f"[shards] Found {n_shards} shard files.")

    # Shuffle shards for randomness
    rng = np.random.RandomState(RANDOM_SEED)
    shard_files = np.array(shard_files)
    rng.shuffle(shard_files)

    # Split shards into train/val/test
    n_train = int(TRAIN_FRAC * n_shards)
    n_val = int(VAL_FRAC * n_shards)
    n_test = n_shards - n_train - n_val

    train_files = shard_files[:n_train]
    val_files   = shard_files[n_train:n_train + n_val]
    test_files  = shard_files[n_train + n_val:]

    print(f"[shards] Split into:")
    print(f"  train shards: {len(train_files)}")
    print(f"  val shards:   {len(val_files)}")
    print(f"  test shards:  {len(test_files)}")

    # 2. Load splits (respecting max rows)
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val   = load_split_from_shards(val_files,   MAX_VAL_ROWS,   "val")
    df_test  = load_split_from_shards(test_files,  MAX_TEST_ROWS,  "test")

    # Basic sanity checks
    for name, df_part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in ["user_id", "target_category_idx", "target_category"]:
            if col not in df_part.columns:
                raise ValueError(f"[{name}] Missing required column '{col}'")
        df_part["target_category_idx"] = df_part["target_category_idx"].astype(int)

    # 3. Baselines on VAL set
    print("\n[baseline] Computing baselines on VAL split ...")

    y_val_true = df_val["target_category_idx"].to_numpy()

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

    # 4. Feature selection
    print("\n[fe] Preparing feature matrices ...")

    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]

    all_cols = df_train.columns.tolist()
    feature_cols = [c for c in all_cols if c not in drop_cols + [label_col]]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].copy()

    X_val = df_val[feature_cols].copy()
    y_val = df_val[label_col].copy()

    X_test = df_test[feature_cols].copy()
    y_test = df_test[label_col].copy()

    # Categorical index-like features
    cat_feature_cols = [c for c in feature_cols if "category_idx" in c]
    numeric_feature_cols = [c for c in feature_cols if c not in cat_feature_cols]

    print(f"[fe] Total feature columns: {len(feature_cols)}")
    print(f"[fe] Numeric features    ({len(numeric_feature_cols)}): {numeric_feature_cols}")
    print(f"[fe] Categorical idx features ({len(cat_feature_cols)}): {cat_feature_cols}")

    # 5. Preprocessing + model
    print("\n[model] Building preprocessing pipeline + multinomial logistic regression ...")

    # ---- NEW: Imputation to handle NaNs before scaler / one-hot ----
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_feature_cols),
            ("cat", categorical_transformer, cat_feature_cols),
        ]
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        n_jobs=-1,
        verbose=1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    # 6. Train
    print("\n[train] Fitting model on TRAIN split ...")
    model.fit(X_train, y_train)
    print("[train] Done.")

    # 7. Evaluation helpers
    def evaluate_split(name, X, y_true):
        print(f"\n[eval:{name}] Evaluating on {name} set (n={len(y_true):,}) ...")
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"[eval:{name}] Accuracy = {acc:.4f}")

        # Top-3 accuracy (if predict_proba is available)
        try:
            y_proba = model.predict_proba(X)
            top3 = top_k_accuracy_score(y_true, y_proba, k=3)
            print(f"[eval:{name}] Top-3 accuracy = {top3:.4f}")
        except Exception as e:
            print(f"[eval:{name}] Could not compute top-3 accuracy: {e}")

        return y_pred, acc

    # 8. Evaluate on splits
    y_train_pred, train_acc = evaluate_split("train", X_train, y_train)
    y_val_pred,   val_acc   = evaluate_split("val",   X_val,   y_val)
    y_test_pred,  test_acc  = evaluate_split("test",  X_test,  y_test)

    print("\n[summary] Accuracies:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val:   {val_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")

    # Detailed classification report on test
    print("\n[report:test] Classification report on TEST split:")
    print(classification_report(y_test, y_test_pred, digits=4))

    print("\nAll done.")


if __name__ == "__main__":
    main()

# python -m modeling.train_logreg_next_category
