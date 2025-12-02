#!/usr/bin/env python
"""
train_histgbm_next_category.py

"Best shot" HistGradientBoosting pipeline to predict target_category_idx
from the sharded sequence data produced by 04_create_train_data.py.

Assumptions:
- Per-shard files exist under:
    data/global/sequence_samples_by_shard/sequence_user_shard=*.parquet
- These include:
    * static user features (importance, entropy, total_purchases, ...)
    * prefix_* features
    * last_* features
    * last_k_category_idx
    * prefix_cat_count_* (from updated script 4)
    * target_category_idx, target_category

Design:
- Shard-level split (user-level, since shards are hash(user_id)):
    * 80% of shards -> train
    * 10% of shards -> val
    * 10% of shards -> test
- Use caps on rows per split for memory:
    * MAX_TRAIN_ROWS, MAX_VAL_ROWS, MAX_TEST_ROWS
- Features:
    * Use all columns except {user_id, target_category, target_category_idx}
    * Add derived features inside this script (no re-run of script 4)
    * Treat *_category_idx columns as categorical for HistGBM
- Model:
    * HistGradientBoostingClassifier (multi-class log_loss)
    * Optional sample weighting for class imbalance
    * Random-search hyperparameter tuning on a train subset
    * Final ensemble of N_ENSEMBLE models with best hyperparams
"""

import os
from pathlib import Path
import sys
import math
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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHARD_DIR = "data/global/sequence_samples_by_shard"

RANDOM_SEED = 42

# Caps on rows per split (to avoid OOM)
MAX_TRAIN_ROWS = 600_000    # bump this up/down if needed
MAX_VAL_ROWS   = 100_000
MAX_TEST_ROWS  = 100_000

# Fractions of shards for each split
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1  # rest -> test

# Hyperparameter search
TUNING_N_TRIALS = 10        # number of random configs to try
TUNING_TRAIN_SUBSET = 200_000  # rows from train used for tuning (if available)

# Ensemble
N_ENSEMBLE = 2  # number of final models to train with best hyperparams

# Class imbalance handling
USE_CLASS_WEIGHTS = False   # if True, uses sample weights ~ 1/sqrt(freq_c)


# ---------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------


def list_shard_files(shard_dir: str):
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found. "
            f"Run 04_create_train_data.py with per-shard outputs enabled."
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
# Feature engineering helpers
# ---------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> None:
    """
    Add simple derived features in-place to a sequence dataframe.

    These are all computed from existing columns, so script 4 does NOT need to be rerun.
    """
    # Log total purchases
    if "total_purchases" in df.columns:
        df["log_total_purchases"] = np.log1p(
            df["total_purchases"].clip(lower=0)
        )

    # Log prefix length
    if "prefix_length" in df.columns:
        df["log_prefix_length"] = np.log1p(
            df["prefix_length"].clip(lower=0)
        )

    # Prefix length as fraction of total_purchases
    if "prefix_length" in df.columns and "total_purchases" in df.columns:
        denom = df["total_purchases"].replace(0, np.nan)
        frac = df["prefix_length"] / denom
        df["prefix_length_frac_total"] = frac.fillna(0.0)

    # Prefix timespan (seconds -> days)
    if "prefix_timespan" in df.columns:
        df["prefix_timespan_days"] = df["prefix_timespan"] / 86400.0

        # Purchases per day in the prefix (rough activity rate)
        span_days = df["prefix_timespan_days"].replace(0, np.nan)
        if "prefix_length" in df.columns:
            rate = df["prefix_length"] / span_days
            df["prefix_purchases_per_day"] = rate.fillna(
                df["prefix_length"]
            )  # if span_days==0, approximate rate by count


def prepare_features_for_histgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
):
    """
    Add derived features, build feature matrices and label vectors,
    and identify categorical feature indices for HistGBM.
    """
    # Add derived features in-place
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"[fe] Adding derived features to {name} split ...")
        add_derived_features(df)

    # Ensure target is int
    for df_part in [df_train, df_val, df_test]:
        if "target_category_idx" not in df_part.columns:
            raise ValueError("Missing 'target_category_idx' in dataframe.")
        df_part["target_category_idx"] = df_part["target_category_idx"].astype(
            int
        )

    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]

    all_cols = df_train.columns.tolist()
    feature_cols = [c for c in all_cols if c not in drop_cols + [label_col]]

    # Treat *_category_idx features as categorical
    cat_feature_cols = [c for c in feature_cols if "category_idx" in c]
    categorical_feature_indices = [
        feature_cols.index(c) for c in cat_feature_cols
    ]

    def df_to_Xy(df: pd.DataFrame):
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy(dtype=np.int64)
        return X, y

    X_train, y_train = df_to_Xy(df_train)
    X_val, y_val = df_to_Xy(df_val)
    X_test, y_test = df_to_Xy(df_test)

    print("\n[fe] Feature prep summary for HistGBM:")
    print(f"  #feature columns: {len(feature_cols)}")
    print(f"  categorical idx features ({len(cat_feature_cols)}): {cat_feature_cols}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        categorical_feature_indices,
    )


def compute_class_sample_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    Compute per-sample weights based on class frequency:

        weight_c = 1 / (freq_c ** power)

    Normalized to mean ~ 1.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))

    weights_per_class = {
        cls: 1.0 / (cnt ** power) for cls, cnt in freq.items()
    }

    w = np.array([weights_per_class[cls] for cls in y], dtype=np.float32)
    w *= float(len(w)) / w.sum()  # normalize
    return w


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------


def predict_ensemble_proba(models, X: np.ndarray) -> np.ndarray:
    """
    Average predict_proba across an ensemble of models.
    """
    probs = [m.predict_proba(X) for m in models]
    return np.mean(probs, axis=0)


def evaluate_split(name: str, models, X: np.ndarray, y_true: np.ndarray):
    print(f"\n[histgbm:{name}] Evaluating on {name} (n={len(y_true):,}) ...")
    proba = predict_ensemble_proba(models, X)
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"[histgbm:{name}] Accuracy = {acc:.4f}")

    try:
        top3 = top_k_accuracy_score(y_true, proba, k=3)
        print(f"[histgbm:{name}] Top-3 accuracy = {top3:.4f}")
    except Exception as e:
        print(f"[histgbm:{name}] Could not compute top-3 accuracy: {e}")

    return y_pred, acc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    np.random.seed(RANDOM_SEED)

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
    val_files = shard_files[n_train : n_train + n_val]
    test_files = shard_files[n_train + n_val :]

    print("[shards] Split into:")
    print(f"  train shards: {len(train_files)}")
    print(f"  val shards:   {len(val_files)}")
    print(f"  test shards:  {len(test_files)}")

    # 2. Load splits (respecting max rows)
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")

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
        print(
            f"[baseline] Val accuracy (prefix_most_freq_category_idx) = {acc_most:.4f}"
        )
    else:
        print(
            "[baseline] prefix_most_freq_category_idx not found; skipping that baseline."
        )

    # 4. Prepare features for HistGBM
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        categorical_feature_indices,
    ) = prepare_features_for_histgbm(df_train, df_val, df_test)

    # Optional class/sample weights
    sample_weight_train = None
    if USE_CLASS_WEIGHTS:
        print("\n[weights] Computing class-based sample weights for training ...")
        sample_weight_train = compute_class_sample_weights(y_train)
        print(
            f"[weights] Example weights: min={sample_weight_train.min():.4f}, "
            f"max={sample_weight_train.max():.4f}, mean={sample_weight_train.mean():.4f}"
        )

    # 5. Hyperparameter tuning (small random search)
    rng = np.random.RandomState(RANDOM_SEED + 123)
    search_space = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_leaf_nodes": [31, 63, 127],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }

    # Subset for tuning
    if TUNING_N_TRIALS > 0:
        n_train = len(y_train)
        n_sub = min(TUNING_TRAIN_SUBSET, n_train)
        idx_sub = rng.choice(n_train, size=n_sub, replace=False)

        X_train_sub = X_train[idx_sub]
        y_train_sub = y_train[idx_sub]
        if sample_weight_train is not None:
            sw_train_sub = sample_weight_train[idx_sub]
        else:
            sw_train_sub = None

        print(
            f"\n[tuning] Starting random search with {TUNING_N_TRIALS} trials "
            f"on a train subset of {n_sub:,} rows."
        )

        best_val_acc = -np.inf
        best_params = None
        trial_results = []

        for trial in range(TUNING_N_TRIALS):
            params = {
                "learning_rate": rng.choice(search_space["learning_rate"]),
                "max_leaf_nodes": int(rng.choice(search_space["max_leaf_nodes"])),
                "min_samples_leaf": int(rng.choice(search_space["min_samples_leaf"])),
                "l2_regularization": float(
                    rng.choice(search_space["l2_regularization"])
                ),
            }

            print(
                f"\n[tuning] Trial {trial + 1}/{TUNING_N_TRIALS} with params: {params}"
            )

            clf = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=params["learning_rate"],
                max_iter=300,
                max_leaf_nodes=params["max_leaf_nodes"],
                min_samples_leaf=params["min_samples_leaf"],
                l2_regularization=params["l2_regularization"],
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=RANDOM_SEED + trial,
                categorical_features=(
                    categorical_feature_indices
                    if len(categorical_feature_indices) > 0
                    else None
                ),
                verbose=1,
            )

            if sw_train_sub is not None:
                clf.fit(X_train_sub, y_train_sub, sample_weight=sw_train_sub)
            else:
                clf.fit(X_train_sub, y_train_sub)

            y_val_pred = clf.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            print(f"[tuning] Trial {trial + 1}: VAL accuracy = {val_acc:.4f}")

            trial_results.append((val_acc, params))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params

        print("\n[tuning] Completed random search. Trial results:")
        for i, (acc, params) in enumerate(
            sorted(trial_results, key=lambda x: x[0], reverse=True), start=1
        ):
            print(f"  #{i}: val_acc={acc:.4f}, params={params}")

        print(
            f"\n[tuning] BEST val_acc={best_val_acc:.4f} with params={best_params}"
        )
    else:
        # Fallback: use a reasonable default similar to your previous good settings
        best_params = {
            "learning_rate": 0.1,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        }
        print(
            "\n[tuning] Skipping tuning. Using default params: "
            f"{best_params}"
        )

    assert best_params is not None

    # 6. Final ensemble training on full train set
    print(
        f"\n[histgbm] Training final ensemble of {N_ENSEMBLE} model(s) "
        f"with best params: {best_params}"
    )

    ensemble_models = []
    for i in range(N_ENSEMBLE):
        seed = RANDOM_SEED + 1000 + i
        print(f"\n[histgbm] Training ensemble member {i + 1}/{N_ENSEMBLE} (seed={seed}) ...")

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=best_params["learning_rate"],
            max_iter=400,
            max_leaf_nodes=best_params["max_leaf_nodes"],
            min_samples_leaf=best_params["min_samples_leaf"],
            l2_regularization=best_params["l2_regularization"],
            max_bins=255,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed,
            categorical_features=(
                categorical_feature_indices
                if len(categorical_feature_indices) > 0
                else None
            ),
            verbose=1,
        )

        if sample_weight_train is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weight_train)
        else:
            clf.fit(X_train, y_train)

        ensemble_models.append(clf)

    print("[histgbm] Final ensemble training done.")

    # 7. Evaluate ensemble on splits
    y_train_pred, train_acc = evaluate_split("train", ensemble_models, X_train, y_train)
    y_val_pred, val_acc = evaluate_split("val", ensemble_models, X_val, y_val)
    y_test_pred, test_acc = evaluate_split("test", ensemble_models, X_test, y_test)

    print("\n[histgbm] Summary:")
    print(f"  Train acc: {train_acc:.4f}")
    print(f"  Val   acc: {val_acc:.4f}")
    print(f"  Test  acc: {test_acc:.4f}")

    # Detailed classification report on test
    print("\n[histgbm:report:test] Classification report on TEST split:")
    print(classification_report(y_test, y_test_pred, digits=4))

    print("\nAll done.")


if __name__ == "__main__":
    main()
