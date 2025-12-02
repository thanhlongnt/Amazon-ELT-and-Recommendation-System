#!/usr/bin/env python
"""
train_other_models_next_category.py

Try multiple models (SVMs, trees, forests) for predicting
target_category_idx from the sharded sequence data produced
by 04_create_train_data.py.

Assumptions:
- Per-shard sequence files exist at:
      data/global/sequence_samples_by_shard/sequence_user_shard=*.parquet
  (i.e., you've already run 04_create_train_data.py once).
- Each shard file has the same schema as used for logistic regression:
    * Columns like:
        user_id, target_category, target_category_idx,
        last_category_idx, last_k_category_idx, prefix_most_freq_category_idx,
        prefix_cat_count_*, and other numeric user/prefix features.

Design:
- Shard-level split (user-level consistent because shard = hash(user_id)):
    * 80% of shards -> train
    * 10% of shards -> val
    * 10% of shards -> test
- To avoid memory issues, we cap rows per split.
- Feature pipelining:
    * Numeric features:
        - SimpleImputer(strategy="median")
        - StandardScaler (for linear models)
        - SimpleImputer only (for tree-based models)
    * *_category_idx features:
        - Imputer + OneHotEncoder (for linear SVM)
        - Imputer only (integers used directly for tree-based models)
- Models:
    * linear_svm: LinearSVC (OvR on top of OHE’d categorical indices)
    * decision_tree: DecisionTreeClassifier (no OHE)
    * random_forest: RandomForestClassifier (no OHE)

Metrics:
- For each model:
    * Accuracy on train/val/test
    * Top-3 accuracy (where decision_function / predict_proba is available)
    * (Optional) classification report on test for the best model.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Target total rows per split.
# You can bump these if you have more RAM / patience.
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
# Evaluation helper
# ---------------------------------------------------------------------


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit and evaluate a model on train/val/test.
    Returns a dict of metrics for convenience.
    """
    print(f"\n\n================ MODEL: {name} ================")

    # Fit
    print(f"[{name}] Training...")
    model.fit(X_train, y_train)
    print(f"[{name}] Training done.")

    metrics = {}

    def _eval_split(split_name, X, y_true):
        print(f"[{name}:{split_name}] Evaluating on {split_name} (n={len(y_true):,}) ...")
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"[{name}:{split_name}] Accuracy = {acc:.4f}")

        # Top-3 accuracy if possible
        top3 = None
        scores = None
        if hasattr(model, "predict_proba"):
            try:
                scores = model.predict_proba(X)
            except Exception as e:
                print(f"[{name}:{split_name}] predict_proba failed for top-k: {e}")
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
            except Exception as e:
                print(f"[{name}:{split_name}] decision_function failed for top-k: {e}")

        if scores is not None:
            try:
                top3 = top_k_accuracy_score(y_true, scores, k=3)
                print(f"[{name}:{split_name}] Top-3 accuracy = {top3:.4f}")
            except Exception as e:
                print(f"[{name}:{split_name}] Could not compute top-3 accuracy: {e}")

        return acc, top3, y_pred

    train_acc, train_top3, _ = _eval_split("train", X_train, y_train)
    val_acc,   val_top3,   _ = _eval_split("val",   X_val,   y_val)
    test_acc,  test_top3,  y_test_pred = _eval_split("test",  X_test,  y_test)

    print(f"\n[{name}] Summary:")
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

    # Shuffle shards for randomness (same seed => deterministic split)
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

    # 2. Load the splits
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val   = load_split_from_shards(val_files,   MAX_VAL_ROWS,   "val")
    df_test  = load_split_from_shards(test_files,  MAX_TEST_ROWS,  "test")

    # Basic sanity checks & label type
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

    # Identify categorical index-like features vs numeric features
    cat_feature_cols = [c for c in feature_cols if "category_idx" in c]
    numeric_feature_cols = [c for c in feature_cols if c not in cat_feature_cols]

    print(f"[fe] Total feature columns: {len(feature_cols)}")
    print(f"[fe] Numeric features ({len(numeric_feature_cols)}):")
    print(f"     {numeric_feature_cols}")
    print(f"[fe] Categorical idx features ({len(cat_feature_cols)}):")
    print(f"     {cat_feature_cols}")

    # 5. Preprocessors

    # For linear models (SVM with OHE)
    linear_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    linear_categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", linear_numeric, numeric_feature_cols),
            ("cat", linear_categorical, cat_feature_cols),
        ]
    )

    # For tree-based models (no OHE, just impute)
    tree_numeric = SimpleImputer(strategy="median")
    tree_categorical = SimpleImputer(strategy="most_frequent")

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", tree_numeric, numeric_feature_cols),
            ("cat", tree_categorical, cat_feature_cols),
        ]
    )

    # 6. Define models
    models = {}

    # 6.1 Linear SVM (on OHE + scaled numeric)
    # models["linear_svm"] = Pipeline(
    #     steps=[
    #         ("preprocessor", linear_preprocessor),
    #         (
    #             "clf",
    #             LinearSVC(
    #                 C=1.0,
    #                 class_weight="balanced",
    #                 max_iter=5000,
    #                 verbose=1,
    #             ),
    #         ),
    #     ]
    # )

    # 6.2 Decision tree (raw numeric + integer categories)
    models["decision_tree"] = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=20,
                    min_samples_leaf=50,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    # 6.3 Random forest (also on imputed numeric + ints)
    models["random_forest"] = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=25,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    # 7. Train & evaluate each model
    all_metrics = {}
    for name, model in models.items():
        metrics = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        all_metrics[name] = metrics

    # 8. Pick best by validation accuracy and print a detailed test report
    best_name = max(all_metrics.keys(), key=lambda n: all_metrics[n]["val_acc"])
    best_metrics = all_metrics[best_name]
    print("\n\n================ BEST MODEL (by val acc) ================")
    print(f"Best model: {best_name}")
    print(f"  Val acc   = {best_metrics['val_acc']:.4f}")
    print(f"  Test acc  = {best_metrics['test_acc']:.4f}")
    if best_metrics["val_top3"] is not None:
        print(f"  Val top-3 = {best_metrics['val_top3']:.4f}")
    if best_metrics["test_top3"] is not None:
        print(f"  Test top-3 = {best_metrics['test_top3']:.4f}")

    print("\n[report:test] Classification report for BEST model on TEST split:")
    print(classification_report(y_test, best_metrics["y_test_pred"], digits=4))

    print("\nAll done.")


if __name__ == "__main__":
    main()