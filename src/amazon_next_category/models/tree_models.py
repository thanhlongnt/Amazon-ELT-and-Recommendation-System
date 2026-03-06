#!/usr/bin/env python
"""Decision tree, random forest, and linear SVM for next-category prediction.

Reads from the sharded sequence dataset produced by
:mod:`amazon_next_category.pipeline.create_sequences`.

Evaluates linear SVM, decision tree, and random forest; reports the best
model by validation accuracy.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from amazon_next_category.utils.config import RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT

logger = logging.getLogger(__name__)

SHARD_DIR = "data/global/sequence_samples_by_shard"

MAX_TRAIN_ROWS = 300_000
MAX_VAL_ROWS = 100_000
MAX_TEST_ROWS = 100_000


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------


def list_shard_files(shard_dir: str) -> list[str]:
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found. Run create_sequences.py first."
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


def load_split_from_shards(files: list[str], max_rows: int, name: str) -> pd.DataFrame:
    logger.info(
        "Loading %s split from %d shards (max_rows=%s)...", name, len(files), max_rows
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
        if max_rows is not None and total >= max_rows:
            break
    if not dfs:
        raise RuntimeError(f"No rows loaded for split '{name}'.")
    df_out = pd.concat(dfs, ignore_index=True)
    logger.info("Final %s size: %d rows", name, len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    logger.info("=== MODEL: %s ===", name)
    logger.info("Training %s...", name)
    model.fit(X_train, y_train)

    metrics: dict = {}

    def _eval_split(split_name: str, X: pd.DataFrame, y_true: pd.Series) -> tuple:
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        logger.info("Accuracy (%s, %s): %.4f", name, split_name, acc)

        scores = None
        top3 = None
        if hasattr(model, "predict_proba"):
            try:
                scores = model.predict_proba(X)
            except Exception as e:
                logger.warning("predict_proba failed for %s: %s", name, e)
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
            except Exception as e:
                logger.warning("decision_function failed for %s: %s", name, e)

        if scores is not None:
            try:
                top3 = top_k_accuracy_score(y_true, scores, k=3)
                logger.info("Top-3 accuracy (%s, %s): %.4f", name, split_name, top3)
            except Exception as e:
                logger.warning("Could not compute top-3 for %s/%s: %s", name, split_name, e)

        return acc, top3, y_pred

    train_acc, _, _ = _eval_split("train", X_train, y_train)
    val_acc, val_top3, _ = _eval_split("val", X_val, y_val)
    test_acc, test_top3, y_test_pred = _eval_split("test", X_test, y_test)

    logger.info(
        "%s summary — train: %.4f, val: %.4f, test: %.4f", name, train_acc, val_acc, test_acc
    )

    metrics["train_acc"] = train_acc
    metrics["val_acc"] = val_acc
    metrics["test_acc"] = test_acc
    metrics["val_top3"] = val_top3
    metrics["test_top3"] = test_top3
    metrics["y_test_pred"] = y_test_pred
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    np.random.seed(RANDOM_SEED)

    shard_files = list_shard_files(SHARD_DIR)
    n_shards = len(shard_files)
    logger.info("Found %d shard files.", n_shards)

    rng = np.random.RandomState(RANDOM_SEED)
    shard_files_arr = np.array(shard_files)
    rng.shuffle(shard_files_arr)

    n_train = int(TRAIN_SPLIT * n_shards)
    n_val = int(VAL_SPLIT * n_shards)

    train_files = list(shard_files_arr[:n_train])
    val_files = list(shard_files_arr[n_train : n_train + n_val])
    test_files = list(shard_files_arr[n_train + n_val :])

    logger.info("Shard split — train: %d, val: %d, test: %d", len(train_files), len(val_files), len(test_files))

    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")

    for name, df_part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in ["user_id", "target_category_idx", "target_category"]:
            if col not in df_part.columns:
                raise ValueError(f"[{name}] Missing required column '{col}'")
        df_part["target_category_idx"] = df_part["target_category_idx"].astype(int)

    # Baselines
    logger.info("Computing baselines on val split...")
    y_val_true = df_val["target_category_idx"].to_numpy()
    majority_class = df_train["target_category_idx"].value_counts().idxmax()
    acc_majority = accuracy_score(y_val_true, np.full_like(y_val_true, majority_class))
    logger.info("Val accuracy (global majority, idx=%d): %.4f", majority_class, acc_majority)

    if "last_category_idx" in df_val.columns:
        acc_last = accuracy_score(y_val_true, df_val["last_category_idx"].astype(int).to_numpy())
        logger.info("Val accuracy (last_category_idx): %.4f", acc_last)

    if "prefix_most_freq_category_idx" in df_val.columns:
        acc_most = accuracy_score(
            y_val_true, df_val["prefix_most_freq_category_idx"].astype(int).to_numpy()
        )
        logger.info("Val accuracy (prefix_most_freq_category_idx): %.4f", acc_most)

    # Feature selection
    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols + [label_col]]

    cat_feature_cols = [c for c in feature_cols if "category_idx" in c]
    numeric_feature_cols = [c for c in feature_cols if c not in cat_feature_cols]

    logger.info(
        "Features — total: %d, numeric: %d, categorical: %d",
        len(feature_cols), len(numeric_feature_cols), len(cat_feature_cols),
    )

    X_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].copy()
    X_val = df_val[feature_cols].copy()
    y_val = df_val[label_col].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[label_col].copy()

    # Preprocessors
    linear_numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    linear_categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    linear_preprocessor = ColumnTransformer([
        ("num", linear_numeric, numeric_feature_cols),
        ("cat", linear_categorical, cat_feature_cols),
    ])

    tree_preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_feature_cols),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_feature_cols),
    ])

    models = {
        "linear_svm": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, verbose=1)),
        ]),
        "decision_tree": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("clf", DecisionTreeClassifier(
                max_depth=20, min_samples_leaf=50, random_state=RANDOM_SEED
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=25, min_samples_leaf=20,
                n_jobs=-1, random_state=RANDOM_SEED,
            )),
        ]),
    }

    all_metrics: dict = {}
    for name, model in models.items():
        metrics = evaluate_model(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        all_metrics[name] = metrics

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["val_acc"])
    best_metrics = all_metrics[best_name]
    logger.info("=== BEST MODEL (by val acc): %s ===", best_name)
    logger.info("  val_acc=%.4f, test_acc=%.4f", best_metrics["val_acc"], best_metrics["test_acc"])
    logger.info(
        "Classification report (test, best model):\n%s",
        classification_report(y_test, best_metrics["y_test_pred"], digits=4),
    )


if __name__ == "__main__":
    main()
