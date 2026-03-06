#!/usr/bin/env python
"""Decision tree, random forest, and linear SVM for next-category prediction.

Reads from the sharded sequence dataset produced by
:mod:`amazon_next_category.pipeline.create_sequences`.

Evaluates linear SVM, decision tree, and random forest; reports the best
model by validation accuracy.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
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
from amazon_next_category.utils.mlflow_utils import setup_experiment
from amazon_next_category.utils.model_io import (
    list_shard_files,
    load_split_from_shards,
    log_baselines,
    select_feature_columns,
    split_shards,
    validate_split_columns,
)

logger = logging.getLogger(__name__)

SHARD_DIR = "data/global/sequence_samples_by_shard"

MAX_TRAIN_ROWS = 300_000
MAX_VAL_ROWS = 100_000
MAX_TEST_ROWS = 100_000


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
    setup_experiment("tree-models")

    shard_files = list_shard_files(SHARD_DIR)
    logger.info("Found %d shard files.", len(shard_files))
    train_files, val_files, test_files = split_shards(
        shard_files, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
    )
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")
    validate_split_columns([("train", df_train), ("val", df_val), ("test", df_test)])
    log_baselines(df_train, df_val)
    feature_cols, cat_feature_cols, numeric_feature_cols = select_feature_columns(df_train)

    label_col = "target_category_idx"
    X_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].copy()
    X_val = df_val[feature_cols].copy()
    y_val = df_val[label_col].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[label_col].copy()

    # Preprocessors
    linear_numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    linear_categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    linear_preprocessor = ColumnTransformer(
        [
            ("num", linear_numeric, numeric_feature_cols),
            ("cat", linear_categorical, cat_feature_cols),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_feature_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_feature_cols),
        ]
    )

    models = {
        "linear_svm": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, verbose=1)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=20, min_samples_leaf=50, random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
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
        ),
    }

    model_params = {
        "linear_svm": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
        "decision_tree": {"max_depth": 20, "min_samples_leaf": 50},
        "random_forest": {"n_estimators": 100, "max_depth": 25, "min_samples_leaf": 20},
    }

    all_metrics: dict = {}
    with mlflow.start_run(run_name="tree_models"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(model_params.get(name, {}))
                metrics = evaluate_model(
                    name, model, X_train, y_train, X_val, y_val, X_test, y_test
                )
                mlflow.log_metrics(
                    {
                        "train_acc": metrics["train_acc"],
                        "val_acc": metrics["val_acc"],
                        "test_acc": metrics["test_acc"],
                    }
                )
                if metrics["val_top3"] is not None:
                    mlflow.log_metric("val_top3", metrics["val_top3"])
                if metrics["test_top3"] is not None:
                    mlflow.log_metric("test_top3", metrics["test_top3"])
                mlflow.sklearn.log_model(model, artifact_path="model")
            all_metrics[name] = metrics

        best_name = max(all_metrics, key=lambda n: all_metrics[n]["val_acc"])
        best_metrics = all_metrics[best_name]
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_val_acc", best_metrics["val_acc"])
        mlflow.log_metric("best_test_acc", best_metrics["test_acc"])

    logger.info("=== BEST MODEL (by val acc): %s ===", best_name)
    logger.info("  val_acc=%.4f, test_acc=%.4f", best_metrics["val_acc"], best_metrics["test_acc"])
    logger.info(
        "Classification report (test, best model):\n%s",
        classification_report(y_test, best_metrics["y_test_pred"], digits=4),
    )


if __name__ == "__main__":
    main()
