#!/usr/bin/env python
"""HistGradientBoosting pipeline for next-category prediction.

Design:
- Shard-level 80/10/10 train/val/test split (user-level safe).
- Cap rows per split for memory efficiency.
- Derived features added before model training.
- Random-search hyperparameter tuning on a train subset.
- Final ensemble of N_ENSEMBLE models with best hyperparams.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

from amazon_next_category.utils.config import RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from amazon_next_category.utils.model_io import (
    list_shard_files,
    load_split_from_shards,
    log_baselines,
    split_shards,
    validate_split_columns,
)

logger = logging.getLogger(__name__)

SHARD_DIR = "data/global/sequence_samples_by_shard"

MAX_TRAIN_ROWS = 600_000
MAX_VAL_ROWS = 100_000
MAX_TEST_ROWS = 100_000

TUNING_N_TRIALS = 10
TUNING_TRAIN_SUBSET = 200_000
N_ENSEMBLE = 2
USE_CLASS_WEIGHTS = False


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> None:
    """Add simple derived features in-place."""
    if "total_purchases" in df.columns:
        df["log_total_purchases"] = np.log1p(df["total_purchases"].clip(lower=0))

    if "prefix_length" in df.columns:
        df["log_prefix_length"] = np.log1p(df["prefix_length"].clip(lower=0))

    if "prefix_length" in df.columns and "total_purchases" in df.columns:
        denom = df["total_purchases"].replace(0, np.nan)
        df["prefix_length_frac_total"] = (df["prefix_length"] / denom).fillna(0.0)

    if "prefix_timespan" in df.columns:
        df["prefix_timespan_days"] = df["prefix_timespan"] / 86400.0
        span_days = df["prefix_timespan_days"].replace(0, np.nan)
        if "prefix_length" in df.columns:
            rate = df["prefix_length"] / span_days
            df["prefix_purchases_per_day"] = rate.fillna(df["prefix_length"])


def prepare_features_for_histgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:
    """Add derived features, build arrays, and return categorical feature indices."""
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        logger.info("Adding derived features to %s split...", name)
        add_derived_features(df)

    for df_part in [df_train, df_val, df_test]:
        df_part["target_category_idx"] = df_part["target_category_idx"].astype(int)

    label_col = "target_category_idx"
    drop_cols = ["user_id", "target_category"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols + [label_col]]

    cat_feature_cols = [c for c in feature_cols if "category_idx" in c]
    feature_col_positions = {c: i for i, c in enumerate(feature_cols)}
    categorical_feature_indices = [feature_col_positions[c] for c in cat_feature_cols]

    def df_to_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy(dtype=np.int64)
        return X, y

    X_train, y_train = df_to_Xy(df_train)
    X_val, y_val = df_to_Xy(df_val)
    X_test, y_test = df_to_Xy(df_test)

    logger.info(
        "Feature prep — %d features (%d categorical), X_train=%s",
        len(feature_cols),
        len(cat_feature_cols),
        X_train.shape,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, categorical_feature_indices


def compute_class_sample_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Return per-sample weights ~ 1 / freq_c^power, normalised to mean=1."""
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    weights_per_class = {cls: 1.0 / (cnt**power) for cls, cnt in freq.items()}
    w = np.array([weights_per_class[cls] for cls in y], dtype=np.float32)
    w *= float(len(w)) / w.sum()
    return w


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------


def predict_ensemble_proba(
    models: list[HistGradientBoostingClassifier], X: np.ndarray
) -> np.ndarray:
    probs = [m.predict_proba(X) for m in models]
    return np.mean(probs, axis=0)


def evaluate_split(
    name: str,
    models: list[HistGradientBoostingClassifier],
    X: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, float]:
    logger.info("Evaluating on %s (n=%d)...", name, len(y_true))
    proba = predict_ensemble_proba(models, X)
    y_pred = proba.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    logger.info("Accuracy (%s): %.4f", name, acc)
    try:
        top3 = top_k_accuracy_score(y_true, proba, k=3)
        logger.info("Top-3 accuracy (%s): %.4f", name, top3)
    except Exception as e:
        logger.warning("Could not compute top-3 for %s: %s", name, e)
    return y_pred, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    np.random.seed(RANDOM_SEED)

    shard_files = list_shard_files(SHARD_DIR)
    logger.info("Found %d shard files.", len(shard_files))
    train_files, val_files, test_files = split_shards(
        shard_files, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
    )
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")
    validate_split_columns(
        [("train", df_train), ("val", df_val), ("test", df_test)], cast_target=False
    )
    log_baselines(df_train, df_val)

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

    sample_weight_train = None
    if USE_CLASS_WEIGHTS:
        logger.info("Computing class-based sample weights...")
        sample_weight_train = compute_class_sample_weights(y_train)

    # Hyperparameter tuning
    rng2 = np.random.RandomState(RANDOM_SEED + 123)
    search_space = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_leaf_nodes": [31, 63, 127],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }

    if TUNING_N_TRIALS > 0:
        n_sub = min(TUNING_TRAIN_SUBSET, len(y_train))
        idx_sub = rng2.choice(len(y_train), size=n_sub, replace=False)
        X_sub, y_sub = X_train[idx_sub], y_train[idx_sub]
        sw_sub = sample_weight_train[idx_sub] if sample_weight_train is not None else None

        logger.info("Starting random search: %d trials on %d rows...", TUNING_N_TRIALS, n_sub)
        best_val_acc = -np.inf
        best_params = None
        trial_results = []

        for trial in range(TUNING_N_TRIALS):
            params = {
                "learning_rate": float(rng2.choice(search_space["learning_rate"])),
                "max_leaf_nodes": int(rng2.choice(search_space["max_leaf_nodes"])),
                "min_samples_leaf": int(rng2.choice(search_space["min_samples_leaf"])),
                "l2_regularization": float(rng2.choice(search_space["l2_regularization"])),
            }
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
                categorical_features=categorical_feature_indices or None,
                verbose=1,
            )
            if sw_sub is not None:
                clf.fit(X_sub, y_sub, sample_weight=sw_sub)
            else:
                clf.fit(X_sub, y_sub)

            val_acc = accuracy_score(y_val, clf.predict(X_val))
            logger.info(
                "Trial %d/%d: val_acc=%.4f, params=%s", trial + 1, TUNING_N_TRIALS, val_acc, params
            )
            trial_results.append((val_acc, params))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params

        for rank, (acc, p) in enumerate(sorted(trial_results, key=lambda x: x[0], reverse=True), 1):
            logger.info("  #%d: val_acc=%.4f, params=%s", rank, acc, p)
        logger.info("Best val_acc=%.4f with params=%s", best_val_acc, best_params)
    else:
        best_params = {
            "learning_rate": 0.1,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        }
        logger.info("Skipping tuning; using default params: %s", best_params)

    assert best_params is not None

    # Final ensemble
    logger.info("Training final ensemble of %d model(s)...", N_ENSEMBLE)
    ensemble_models = []
    for i in range(N_ENSEMBLE):
        seed = RANDOM_SEED + 1000 + i
        logger.info("Training ensemble member %d/%d (seed=%d)...", i + 1, N_ENSEMBLE, seed)
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
            categorical_features=categorical_feature_indices or None,
            verbose=1,
        )
        if sample_weight_train is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weight_train)
        else:
            clf.fit(X_train, y_train)
        ensemble_models.append(clf)

    y_train_pred, train_acc = evaluate_split("train", ensemble_models, X_train, y_train)
    y_val_pred, val_acc = evaluate_split("val", ensemble_models, X_val, y_val)
    y_test_pred, test_acc = evaluate_split("test", ensemble_models, X_test, y_test)

    logger.info("Summary — train: %.4f, val: %.4f, test: %.4f", train_acc, val_acc, test_acc)
    logger.info(
        "Classification report (test):\n%s",
        classification_report(y_test, y_test_pred, digits=4),
    )


if __name__ == "__main__":
    main()
