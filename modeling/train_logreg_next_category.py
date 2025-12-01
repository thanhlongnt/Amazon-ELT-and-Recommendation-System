#!/usr/bin/env python
"""
train_logreg_next_category.py

Train a multinomial logistic regression model to predict
target_category_idx from the sharded sequence data produced
by 04_create_train_data.py.

Backwards-compatibility logic:

- If the big global Parquet
    data/global/sequence_training_samples.parquet
  is not local, we use common_scripts.data_io to fetch it from Drive.

- If we *just downloaded* that global file and local shards exist under
    data/global/sequence_samples_by_shard
  we delete those shards (they're likely out of sync) and rebuild
  shard files from the global parquet.

- If shard files don't exist at all but the global parquet is present,
  we also build shards from the global parquet.

We then train from the shards to avoid loading the entire global dataset
into RAM at once.

Modeling:

- Split by shards:
    80% train, 10% val, 10% test (user-level split, since shards are
    defined by hash(user_id))

- Cap rows:
    train: 1,000,000
    val:     200,000
    test:    200,000

- Features:
    * All numeric columns (including prefix_cat_count_*).
    * All *_category_idx columns as categorical with one-hot.
    * Additional derived features built in `add_derived_features`.

- Preprocessing:
    * Numeric: SimpleImputer(median) + StandardScaler(with_mean=False)
    * Categorical: SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown)

- Classifier:
    * LogisticRegression(multi_class="multinomial", lbfgs, n_jobs=-1, verbose=1)
"""

import os
import sys
import shutil
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
MAX_VAL_ROWS = 200_000
MAX_TEST_ROWS = 200_000

# Fractions of shards for each split
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # rest -> test

# Number of shards to create when sharding global file
N_SHARDS_FOR_GLOBAL = 256

# ---------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------


def list_shard_files(shard_dir: str):
    """List per-shard parquet files, e.g. sequence_user_shard=0.parquet."""
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(
            f"Shard directory '{shard_dir}' not found. "
            f"Make sure 04_create_train_data.py was run, or that we have "
            f"a global sequence file to shard from."
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


def shard_global_sequence_file(global_path: str, shard_dir: str, n_shards: int = 256):
    """
    Take the big global sequence_training_samples.parquet and shard it by user_id
    into per-shard files:

        data/global/sequence_samples_by_shard/sequence_user_shard=<k>.parquet

    This runs in a streaming fashion over Parquet row-groups to avoid loading
    the entire dataset into RAM.
    """
    print(
        f"[shard-global] Sharding global sequence file '{global_path}' "
        f"into {n_shards} shards under '{shard_dir}' ..."
    )

    from pandas.util import hash_pandas_object
    import pyarrow.parquet as pq
    import pyarrow as pa
    import gc

    if not os.path.exists(global_path):
        raise FileNotFoundError(
            f"[shard-global] Global sequence file not found at {global_path}"
        )

    os.makedirs(shard_dir, exist_ok=True)

    pf = pq.ParquetFile(global_path)
    num_row_groups = pf.num_row_groups
    print(f"[shard-global] Parquet file has {num_row_groups} row groups.")

    writers = {}
    total_rows = 0

    try:
        for rg_idx in range(num_row_groups):
            table = pf.read_row_group(rg_idx)
            df = table.to_pandas()

            if "user_id" not in df.columns:
                raise ValueError(
                    "[shard-global] Global sequence file is missing 'user_id' column."
                )

            df["user_id"] = df["user_id"].astype(str)

            # Compute shard id from user_id
            shard_ids = (
                hash_pandas_object(df["user_id"], index=False).values % n_shards
            )
            df["user_shard"] = shard_ids.astype("int32")

            print(
                f"[shard-global] Row group {rg_idx + 1}/{num_row_groups}: "
                f"{len(df):,} rows"
            )

            # Group by shard and append to its writer
            for shard_id, shard_df in df.groupby("user_shard"):
                shard_df = shard_df.drop(columns=["user_shard"])
                shard_table = pa.Table.from_pandas(shard_df)

                shard_path = os.path.join(
                    shard_dir, f"sequence_user_shard={int(shard_id)}.parquet"
                )

                if shard_id not in writers:
                    writers[shard_id] = pq.ParquetWriter(
                        shard_path, shard_table.schema
                    )

                writers[shard_id].write_table(shard_table)

            total_rows += len(df)
            del df, table
            gc.collect()

    finally:
        # Close all writers
        for shard_id, writer in writers.items():
            writer.close()
            print(f"[shard-global] Closed writer for shard {int(shard_id)}")

    print(
        f"[shard-global] Finished sharding global file. Total rows processed: {total_rows:,}"
    )


def ensure_shards_from_global_if_needed():
    """
    If shard files already exist under SHARD_DIR, do nothing.

    Otherwise, if the global parquet exists, shard it into SHARD_DIR.

    If neither shards nor global parquet are available, raise a helpful error.
    """
    existing_files = []
    if os.path.exists(SHARD_DIR):
        existing_files = [
            os.path.join(SHARD_DIR, f)
            for f in os.listdir(SHARD_DIR)
            if f.endswith(".parquet") and f.startswith("sequence_user_shard=")
        ]

    if existing_files:
        print(
            f"[shards] Found {len(existing_files)} existing shard files in {SHARD_DIR}; "
            f"reusing them."
        )
        return

    # No shard files; try to shard from global parquet
    if os.path.exists(GLOBAL_OUT_PATH):
        shard_global_sequence_file(
            GLOBAL_OUT_PATH, SHARD_DIR, n_shards=N_SHARDS_FOR_GLOBAL
        )
    else:
        raise FileNotFoundError(
            "[shards] No shard files found AND global sequence file "
            f"'{GLOBAL_OUT_PATH}' does not exist. "
            "Run 04_create_train_data.py first, or make sure the global "
            "sequence dataset is registered on Drive."
        )


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that can be computed from existing columns
    without re-running script 4.

    New features (if source columns are present):

        - prefix_reviews_per_day
        - last_minus_prefix_avg_rating
        - last_item_avg_minus_prefix_item_avg
        - prefix_cat_dom_ratio
        - prefix_cat_entropy_dynamic
        - prefix_cat_norm_entropy_dynamic
    """
    df = df.copy()

    # 1) Activity tempo: reviews per day in prefix
    if {"prefix_length", "prefix_timespan"}.issubset(df.columns):
        span_days = df["prefix_timespan"].astype(float) / (3600.0 * 24.0)
        # avoid division by zero; treat very short spans as 1 day
        span_days = span_days.replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = df["prefix_length"].astype(float) / span_days
        rate = rate.replace([np.inf, -np.inf], np.nan)
        df["prefix_reviews_per_day"] = rate

    # 2) Rating dynamics: last rating vs prefix mean
    if "last_rating" in df.columns and "prefix_avg_rating" in df.columns:
        df["last_minus_prefix_avg_rating"] = (
            df["last_rating"] - df["prefix_avg_rating"]
        )

    # 3) Item rating dynamics: last item's avg rating vs prefix item avg
    if (
        "last_item_avg_rating" in df.columns
        and "prefix_avg_item_avg_rating" in df.columns
    ):
        df["last_item_avg_minus_prefix_item_avg"] = (
            df["last_item_avg_rating"] - df["prefix_avg_item_avg_rating"]
        )

    # 4) Category concentration / entropy from prefix_cat_count_* columns
    cat_count_cols = [c for c in df.columns if c.startswith("prefix_cat_count_")]
    if cat_count_cols:
        counts = df[cat_count_cols].to_numpy(dtype=np.float32)
        total_counts = counts.sum(axis=1)
        max_counts = counts.max(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            dom_ratio = np.where(
                total_counts > 0, max_counts / total_counts, np.nan
            )
        df["prefix_cat_dom_ratio"] = dom_ratio

        # Dynamic entropy of prefix category distribution
        total_counts_safe = np.where(total_counts > 0, total_counts, 1.0)
        probs = counts / total_counts_safe[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(probs, where=(probs > 0))
        entropy = -np.nansum(probs * log_probs, axis=1)
        df["prefix_cat_entropy_dynamic"] = entropy

        num_cats = counts.shape[1]
        if num_cats > 1:
            df["prefix_cat_norm_entropy_dynamic"] = entropy / np.log(num_cats)
        else:
            df["prefix_cat_norm_entropy_dynamic"] = np.nan

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    np.random.seed(RANDOM_SEED)

    # 0. Resync data registry and ensure global sequence file is available (if registered)
    #    This makes it easy to run on a fresh machine.
    had_global_before = os.path.exists(GLOBAL_OUT_PATH)

    print("[data_io] Resyncing registry and trying to ensure global sequence dataset is local...")
    try:
        data_io.resync_registry()
        data_io.ensure_local_path(GLOBAL_OUT_PATH)
    except Exception as e:
        print(f"[data_io] WARNING: could not ensure {GLOBAL_OUT_PATH}: {e}")

    has_global_after = os.path.exists(GLOBAL_OUT_PATH)
    downloaded_global = has_global_after and not had_global_before
    if downloaded_global:
        print(
            f"[data_io] Detected freshly downloaded global sequence file at {GLOBAL_OUT_PATH}."
        )

    # If we just downloaded global from Drive and shards already exist, nuke shards so
    # we can rebuild them from the canonical global parquet.
    if downloaded_global and os.path.exists(SHARD_DIR):
        print(
            f"[shards] Existing shards under {SHARD_DIR} will be removed to "
            "rebuild from the freshly downloaded global sequence file."
        )
        shutil.rmtree(SHARD_DIR)

    # 1. Ensure shard files exist (possibly sharding from global parquet)
    ensure_shards_from_global_if_needed()

    # 2. Discover shard files
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

    # 3. Load splits (respecting max rows)
    df_train = load_split_from_shards(train_files, MAX_TRAIN_ROWS, "train")
    df_val = load_split_from_shards(val_files, MAX_VAL_ROWS, "val")
    df_test = load_split_from_shards(test_files, MAX_TEST_ROWS, "test")

    # 4. Sanity checks and derived features
    for name, df_part in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for col in ["user_id", "target_category_idx", "target_category"]:
            if col not in df_part.columns:
                raise ValueError(f"[{name}] Missing required column '{col}'")
        df_part["target_category_idx"] = df_part["target_category_idx"].astype(int)

    # Add derived features BEFORE we select feature columns
    df_train = add_derived_features(df_train)
    df_val = add_derived_features(df_val)
    df_test = add_derived_features(df_test)

    # 5. Baselines on VAL set
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
        print(
            f"[baseline] Val accuracy (prefix_most_freq_category_idx) = {acc_most:.4f}"
        )
    else:
        print(
            "[baseline] prefix_most_freq_category_idx not found; "
            "skipping that baseline."
        )

    # 6. Feature selection
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

    # 7. Preprocessing + model
    print("\n[model] Building preprocessing pipeline + multinomial logistic regression ...")

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
        multi_class="multinomial",  # will be default in future sklearn
        solver="lbfgs",
        max_iter=400,  # bump up from 200 to help convergence
        n_jobs=-1,
        verbose=1,  # gives you per-iteration progress, esp. during "Using backend LokyBackend..."
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    # 8. Train
    print("\n[train] Fitting model on TRAIN split ...")
    model.fit(X_train, y_train)
    print("[train] Done.")

    # 9. Evaluation helpers
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

    # 10. Evaluate on splits
    y_train_pred, train_acc = evaluate_split("train", X_train, y_train)
    y_val_pred, val_acc = evaluate_split("val", X_val, y_val)
    y_test_pred, test_acc = evaluate_split("test", X_test, y_test)

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
