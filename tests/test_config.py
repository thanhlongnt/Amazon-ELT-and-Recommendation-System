"""Sanity checks for amazon_next_category.utils.config constants."""

from __future__ import annotations

from amazon_next_category.utils.config import (
    CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    IMPORTANCE_PERCENTILE,
    MAX_TRAIN_ROWS,
    META_URL_TEMPLATE,
    MIN_DISTINCT_CATEGORIES,
    MIN_TOTAL_PURCHASES,
    N_LATEST_CATEGORIES,
    N_SHARDS,
    RANDOM_SEED,
    REVIEW_URL_TEMPLATE,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def test_n_shards_is_positive_int() -> None:
    assert isinstance(N_SHARDS, int)
    assert N_SHARDS > 0


def test_importance_percentile_in_range() -> None:
    assert isinstance(IMPORTANCE_PERCENTILE, float)
    assert 0.0 < IMPORTANCE_PERCENTILE < 1.0


def test_min_thresholds_positive() -> None:
    assert MIN_TOTAL_PURCHASES > 0
    assert MIN_DISTINCT_CATEGORIES > 0


def test_n_latest_categories_positive() -> None:
    assert isinstance(N_LATEST_CATEGORIES, int)
    assert N_LATEST_CATEGORIES > 0


def test_max_train_rows_positive() -> None:
    assert isinstance(MAX_TRAIN_ROWS, int)
    assert MAX_TRAIN_ROWS > 0


def test_split_fractions_sum_at_most_one() -> None:
    assert 0.0 < TRAIN_SPLIT < 1.0
    assert 0.0 < VAL_SPLIT < 1.0
    assert TRAIN_SPLIT + VAL_SPLIT <= 1.0


def test_random_seed_is_int() -> None:
    assert isinstance(RANDOM_SEED, int)


def test_chunk_size_and_timeout_positive() -> None:
    assert CHUNK_SIZE > 0
    assert DOWNLOAD_TIMEOUT > 0


def test_url_templates_contain_placeholder() -> None:
    assert "{category}" in REVIEW_URL_TEMPLATE
    assert "{category}" in META_URL_TEMPLATE
