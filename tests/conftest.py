"""Shared pytest fixtures for amazon-next-category tests."""

from __future__ import annotations

import pytest
import pandas as pd


@pytest.fixture()
def user_counts_df() -> pd.DataFrame:
    """Small mock DataFrame matching the ``user_counts_*.parquet`` schema."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u1"],
            "num_purchases": [3, 5, 2, 4],
            "category": ["Books", "Movies_and_TV", "Books", "Movies_and_TV"],
        }
    )


@pytest.fixture()
def top_users_df() -> pd.DataFrame:
    """Small mock DataFrame matching the ``top_users.parquet`` schema."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "total_purchases": [7, 5],
            "distinct_categories": [3, 4],
            "entropy": [1.2, 1.5],
            "norm_entropy": [0.8, 0.9],
            "importance": [12.6, 12.5],
        }
    )


@pytest.fixture()
def mock_registry(tmp_path) -> dict:
    """A minimal registry dict with one 'raw' entry pointing to a temp file."""
    data_file = tmp_path / "data" / "raw" / "all_categories.txt"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("Books\nMovies_and_TV\n")

    return {
        "drive_root_folder_id": "fake_root_id",
        "raw": {
            "all_categories.txt": {
                "local_path": "data/raw/all_categories.txt",
            }
        },
        "processed": {},
        "locks": {},
    }
