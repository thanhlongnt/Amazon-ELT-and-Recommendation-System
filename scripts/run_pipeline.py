#!/usr/bin/env python
"""CLI entry point: load the sequence training dataset and show basic info.

Run the full pipeline by executing the pipeline stages in order:

1. ``build_user_counts``  — download + parse raw reviews, compute per-user counts
2. ``filter_users``       — aggregate globally, compute importance, extract top users
3. ``extract_features``   — filter to top users, produce per-review/user/item features
4. ``create_sequences``   — shard + build temporal sequence samples
5. (this script)          — load sequence dataset; run a model from ``models/``

Usage::

    python scripts/run_pipeline.py

"""

from __future__ import annotations

import logging

import pandas as pd

import amazon_next_category.io.data_io as data_io

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_io.resync_registry()
    data_io.ensure_local_path("data/global/sequence_training_samples.parquet")

    data = pd.read_parquet("data/global/sequence_training_samples.parquet")
    logger.info("Sequence dataset loaded — shape: %s", data.shape)
    logger.info("Columns: %s", list(data.columns))


if __name__ == "__main__":
    main()
