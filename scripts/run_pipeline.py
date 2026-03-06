#!/usr/bin/env python
"""CLI entry point: run the full pipeline end-to-end, or just load the dataset.

Pipeline stages (run in order):
1. ``build_user_counts``  — download + parse raw reviews, compute per-user counts
2. ``filter_users``       — aggregate globally, compute importance, extract top users
3. ``extract_features``   — filter to top users, produce per-review/user/item features
4. ``create_sequences``   — shard + build temporal sequence samples
5. load + inspect         — load sequence dataset; run a model from ``models/``

Usage::

    # Run all stages end-to-end
    python scripts/run_pipeline.py --run-all

    # Run all stages for a subset of categories
    python scripts/run_pipeline.py --run-all --categories Electronics Toys_and_Games

    # Just load the final dataset (default)
    python scripts/run_pipeline.py

"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

import amazon_next_category.io.data_io as data_io

logger = logging.getLogger(__name__)

PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_stage(module: str, extra_args: list[str]) -> None:
    cmd = [PYTHON, "-m", module] + extra_args
    logger.info("=== Running: %s ===", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Stage {module} failed with exit code {result.returncode}")


def run_all(categories: list[str] | None = None) -> None:
    cat_args = (["--categories"] + categories) if categories else []

    run_stage("amazon_next_category.pipeline.build_user_counts", cat_args)
    run_stage("amazon_next_category.pipeline.filter_users", [])
    run_stage("amazon_next_category.pipeline.extract_features", cat_args)
    run_stage("amazon_next_category.pipeline.create_sequences", cat_args)


def load_dataset() -> None:
    data_io.resync_registry()
    data_io.ensure_local_path("data/global/sequence_training_samples.parquet")

    data = pd.read_parquet("data/global/sequence_training_samples.parquet")
    logger.info("Sequence dataset loaded — shape: %s", data.shape)
    logger.info("Columns: %s", list(data.columns))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the next-category prediction pipeline.")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all pipeline stages (1-4) before loading the dataset.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help="Limit stages 1, 3, 4 to these categories.",
    )
    args = parser.parse_args()

    if args.run_all:
        run_all(categories=args.categories)

    load_dataset()


if __name__ == "__main__":
    main()
