"""Shared MLflow setup utilities."""
from __future__ import annotations

import os

import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_PREFIX = "amazon-next-category"


def setup_experiment(model_name: str) -> None:
    """Set the MLflow tracking URI and active experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{model_name}")
