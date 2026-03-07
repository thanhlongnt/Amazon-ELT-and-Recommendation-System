![CI](https://github.com/thanhlongnt/cse158_asgn2/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Amazon Next-Category Prediction

Predict the **next Amazon purchase category** from a user's historical review sequence.
Built on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset (McAuley Lab, UCSD).

---

## Overview

Given a user's ordered purchase history up to time *t*, predict the category of their next purchase at *t+1*.

### Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `pipeline.build_user_counts` | Stream raw `.jsonl.gz` reviews → per-user purchase counts |
| 2 | `pipeline.filter_users` | Aggregate globally, score by importance, extract top users |
| 3 | `pipeline.extract_features` | Filter reviews to top users, produce per-review/user/item features |
| 4 | `pipeline.create_sequences` | Shard + build temporal sequence samples (prefix → target category), with bounded backpressure and JSON-checkpoint resumability |

### Models

| Module | Algorithm |
|--------|-----------|
| `models.logistic_regression` | Multinomial logistic regression (lbfgs) |
| `models.gradient_boosting` | HistGradientBoostingClassifier with random-search tuning |
| `models.tree_models` | Decision tree, random forest, linear SVM |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/thanhlongnt/cse158_asgn2.git
cd cse158_asgn2

# 2. Create a virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate

# 3. Install the package (editable mode)
pip install -e .

# 4. (Optional) install dev dependencies
pip install -r requirements-dev.txt
```

---

## Usage

### Run the full pipeline

```bash
# Run all stages end-to-end, then load the dataset
python scripts/run_pipeline.py --run-all

# Limit to a subset of categories
python scripts/run_pipeline.py --run-all --categories Electronics Toys_and_Games

# Just load the final dataset (skips pipeline stages)
python scripts/run_pipeline.py
```

Or run each stage individually:

```bash
# Step 1 – build per-category user counts
python -m amazon_next_category.pipeline.build_user_counts

# Step 2 – filter important users
python -m amazon_next_category.pipeline.filter_users

# Step 3 – extract top-user features
python -m amazon_next_category.pipeline.extract_features

# Step 4 – create sequence training samples
python -m amazon_next_category.pipeline.create_sequences

# Step 4 – resume after a crash (skips already-completed shards)
python -m amazon_next_category.pipeline.create_sequences \
    --skip-user-sharding --skip-review-sharding \
    --resume-phase2

# Step 4 – custom checkpoint file location
python -m amazon_next_category.pipeline.create_sequences \
    --skip-user-sharding --skip-review-sharding \
    --resume-phase2 --checkpoint-file /tmp/my_progress.json
```

#### Step 4 options reference

| Flag | Default | Description |
|------|---------|-------------|
| `--resume-phase2` | off | Skip shards already recorded in the checkpoint; safe to pass on first run |
| `--checkpoint-file PATH` | `<per-shard-output-dir>/progress.json` | Custom path for the JSON progress checkpoint |
| `--per-shard-output-dir PATH` | `data/global/sequence_samples_by_shard` | Directory for per-shard Parquet outputs and the default checkpoint |
| `--skip-user-sharding` | off | Reuse existing user-shard temp files (Phase 0) |
| `--skip-review-sharding` | off | Reuse existing review-shard temp files (Phase 1) |

**Crash-safe workflow:** if the process is killed mid-run, restart with `--skip-user-sharding --skip-review-sharding --resume-phase2`. The checkpoint (`progress.json`) records every completed shard; the restart will log `Resuming: N/256 shards already complete` and skip those shards, reprocessing only the remainder.

### Train a model

```bash
python -m amazon_next_category.models.gradient_boosting
python -m amazon_next_category.models.logistic_regression
python -m amazon_next_category.models.tree_models
```

### Explore the dataset

```bash
python scripts/explore_data.py --n 5
```

### Experiment tracking with MLflow

Every model training run is automatically logged to MLflow (params, metrics, and model artifacts).

**View past runs (CLI):**
```bash
mlflow runs list --experiment-name "amazon-next-category/logistic-regression"
mlflow runs list --experiment-name "amazon-next-category/gradient-boosting"
mlflow runs list --experiment-name "amazon-next-category/tree-models"
```

**Launch the interactive UI:**
```bash
mlflow ui        # opens http://localhost:5000
```

The UI lets you compare runs side-by-side, filter by metric, and download artifacts.

**Reload a saved model:**
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("mlruns/<experiment_id>/<run_id>/artifacts/model")
predictions = model.predict(X)
```
Run IDs are shown in `mlflow ui` or via `mlflow runs list`.

**Use a remote tracking server** (optional):
```bash
export MLFLOW_TRACKING_URI=http://my-mlflow-server:5000
python -m amazon_next_category.models.logistic_regression
```
Without this env var, runs are stored locally in `mlruns/`.

---

## Project Structure

```
cse158_asgn2/
├── src/
│   └── amazon_next_category/
│       ├── io/           # Data registry + Google Drive sync
│       ├── pipeline/     # Steps 1–4: raw → sequence dataset; pipeline_utils.py, run_pipeline.py
│       ├── models/       # Logistic regression, HistGBM, tree models
│       └── utils/        # config.py (constants), model_io.py (shard helpers), mlflow_utils.py
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── notebooks/            # analysis.ipynb
├── configs/              # data_registry.yaml, drive_config.yaml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Evaluation

Metrics reported for each model on the held-out test split:

- **Accuracy** — primary metric
- **Top-3 accuracy** — true category within top-3 predictions
- **Classification report** — per-class precision / recall / F1

Baselines computed on the validation split:

| Baseline | Description |
|----------|-------------|
| Global majority | Always predict the most-frequent training category |
| Last-category | Predict the same category as the user's last purchase |
| Prefix-most-frequent | Predict the user's most frequent category so far |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Format
black src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/ --max-line-length=100

# Type check
mypy src/

# Install pre-commit hooks
pre-commit install
```

---

## Predictive Task Details

- **Task** – Predict category at *t+1* given purchase prefix up to *t*.
- **Split** – User-level (hash-based shard split) to prevent leakage.
- **Features** – Static user importance/entropy, prefix length/timespan, last-N category indices, prefix category counts, last purchase rating/helpful/item-avg.