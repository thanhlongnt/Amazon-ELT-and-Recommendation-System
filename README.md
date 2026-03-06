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
| 4 | `pipeline.create_sequences` | Shard + build temporal sequence samples (prefix → target category) |

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
```

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

---

## Project Structure

```
cse158_asgn2/
├── src/
│   └── amazon_next_category/
│       ├── io/           # Data registry + Google Drive sync
│       ├── pipeline/     # Steps 1–4: raw → sequence dataset
│       ├── models/       # Logistic regression, HistGBM, tree models
│       └── utils/        # config.py (constants)
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

See `notebooks/analysis.ipynb` for a full exploratory analysis and results.
