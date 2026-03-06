"""Central configuration constants for the amazon-next-category project.

Import from here rather than scattering magic numbers across pipeline and model files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

#: Number of hash-based user shards used when distributing reviews to disk.
N_SHARDS: int = 256

#: Minimum percentile threshold for a user to be considered "important".
IMPORTANCE_PERCENTILE: float = 0.95

#: Minimum number of total purchases for a user to qualify as important.
MIN_TOTAL_PURCHASES: int = 3

#: Minimum number of distinct categories for a user to qualify as important.
MIN_DISTINCT_CATEGORIES: int = 3

#: Number of most-recent purchase categories to encode as features.
N_LATEST_CATEGORIES: int = 5

#: Minimum prefix length required before a sequence sample can be created.
MIN_PREFIX_LENGTH: int = 3

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

#: Maximum rows to load for the training split.
MAX_TRAIN_ROWS: int = 1_000_000

#: Fraction of shards assigned to the training split.
TRAIN_SPLIT: float = 0.8

#: Fraction of shards assigned to the validation split.
VAL_SPLIT: float = 0.1

#: Global random seed for reproducibility.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

#: Byte chunk size used when streaming HTTP downloads.
CHUNK_SIZE: int = 8192

#: HTTP timeout (seconds) for raw-data downloads.
DOWNLOAD_TIMEOUT: int = 60

# ---------------------------------------------------------------------------
# Data URLs
# ---------------------------------------------------------------------------

#: URL template for per-category review JSONL.GZ files.
REVIEW_URL_TEMPLATE: str = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/{category}.jsonl.gz"
)

#: URL template for per-category meta JSONL.GZ files.
META_URL_TEMPLATE: str = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{category}.jsonl.gz"
)
