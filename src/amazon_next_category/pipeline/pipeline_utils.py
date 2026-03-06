"""Shared pipeline helper functions used by build_user_counts and extract_features."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import requests

from amazon_next_category.io.data_io import ensure_local_path
from amazon_next_category.utils.config import CHUNK_SIZE, DOWNLOAD_TIMEOUT

logger = logging.getLogger(__name__)


def download_if_needed(url: str, dest: Path, force: bool = False) -> None:
    """Stream-download *url* to *dest*, skipping if it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.debug("File already exists: %s", dest)
        return

    logger.info("Downloading%s: %s -> %s", " (force)" if force else "", url, dest)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    logger.info("Download complete: %s", dest)


def ensure_raw_gzip_or_download(
    path: Path, url: str, allow_download: bool, repo_root: Path
) -> None:
    """Ensure a raw gzip exists locally (Drive first, then UCSD fallback)."""
    if path.exists():
        return

    rel = str(path.relative_to(repo_root))
    try:
        logger.info("Trying Drive for %s", rel)
        ensure_local_path(rel)
        if path.exists():
            return
    except Exception:
        pass

    if allow_download:
        download_if_needed(url, path, force=False)
    else:
        raise FileNotFoundError(
            f"Missing {path} and --no-download is set; "
            "no Drive entry or HTTP download attempted."
        )


def ensure_outputs_from_drive(paths: List[Path], repo_root: Path) -> None:
    """Pull expected outputs from Drive (if registered) to enable skipping."""
    for p in paths:
        if p.exists():
            continue
        rel = str(p.relative_to(repo_root))
        try:
            ensure_local_path(rel)
        except Exception:
            pass


def save_rating_hist_plot(rating_hist: Counter, out_path: Path, title: str) -> None:
    if not rating_hist:
        logger.warning("No ratings to plot.")
        return
    items = sorted(rating_hist.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved rating hist: %s", out_path)


def save_helpful_hist_plot(helpful_hist: Counter, out_path: Path, title: str) -> None:
    if not helpful_hist:
        logger.warning("No helpful_votes to plot.")
        return
    items = sorted(helpful_hist.items())
    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Helpful votes (clipped)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved helpful hist: %s", out_path)


def read_all_categories_from_file(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
