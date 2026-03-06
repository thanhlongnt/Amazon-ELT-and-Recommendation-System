"""Scan local data/ and shared Google Drive folder; auto-generate data_registry.yaml.

Usage::

    python -m amazon_next_category.io.registry_sync --mode resync

This will:
- connect to Google Drive
- walk the shared folder tree (assumed to mirror the ``data/`` structure)
- scan the local ``data/`` tree
- build / overwrite ``configs/data_registry.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import yaml
from pydrive2.drive import GoogleDrive

from amazon_next_category.io.data_io import _get_drive_and_root

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
REGISTRY_PATH = REPO_ROOT / "configs" / "data_registry.yaml"
DRIVE_CFG_PATH = REPO_ROOT / "configs" / "drive_config.yaml"


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------


def load_drive_root_id() -> str:
    """Read the shared Drive root folder ID from drive_config.yaml."""
    if not DRIVE_CFG_PATH.exists():
        raise FileNotFoundError(
            f"Missing drive_config.yaml at {DRIVE_CFG_PATH}. "
            "Create it with a `drive_root_folder_id` field."
        )
    with open(DRIVE_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root_id = cfg.get("drive_root_folder_id")
    if not root_id:
        raise ValueError("drive_root_folder_id missing in drive_config.yaml")
    return root_id


def get_drive() -> GoogleDrive:
    """Return the shared Drive client (credentials cached by data_io)."""
    drive, _ = _get_drive_and_root()
    return drive


# ---------------------------------------------------------------------------
# Drive tree walker
# ---------------------------------------------------------------------------


def walk_drive_tree(drive: GoogleDrive, root_folder_id: str) -> Dict[str, str]:
    """Recursively walk the shared Drive folder.

    Returns a mapping ``'data/processed/.../file.ext' -> file_id``.
    The Drive folder is assumed to mirror the ``data/`` structure, so
    *root_folder_id* corresponds to ``data/``.
    """
    relpath_to_id: Dict[str, str] = {}

    def _walk(folder_id: str, curr_rel: Path) -> None:
        file_list = drive.ListFile(
            {"q": f"'{folder_id}' in parents and trashed=false"}
        ).GetList()

        for f in file_list:
            name = f["title"]
            mime = f["mimeType"]
            fid = f["id"]
            if mime == "application/vnd.google-apps.folder":
                _walk(fid, curr_rel / name)
            else:
                rel_path = Path("data") / curr_rel / name
                relpath_to_id[str(rel_path)] = fid

    _walk(root_folder_id, Path())
    return relpath_to_id


# ---------------------------------------------------------------------------
# Local scan
# ---------------------------------------------------------------------------


def scan_local_data() -> Dict[str, bool]:
    """Return ``{rel_path: True}`` for all files under ``data/``."""
    local_map: Dict[str, bool] = {}
    if not DATA_DIR.exists():
        return local_map

    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT)
            local_map[str(rel)] = True
    return local_map


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def infer_namespace_key(rel_path: str) -> Tuple[str, str]:
    """Heuristically map a relative path to ``(namespace, key)``.

    Examples::

        'data/processed/Automotive/user_counts_Automotive.parquet'
        -> ('processed', 'user_counts_Automotive')

        'data/raw/all_categories.txt'
        -> ('raw', 'all_categories.txt')
    """
    p = Path(rel_path)
    parts = p.parts

    if len(parts) >= 3 and parts[1] == "processed":
        return "processed", p.stem

    if len(parts) >= 3 and parts[1] == "raw":
        return "raw", p.name

    if len(parts) >= 3 and parts[1] == "global":
        return "processed", p.stem

    if len(parts) >= 3 and parts[1] == "locks":
        return "locks", p.stem

    return "", ""


def build_registry(
    drive_root_id: str,
    local_map: Dict[str, bool],
    remote_map: Dict[str, str],
) -> Dict:
    """Build the full registry dict ready to dump as YAML."""
    registry: Dict = {
        "drive_root_folder_id": drive_root_id,
        "raw": {},
        "processed": {},
        "locks": {},
    }

    all_paths = sorted(set(local_map.keys()) | set(remote_map.keys()))

    for rel in all_paths:
        ns, key = infer_namespace_key(rel)
        if not ns or not key:
            continue

        ns_dict = registry.setdefault(ns, {})
        entry = ns_dict.setdefault(key, {})
        entry["local_path"] = rel
        fid = remote_map.get(rel)
        if fid:
            entry["drive_file_id"] = fid

    return registry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Sync local data/ and shared Google Drive into data_registry.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=["resync"],
        default="resync",
        help="Currently only 'resync' is supported.",
    )
    args = parser.parse_args()  # noqa: F841

    drive_root_id = load_drive_root_id()
    drive = get_drive()

    logger.info("Scanning Google Drive shared data folder...")
    remote_map = walk_drive_tree(drive, drive_root_id)
    logger.info("Found %d remote files under shared drive root.", len(remote_map))

    logger.info("Scanning local data/ tree...")
    local_map = scan_local_data()
    logger.info("Found %d local files under data/", len(local_map))

    logger.info("Building data registry...")
    registry = build_registry(drive_root_id, local_map, remote_map)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=True)

    logger.info("Wrote registry to %s", REGISTRY_PATH)
    logger.info("You can now use data_io.ensure_local(...) from your scripts.")


if __name__ == "__main__":
    main()
