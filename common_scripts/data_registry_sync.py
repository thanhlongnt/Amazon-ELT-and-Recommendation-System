#!/usr/bin/env python
"""
Scan local data/ and shared Google Drive folder and auto-generate configs/data_registry.yaml.

Usage:

  python common_scripts/data_registry_sync.py --mode resync

This will:
  - connect to Google Drive
  - walk the shared folder tree (assumed to mirror `data/` structure)
  - scan local `data/` tree
  - build / overwrite configs/data_registry.yaml so that `data_io.ensure_local(...)`
    knows how to download files by key.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys
import yaml
from pydrive2.drive import GoogleDrive

# Make sure we can import siblings (data_io.py) no matter where we run it from
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import _get_drive_and_root

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REGISTRY_PATH = REPO_ROOT / "configs" / "data_registry.yaml"
DRIVE_CFG_PATH = REPO_ROOT / "configs" / "drive_config.yaml"
CLIENT_SECRETS_PATH = REPO_ROOT / "configs" / "client_secrets.json"

def load_drive_root_id() -> str:
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
    """
    Use the shared Drive client from data_io, which caches credentials and only
    does browser-based auth when necessary.
    """
    drive, _ = _get_drive_and_root()
    return drive


def walk_drive_tree(drive: GoogleDrive, root_folder_id: str) -> Dict[str, str]:
    """
    Recursively walk the shared Drive folder and return mapping:
      'data/processed/Category/file.ext' -> file_id
    We assume the Drive folder mirrors the `data/` substructure,
    so root_folder_id corresponds to `data/`.
    """
    relpath_to_id: Dict[str, str] = {}

    def _walk(folder_id: str, curr_rel: Path):
        # List children of this folder
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
                rel_path = Path("data") / curr_rel / name  # data/... path
                relpath_to_id[str(rel_path)] = fid

    _walk(root_folder_id, Path())
    return relpath_to_id


def scan_local_data() -> Dict[str, bool]:
    """
    Return set-like dict:
      'data/processed/.../file.ext' -> True
    for all local files under data/
    """
    local_map: Dict[str, bool] = {}
    if not DATA_DIR.exists():
        return local_map

    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO_ROOT)
            local_map[str(rel)] = True
    return local_map


def infer_namespace_key(rel_path: str) -> Tuple[str, str]:
    """
    Heuristic to map a relative path like:
      'data/processed/Automotive/user_counts_Automotive.parquet'
    to (namespace='processed', key='user_counts_Automotive').

    This matches how data_io currently expects the registry to be structured.
    """
    p = Path(rel_path)
    parts = p.parts  # ('data','processed','Automotive','user_counts_Automotive.parquet',...)

    if len(parts) >= 3 and parts[1] == "processed":
        namespace = "processed"
        key = p.stem  # e.g. 'user_counts_Automotive', 'top_users'
        return namespace, key

    if len(parts) >= 3 and parts[1] == "raw":
        namespace = "raw"
        key = p.name
        return namespace, key

    # global files under data/global/
    if len(parts) >= 3 and parts[1] == "global":
        namespace = "processed"
        key = p.stem
        return namespace, key

    # lock files under data/locks/...
    if len(parts) >= 3 and parts[1] == "locks":
        namespace = "locks"
        key = p.stem
        return namespace, key

    return "", ""


def build_registry(
    drive_root_id: str,
    local_map: Dict[str, bool],
    remote_map: Dict[str, str],
) -> Dict:
    """
    Build the full registry dict to dump as YAML.
    """
    registry = {
        "drive_root_folder_id": drive_root_id,
        "raw": {},
        "processed": {},
        "locks": {},
    }

    all_paths = sorted(set(local_map.keys()) | set(remote_map.keys()))

    for rel in all_paths:
        ns, key = infer_namespace_key(rel)
        if not ns or not key:
            # Unknown / unhandled location -> skip
            continue

        ns_dict = registry.setdefault(ns, {})
        entry = ns_dict.setdefault(key, {})
        entry["local_path"] = rel
        fid = remote_map.get(rel)
        if fid:
            entry["drive_file_id"] = fid

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local data/ and shared Google Drive into data_registry.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=["resync"],
        default="resync",
        help="Currently only 'resync' is supported: rebuild the registry from scratch.",
    )
    args = parser.parse_args()

    drive_root_id = load_drive_root_id()
    drive = get_drive()

    print(">>> Scanning Google Drive shared data folder...")
    remote_map = walk_drive_tree(drive, drive_root_id)
    print(f"    Found {len(remote_map)} remote files under shared drive root.")

    print(">>> Scanning local data/ tree...")
    local_map = scan_local_data()
    print(f"    Found {len(local_map)} local files under data/")

    print(">>> Building data registry...")
    registry = build_registry(drive_root_id, local_map, remote_map)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=True)

    print(f"[OK] Wrote registry to {REGISTRY_PATH}")
    print("You can now use data_io.ensure_local(...) from your scripts.")


if __name__ == "__main__":
    main()