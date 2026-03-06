"""Data I/O utilities: registry loading, local-file resolution, and Drive sync.

All Google Drive interaction (OAuth, upload, download) is centralised here.
Other modules should call :func:`ensure_local` or :func:`ensure_local_path`
rather than accessing Drive directly.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

logger = logging.getLogger(__name__)


def _q_escape(s: str) -> str:
    """Escape single quotes in Google Drive API title queries."""
    return s.replace("'", "\\'")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "data_registry.yaml"
DRIVE_CONFIG_PATH = REPO_ROOT / "configs" / "drive_config.yaml"
CLIENT_SECRETS_PATH = REPO_ROOT / "configs" / "client_secrets.json"
DRIVE_CREDENTIALS_PATH = REPO_ROOT / "configs" / "pydrive_credentials.json"

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_DATA_REGISTRY: Dict[str, Any] = {}
_LOADED: bool = False

_DRIVE: Optional[GoogleDrive] = None
_DRIVE_ROOT_ID: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def resync_registry() -> None:
    """Rebuild data_registry.yaml by invoking the registry-sync script.

    Safe to call at the start of any script that needs registry-driven I/O.
    Credentials are cached, so the browser OAuth only runs once.
    """
    sync_script = REPO_ROOT / "src" / "amazon_next_category" / "io" / "registry_sync.py"
    if not sync_script.exists():
        logger.warning("registry_sync.py not found at %s; cannot resync.", sync_script)
        return

    logger.info("Resyncing data registry from Google Drive...")
    cmd = [sys.executable, str(sync_script), "--mode", "resync"]
    subprocess.run(cmd, check=True)

    global _LOADED
    _LOADED = False  # force reload on next _load_registry
    logger.info("Registry resync complete.")


def _load_registry(force: bool = False) -> None:
    """Lazy-load data_registry.yaml into the module-level cache."""
    global _DATA_REGISTRY, _LOADED
    if _LOADED and not force:
        return
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"data_registry.yaml not found at {CONFIG_PATH}. "
            "Run registry_sync.py or call resync_registry() first."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _DATA_REGISTRY = yaml.safe_load(f) or {}
    _LOADED = True


def get_entry(namespace: str, key: str) -> dict:
    """Return the registry entry for ``namespace.key``."""
    _load_registry()
    try:
        ns = _DATA_REGISTRY[namespace]
    except KeyError:
        raise KeyError(f"No namespace '{namespace}' in data registry.")
    try:
        return ns[key]
    except KeyError:
        raise KeyError(f"No registry entry for {namespace}.{key}")


def ensure_local(namespace: str, key: str) -> Path:
    """Ensure the file for ``namespace.key`` is present locally.

    Downloads from Google Drive via *gdown* if the file is missing and a
    ``drive_file_id`` is registered.
    """
    entry = get_entry(namespace, key)
    local_rel = entry["local_path"]
    local_path = REPO_ROOT / local_rel
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return local_path

    file_id = entry.get("drive_file_id")
    if not file_id:
        raise FileNotFoundError(
            f"{local_path} is missing and no drive_file_id specified in registry "
            f"for {namespace}.{key}"
        )

    logger.info("Downloading %s.%s from Drive -> %s", namespace, key, local_path)
    cmd = ["gdown", "--id", file_id, "-O", str(local_path)]
    subprocess.run(cmd, check=True)
    return local_path


def ensure_local_path(rel_path: str) -> Path:
    """Resolve a repo-relative path to a local :class:`pathlib.Path`.

    Looks up the path in the registry and delegates to :func:`ensure_local`.
    """
    _load_registry()
    rel_path = str(Path(rel_path))
    for ns_name, ns_dict in _DATA_REGISTRY.items():
        if not isinstance(ns_dict, dict):
            continue
        for key, entry in ns_dict.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("local_path") == rel_path:
                return ensure_local(ns_name, key)
    raise KeyError(f"No registry entry found with local_path={rel_path} in {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Drive auth helpers
# ---------------------------------------------------------------------------


def _load_drive_root_id() -> str:
    """Read the shared Drive root folder ID from drive_config.yaml."""
    if not DRIVE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing drive_config.yaml at {DRIVE_CONFIG_PATH}. "
            "Create it with a `drive_root_folder_id` field."
        )
    with open(DRIVE_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root_id = cfg.get("drive_root_folder_id")
    if not root_id:
        raise ValueError("drive_root_folder_id missing in drive_config.yaml")
    return root_id


def _get_drive_and_root() -> tuple[GoogleDrive, str]:
    """Return a cached ``(GoogleDrive client, drive_root_folder_id)`` pair.

    Loads OAuth credentials from *client_secrets.json*, caches them in
    *pydrive_credentials.json*, and only opens the browser on the first run
    or when the token expires.
    """
    global _DRIVE, _DRIVE_ROOT_ID
    if _DRIVE is not None and _DRIVE_ROOT_ID is not None:
        return _DRIVE, _DRIVE_ROOT_ID

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(str(CLIENT_SECRETS_PATH))

    if DRIVE_CREDENTIALS_PATH.exists():
        gauth.LoadCredentialsFile(str(DRIVE_CREDENTIALS_PATH))

    if gauth.credentials is None or gauth.access_token_expired:
        logger.info("Performing Google OAuth (one-time)...")
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile(str(DRIVE_CREDENTIALS_PATH))
    else:
        logger.debug("Using cached Google Drive credentials.")

    drive = GoogleDrive(gauth)
    root_id = _load_drive_root_id()

    _DRIVE = drive
    _DRIVE_ROOT_ID = root_id
    return _DRIVE, _DRIVE_ROOT_ID


# ---------------------------------------------------------------------------
# Drive upload / existence / delete
# ---------------------------------------------------------------------------


def upload_to_drive(local_path: Path) -> str:
    """Upload (or update) *local_path* under the shared Drive ``data/`` tree.

    The remote path mirrors the repo's ``data/`` subfolder structure.

    Returns the Drive file ID of the uploaded file.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    rel = local_path.relative_to(REPO_ROOT)
    if not rel.parts or rel.parts[0] != "data":
        raise ValueError(f"upload_to_drive only supports files under 'data/' (got: {rel})")

    rel_under_data = rel.parts[1:]
    if not rel_under_data:
        raise ValueError(f"Unexpected path with no components under data/: {rel}")

    drive, root_id = _get_drive_and_root()
    parent_id = root_id

    for folder_name in rel_under_data[:-1]:
        folder_name_q = _q_escape(folder_name)
        query = (
            f"'{parent_id}' in parents and "
            f"title = '{folder_name_q}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        )
        folder_list = drive.ListFile({"q": query}).GetList()
        if folder_list:
            parent_id = folder_list[0]["id"]
        else:
            folder_metadata = {
                "title": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [{"id": parent_id}],
            }
            folder = drive.CreateFile(folder_metadata)
            folder.Upload()
            parent_id = folder["id"]

    filename = rel_under_data[-1]
    filename_q = _q_escape(filename)
    query = f"'{parent_id}' in parents and title = '{filename_q}' and trashed=false"
    existing = drive.ListFile({"q": query}).GetList()

    if existing:
        gfile = existing[0]
        logger.info("Updating existing Drive file for %s", rel)
    else:
        gfile = drive.CreateFile({"title": filename, "parents": [{"id": parent_id}]})
        logger.info("Creating new Drive file for %s", rel)

    gfile.SetContentFile(str(local_path))
    gfile.Upload()
    file_id: str = gfile["id"]
    logger.info("Uploaded %s to Drive (file_id=%s)", rel, file_id)
    return file_id


def remote_file_exists_by_rel_path(rel_path: str) -> bool:
    """Return ``True`` if a file at *rel_path* exists on Drive."""
    drive, root_id = _get_drive_and_root()
    rel = Path(rel_path)

    parts = rel.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    else:
        parts = ("data",) + parts

    if not parts:
        return False

    folder_parts, filename = parts[:-1], parts[-1]
    parent_id = root_id

    for folder_name in folder_parts:
        folder_name_q = _q_escape(folder_name)
        query = (
            f"'{parent_id}' in parents and "
            f"title = '{folder_name_q}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        )
        flist = drive.ListFile({"q": query}).GetList()
        if not flist:
            return False
        parent_id = flist[0]["id"]

    filename_q = _q_escape(filename)
    query = f"'{parent_id}' in parents and title = '{filename_q}' and trashed=false"
    files = drive.ListFile({"q": query}).GetList()
    return bool(files)


def delete_remote_by_rel_path(rel_path: str) -> None:
    """Delete a remote file at *rel_path* from Drive if it exists."""
    drive, root_id = _get_drive_and_root()
    rel = Path(rel_path)

    parts = rel.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    else:
        parts = ("data",) + parts

    if not parts:
        return

    folder_parts, filename = parts[:-1], parts[-1]
    parent_id = root_id

    for folder_name in folder_parts:
        folder_name_q = _q_escape(folder_name)
        query = (
            f"'{parent_id}' in parents and "
            f"title = '{folder_name_q}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        )
        flist = drive.ListFile({"q": query}).GetList()
        if not flist:
            return
        parent_id = flist[0]["id"]

    filename_q = _q_escape(filename)
    query = f"'{parent_id}' in parents and title = '{filename_q}' and trashed=false"
    files = drive.ListFile({"q": query}).GetList()
    for f in files:
        f.Delete()
