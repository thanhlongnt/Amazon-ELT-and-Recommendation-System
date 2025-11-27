# pip install pydrive2 gdown google-api-python-client

import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "data_registry.yaml"
DRIVE_CONFIG_PATH = REPO_ROOT / "configs" / "drive_config.yaml"
CLIENT_SECRETS_PATH = REPO_ROOT / "configs" / "client_secrets.json"
DRIVE_CREDENTIALS_PATH = REPO_ROOT / "configs" / "pydrive_credentials.json"

_DATA_REGISTRY: Dict[str, Any] = {}
_LOADED = False

# Cached Google Drive client + root folder id
_DRIVE: Optional[GoogleDrive] = None
_DRIVE_ROOT_ID: Optional[str] = None


def resync_registry() -> None:
    """
    Automatically (re)build data_registry.yaml by calling data_registry_sync.py.

    Safe to call at the start of any script that needs registry-driven IO.
    Relies on our shared Google auth (which caches credentials).
    """
    sync_script = REPO_ROOT / "common_scripts" / "data_registry_sync.py"
    if not sync_script.exists():
        print(
            f"[data_io] WARNING: {sync_script} not found; "
            f"cannot resync data registry."
        )
        return

    print("[data_io] Resyncing data registry from Google Drive...")
    cmd = [sys.executable, str(sync_script), "--mode", "resync"]
    subprocess.run(cmd, check=True)
    global _LOADED
    _LOADED = False  # force reload on next _load_registry
    print("[data_io] Registry resync complete.")


def _load_registry(force: bool = False) -> None:
    """Lazy-load data_registry.yaml into memory."""
    global _DATA_REGISTRY, _LOADED
    if _LOADED and not force:
        return
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"data_registry.yaml not found at {CONFIG_PATH}. "
            "Run common_scripts/data_registry_sync.py or call resync_registry() first."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _DATA_REGISTRY = yaml.safe_load(f) or {}
    _LOADED = True


def get_entry(namespace: str, key: str) -> dict:
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
    """
    Ensure the data file for (namespace, key) exists locally.
    If not, and drive_file_id is present, download it via gdown.
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

    print(f"[data_io] Downloading {namespace}.{key} from Drive -> {local_path}")
    cmd = [
        "gdown",
        "--id",
        file_id,
        "-O",
        str(local_path),
    ]
    subprocess.run(cmd, check=True)
    return local_path


def ensure_local_path(rel_path: str) -> Path:
    """
    Convenience: ensure a file at `rel_path` exists locally, using registry to
    find its drive_file_id (if any).
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
    raise KeyError(
        f"No registry entry found with local_path={rel_path} in {CONFIG_PATH}"
    )


def _load_drive_root_id() -> str:
    """
    Read the shared Drive root folder ID from drive_config.yaml.
    This is the folder that corresponds to repo_root/data on Drive.
    """
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


def _get_drive_and_root():
    """
    Return a cached (GoogleDrive client, drive_root_folder_id) pair.

    This:
      - Loads client_secrets.json for OAuth config
      - Loads cached pydrive_credentials.json if present
      - Only calls LocalWebserverAuth() (browser) if credentials are missing
        or expired, and then saves them for future use.
    """
    global _DRIVE, _DRIVE_ROOT_ID
    if _DRIVE is not None and _DRIVE_ROOT_ID is not None:
        return _DRIVE, _DRIVE_ROOT_ID

    gauth = GoogleAuth()
    # Configure OAuth client
    gauth.LoadClientConfigFile(str(CLIENT_SECRETS_PATH))

    # Try to load cached credentials (no browser)
    if DRIVE_CREDENTIALS_PATH.exists():
        gauth.LoadCredentialsFile(str(DRIVE_CREDENTIALS_PATH))

    if gauth.credentials is None or gauth.access_token_expired:
        # Only in this case do we actually open the browser
        print("[data_io] Performing Google OAuth (one-time)...")
        gauth.LocalWebserverAuth()
        # Cache credentials for later runs
        gauth.SaveCredentialsFile(str(DRIVE_CREDENTIALS_PATH))
    else:
        # Use existing valid creds silently
        print("[data_io] Using cached Google Drive credentials.")

    drive = GoogleDrive(gauth)
    root_id = _load_drive_root_id()

    _DRIVE = drive
    _DRIVE_ROOT_ID = root_id
    return _DRIVE, _DRIVE_ROOT_ID


def upload_to_drive(local_path: Path) -> str:
    """
    Upload or update `local_path` under the shared Drive 'data/' tree,
    mirroring the repo's data/ subfolders.

    Returns the file_id of the uploaded Google Drive file.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

    rel = local_path.relative_to(REPO_ROOT)
    if not rel.parts or rel.parts[0] != "data":
        raise ValueError(
            f"upload_to_drive currently only supports files under 'data/' "
            f"(got: {rel})"
        )

    rel_under_data = rel.parts[1:]
    if not rel_under_data:
        raise ValueError(
            f"Unexpected path with no components under data/: {rel}"
        )

    drive, root_id = _get_drive_and_root()
    parent_id = root_id

    def _q_escape(s: str) -> str:
        return s.replace("'", "\\'")

    # Create/find folder chain under Drive root
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
    query = (
        f"'{parent_id}' in parents and "
        f"title = '{filename_q}' and trashed=false"
    )
    existing = drive.ListFile({"q": query}).GetList()

    if existing:
        gfile = existing[0]
        print(f"[data_io] Updating existing Drive file for {rel}")
    else:
        gfile = drive.CreateFile(
            {"title": filename, "parents": [{"id": parent_id}]}
        )
        print(f"[data_io] Creating new Drive file for {rel}")

    gfile.SetContentFile(str(local_path))
    gfile.Upload()
    file_id = gfile["id"]
    print(f"[data_io] Uploaded {rel} to Drive (file_id={file_id})")
    return file_id