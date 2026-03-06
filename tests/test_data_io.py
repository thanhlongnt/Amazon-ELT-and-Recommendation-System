"""Unit tests for amazon_next_category.io.data_io.

These tests use monkeypatching to avoid real Drive / filesystem access.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

import amazon_next_category.io.data_io as data_io

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, registry: dict) -> Path:
    """Write a fake registry YAML under tmp_path/configs/ and return the path."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    reg_path = cfg_dir / "data_registry.yaml"
    with open(reg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f)
    return reg_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        registry = {
            "raw": {"all_categories.txt": {"local_path": "data/raw/all_categories.txt"}},
            "processed": {},
        }
        reg_path = _write_registry(tmp_path, registry)

        # Patch module-level CONFIG_PATH and reset _LOADED
        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            data_io._load_registry()
            assert "raw" in data_io._DATA_REGISTRY
            assert "all_categories.txt" in data_io._DATA_REGISTRY["raw"]

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "configs" / "missing.yaml"
        with (
            patch.object(data_io, "CONFIG_PATH", non_existent),
            patch.object(data_io, "_LOADED", False),
        ):
            with pytest.raises(FileNotFoundError, match="data_registry.yaml not found"):
                data_io._load_registry()


class TestGetEntry:
    def test_returns_entry(self, tmp_path: Path) -> None:
        registry = {
            "raw": {"cats": {"local_path": "data/raw/all_categories.txt"}},
            "processed": {},
        }
        reg_path = _write_registry(tmp_path, registry)

        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            entry = data_io.get_entry("raw", "cats")
            assert entry["local_path"] == "data/raw/all_categories.txt"

    def test_missing_namespace_raises(self, tmp_path: Path) -> None:
        registry = {"raw": {}, "processed": {}}
        reg_path = _write_registry(tmp_path, registry)

        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            with pytest.raises(KeyError, match="No namespace"):
                data_io.get_entry("nonexistent", "key")

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        registry = {"raw": {}, "processed": {}}
        reg_path = _write_registry(tmp_path, registry)

        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            with pytest.raises(KeyError, match="No registry entry"):
                data_io.get_entry("raw", "nonexistent")


class TestEnsureLocal:
    def test_returns_path_when_file_exists(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data" / "raw" / "all_categories.txt"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text("Books\n")

        registry = {
            "raw": {
                "all_categories.txt": {
                    "local_path": "data/raw/all_categories.txt",
                }
            },
            "processed": {},
        }
        reg_path = _write_registry(tmp_path, registry)

        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "REPO_ROOT", tmp_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            result = data_io.ensure_local("raw", "all_categories.txt")
            assert result == data_file
            assert result.exists()

    def test_raises_when_missing_no_drive_id(self, tmp_path: Path) -> None:
        registry = {
            "raw": {
                "cats": {
                    "local_path": "data/raw/all_categories.txt",
                    # no drive_file_id
                }
            },
            "processed": {},
        }
        reg_path = _write_registry(tmp_path, registry)

        with (
            patch.object(data_io, "CONFIG_PATH", reg_path),
            patch.object(data_io, "REPO_ROOT", tmp_path),
            patch.object(data_io, "_LOADED", False),
            patch.object(data_io, "_DATA_REGISTRY", {}),
        ):
            with pytest.raises(FileNotFoundError, match="no drive_file_id"):
                data_io.ensure_local("raw", "cats")
