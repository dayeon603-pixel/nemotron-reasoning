"""Tests for scripts/package_submission.py.

Critical paths tested:
  - validate_adapter_dir: missing dir, missing files, bad JSON, rank > 32, rank OK.
  - validate_zip_layout: missing zip, missing root files, extra files (warn only).
  - repackage_submission: produces a zip with correct root-level layout.
  - main(): --repackage flag round-trip.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

# package_submission is a script, not a module under src/.
# Import via importlib to avoid adding scripts/ to sys.path globally.
import importlib.util
import sys


def _import_pkg_sub() -> object:
    """Dynamically import scripts/package_submission.py."""
    spec = importlib.util.spec_from_file_location(
        "package_submission",
        Path(__file__).resolve().parents[1] / "scripts" / "package_submission.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


PKG = _import_pkg_sub()

validate_adapter_dir = PKG.validate_adapter_dir  # type: ignore[attr-defined]
validate_zip_layout = PKG.validate_zip_layout  # type: ignore[attr-defined]
repackage_submission = PKG.repackage_submission  # type: ignore[attr-defined]
MAX_LORA_RANK: int = PKG.MAX_LORA_RANK  # type: ignore[attr-defined]
REQUIRED_ADAPTER_FILES: tuple[str, ...] = PKG.REQUIRED_ADAPTER_FILES  # type: ignore[attr-defined]


# ── fixtures ──────────────────────────────────────────────────────────────────

def _write_valid_adapter(adapter_dir: Path, rank: int = 16) -> None:
    """Write a minimal valid adapter directory."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "r": rank,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    # Dummy safetensors (just needs to exist; content irrelevant for validation)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 16)


# ── validate_adapter_dir ──────────────────────────────────────────────────────

class TestValidateAdapterDir:
    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_adapter_dir(missing)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            validate_adapter_dir(adapter_dir)

    def test_missing_safetensors_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"r": 8, "target_modules": []}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(FileNotFoundError, match=r"\*\.safetensors"):
            validate_adapter_dir(adapter_dir)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("NOT_JSON")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        with pytest.raises(ValueError, match="not valid JSON"):
            validate_adapter_dir(adapter_dir)

    def test_missing_rank_key_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_alpha": 16}))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        with pytest.raises(ValueError, match="'r'"):
            validate_adapter_dir(adapter_dir)

    def test_rank_exceeds_max_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=64)
        with pytest.raises(ValueError, match="exceeds challenge maximum"):
            validate_adapter_dir(adapter_dir)

    def test_rank_exactly_max_passes(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=MAX_LORA_RANK)  # 32
        config = validate_adapter_dir(adapter_dir)
        assert config["r"] == MAX_LORA_RANK

    def test_rank_below_max_passes(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=16)
        config = validate_adapter_dir(adapter_dir)
        assert config["r"] == 16

    def test_returns_config_dict(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=8)
        config = validate_adapter_dir(adapter_dir)
        assert isinstance(config, dict)
        assert config["r"] == 8


# ── validate_zip_layout ───────────────────────────────────────────────────────

class TestValidateZipLayout:
    def test_missing_zip_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_zip_layout(tmp_path / "missing.zip")

    def test_zip_missing_config_raises(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_model.safetensors", b"\x00" * 4)
        with pytest.raises(ValueError, match="adapter_config.json"):
            validate_zip_layout(zip_path)

    def test_zip_missing_safetensors_raises(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", json.dumps({"r": 8}))
        with pytest.raises(ValueError, match="adapter_model.safetensors"):
            validate_zip_layout(zip_path)

    def test_nested_path_not_treated_as_root(self, tmp_path: Path) -> None:
        """Files under a subdir do not satisfy root-level requirement."""
        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("best/adapter_config.json", json.dumps({"r": 8}))
            zf.writestr("best/adapter_model.safetensors", b"\x00")
        # Both required files are missing at root
        with pytest.raises(ValueError, match="adapter_config.json"):
            validate_zip_layout(zip_path)

    def test_valid_zip_passes(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", json.dumps({"r": 8}))
            zf.writestr("adapter_model.safetensors", b"\x00" * 4)
        # Should not raise
        validate_zip_layout(zip_path)

    def test_extra_files_do_not_fail(self, tmp_path: Path) -> None:
        """Extra files in zip should warn but not raise."""
        zip_path = tmp_path / "submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", json.dumps({"r": 8}))
            zf.writestr("adapter_model.safetensors", b"\x00" * 4)
            zf.writestr("tokenizer_config.json", "{}")  # extra, harmless
        validate_zip_layout(zip_path)  # must not raise


# ── repackage_submission ──────────────────────────────────────────────────────

class TestRepackageSubmission:
    def test_creates_valid_zip(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=16)
        zip_path = tmp_path / "submission.zip"
        repackage_submission(adapter_dir, zip_path)
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            assert set(zf.namelist()) == set(REQUIRED_ADAPTER_FILES)

    def test_files_at_root_not_nested(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=8)
        zip_path = tmp_path / "submission.zip"
        repackage_submission(adapter_dir, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                # No path separators in arcnames -> files are at root
                assert "/" not in name, f"File nested at: {name}"

    def test_missing_adapter_file_raises(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # Only write config, not safetensors
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 8}))
        zip_path = tmp_path / "submission.zip"
        with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
            repackage_submission(adapter_dir, zip_path)

    def test_overwrite_existing_zip(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=4)
        zip_path = tmp_path / "submission.zip"
        # Write a stale zip first
        zip_path.write_bytes(b"stale")
        repackage_submission(adapter_dir, zip_path)
        # Must be a valid zip now
        assert zipfile.is_zipfile(zip_path)


# ── round-trip: validate -> repackage -> validate ─────────────────────────────

class TestRoundTrip:
    def test_validate_repackage_validate(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        _write_valid_adapter(adapter_dir, rank=32)
        zip_path = tmp_path / "submission.zip"

        validate_adapter_dir(adapter_dir)
        repackage_submission(adapter_dir, zip_path)
        validate_zip_layout(zip_path)  # no exceptions -> green
