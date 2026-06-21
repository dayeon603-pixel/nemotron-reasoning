"""Validate and (optionally) re-package the LoRA adapter into submission.zip.

NOTE: src/sft_train.py already calls _package_submission() at the end of
training and writes submission.zip with adapter files at zip root.  This
script is therefore a VALIDATION tool, not a packaging tool.  It:

  1. Confirms adapter_config.json exists in the adapter directory.
  2. Confirms lora_rank (r) <= 32 (challenge hard limit).
  3. Confirms adapter_model.safetensors is present (required by scorer).
  4. Inspects the existing submission.zip (if present) and validates its
     internal layout (files at zip root, no subdirectory prefix).
  5. If --repackage is passed, re-zips the adapter dir to submission.zip
     (idempotent, overwrites any existing file).

This is intentionally conservative — it never silently mutates a zip that
sft_train already produced correctly; it just validates and optionally
overwrites on explicit request.

Usage:
    # Validate only (default):
    python scripts/package_submission.py --adapter-dir outputs/lora_adapter/best

    # Validate and re-zip (e.g. after manual weight edits):
    python scripts/package_submission.py \
        --adapter-dir outputs/lora_adapter/best \
        --zip-path submission.zip \
        --repackage
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

__all__: list[str] = []

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

REQUIRED_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
MAX_LORA_RANK: int = 32       # challenge hard constraint


# ── validation helpers ────────────────────────────────────────────────────────

def validate_adapter_dir(adapter_dir: Path) -> dict[str, object]:
    """Validate the adapter directory produced by peft model.save_pretrained().

    Checks:
      - adapter_config.json and adapter_model.safetensors both exist.
      - adapter_config.json is valid JSON containing an 'r' key.
      - The LoRA rank 'r' is <= MAX_LORA_RANK (32).

    Args:
        adapter_dir: Path to the saved PEFT adapter directory.

    Returns:
        Parsed adapter_config.json as a dict.

    Raises:
        FileNotFoundError: If adapter_dir or required files are missing.
        ValueError: If lora rank > MAX_LORA_RANK or adapter_config.json
            is malformed / missing the 'r' key.
    """
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}. "
            "Run src/sft_train.py first."
        )

    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Required adapter file missing: {config_file}. "
            "Ensure model.save_pretrained() completed successfully."
        )
    # Accept both monolithic and sharded weight layouts.
    if not list(adapter_dir.glob("*.safetensors")):
        raise FileNotFoundError(
            f"No *.safetensors weights found in {adapter_dir}. "
            "Ensure model.save_pretrained() wrote safetensors output."
        )

    config_path = adapter_dir / "adapter_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            adapter_config: dict[str, object] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"adapter_config.json is not valid JSON: {config_path}"
        ) from exc

    if "r" not in adapter_config:
        raise ValueError(
            f"adapter_config.json does not contain an 'r' (rank) key. "
            f"Keys found: {sorted(adapter_config.keys())}"
        )

    lora_rank: int = int(adapter_config["r"])  # type: ignore[arg-type]
    if lora_rank > MAX_LORA_RANK:
        raise ValueError(
            f"LoRA rank r={lora_rank} exceeds challenge maximum of {MAX_LORA_RANK}. "
            "Retrain with lora_r <= 32 in configs/train.yaml."
        )

    logger.info(
        "Adapter validated: dir=%s  rank=%d  target_modules=%s",
        adapter_dir,
        lora_rank,
        adapter_config.get("target_modules", "unknown"),
    )
    return adapter_config


def validate_zip_layout(zip_path: Path) -> None:
    """Verify submission.zip has required files at its root (no subdirectory).

    The Kaggle scorer expects adapter files to appear without any directory
    prefix inside the zip — e.g. 'adapter_config.json', not
    'best/adapter_config.json'.

    Args:
        zip_path: Path to submission.zip.

    Raises:
        FileNotFoundError: If zip_path does not exist.
        ValueError: If required files are missing from the zip root or are
            nested under a subdirectory.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"submission.zip not found: {zip_path}. "
            "Run src/sft_train.py or pass --repackage to create it."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = set(zf.namelist())

    logger.info("submission.zip contents: %s", sorted(zip_names))

    for fname in REQUIRED_ADAPTER_FILES:
        if fname not in zip_names:
            raise ValueError(
                f"submission.zip is missing '{fname}' at zip root. "
                f"Found: {sorted(zip_names)}. "
                "Re-run with --repackage to rebuild the zip from the adapter dir."
            )

    # Warn (don't fail) if there are unexpected extra files — scorer may ignore them
    extra = zip_names - set(REQUIRED_ADAPTER_FILES)
    if extra:
        logger.warning(
            "submission.zip contains extra files not required by scorer: %s",
            sorted(extra),
        )


def repackage_submission(adapter_dir: Path, zip_path: Path) -> None:
    """Re-zip the adapter directory into submission.zip (files at root).

    This replicates the exact logic of src.sft_train._package_submission.
    Only call this if you need to override the zip that sft_train produced.

    Args:
        adapter_dir: Directory containing adapter_config.json and
            adapter_model.safetensors.
        zip_path: Destination path for submission.zip (overwritten if exists).

    Raises:
        FileNotFoundError: If required adapter files are missing from adapter_dir.
    """
    for fname in REQUIRED_ADAPTER_FILES:
        fpath = adapter_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Cannot repackage: {fpath} not found."
            )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in REQUIRED_ADAPTER_FILES:
            zf.write(adapter_dir / fname, arcname=fname)

    logger.info(
        "submission.zip repackaged: %s (%d bytes)",
        zip_path,
        zip_path.stat().st_size,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validate LoRA adapter and submission.zip for Nemotron Reasoning Challenge. "
            "sft_train.py already produces submission.zip — this script validates it. "
            "Use --repackage only if you need to override the zip."
        )
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/lora_adapter/best"),
        help="Path to adapter directory (default: outputs/lora_adapter/best).",
    )
    p.add_argument(
        "--zip-path",
        type=Path,
        default=Path("submission.zip"),
        help="Path to submission.zip (default: submission.zip).",
    )
    p.add_argument(
        "--repackage",
        action="store_true",
        default=False,
        help=(
            "Re-zip the adapter dir into --zip-path. "
            "Default: validate only (do not modify the zip)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to sys.argv if None).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _build_parser().parse_args(argv)
    adapter_dir: Path = args.adapter_dir
    zip_path: Path = args.zip_path
    repackage: bool = args.repackage

    # Always validate the adapter directory first
    adapter_config = validate_adapter_dir(adapter_dir)
    logger.info(
        "Adapter config OK — rank=%s, base_model=%s",
        adapter_config.get("r"),
        adapter_config.get("base_model_name_or_path", "unknown"),
    )

    if repackage:
        logger.info("--repackage requested: rebuilding %s from %s", zip_path, adapter_dir)
        repackage_submission(adapter_dir, zip_path)

    # Validate zip layout regardless of --repackage
    validate_zip_layout(zip_path)
    logger.info("submission.zip layout OK: %s", zip_path)
    logger.info(
        "All checks passed. Ready to submit:\n"
        "  kaggle competitions submit "
        "-c nvidia-nemotron-model-reasoning-challenge "
        "-f %s -m 'LoRA rank-%s SFT on synthetic data'",
        zip_path,
        adapter_config.get("r"),
    )


if __name__ == "__main__":
    main()
