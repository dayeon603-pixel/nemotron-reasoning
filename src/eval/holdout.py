"""Seeded, stratified train/val split for the Nemotron Reasoning Challenge.

The split is performed ONCE, persisted to ``data/val_ids.json``, and
reloaded on every subsequent call.  This guarantees:

* No leakage — val IDs are fixed before any feature engineering or
  prompt tuning is done on the training data.
* Reproducibility — same seed + same CSV always yields the same split.
* Stratification — domain distribution in val mirrors train, so per-
  domain accuracy estimates are not distorted by accidental clustering.

Split ratio: 80 % train / 20 % val (configurable via ``val_fraction``).
Default seed: 42.

TEMPORAL NOTE: The Nemotron challenge data is synthetically generated
(not a time series), so stratified random splitting is appropriate here.
If you add any real-market or time-ordered data in future, switch to a
temporal split and update this module.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.eval.cv import infer_domain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — no hardcoded paths in function bodies
# ---------------------------------------------------------------------------

_DEFAULT_SEED: int = 42
_DEFAULT_VAL_FRACTION: float = 0.20
_VAL_IDS_FILENAME: str = "val_ids.json"

# Resolve data/ relative to the repository root (two levels above this file:
# src/eval/holdout.py -> src/eval -> src -> repo root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR: Path = _REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class SplitResult(NamedTuple):
    """Train and validation DataFrames from a seeded stratified split.

    Attributes:
        train: Training subset.
        val: Validation subset.
        val_ids_path: Absolute path where val IDs were persisted.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    val_ids_path: Path


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def make_split(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    val_fraction: float = _DEFAULT_VAL_FRACTION,
    seed: int = _DEFAULT_SEED,
    data_dir: Path = _DEFAULT_DATA_DIR,
    force_resplit: bool = False,
) -> SplitResult:
    """Create or reload a seeded, stratified train/val split.

    On first call (or when ``force_resplit=True``), assigns val IDs by
    stratified sampling within each inferred domain bucket and persists
    them to ``data_dir / val_ids.json``.  On subsequent calls, reloads
    from that file — the split is frozen.

    Args:
        df: Full training DataFrame.  Must contain an ``'id'`` column.
            Should contain a prompt column for domain-stratification; if
            absent, all rows are treated as domain ``'other'``.
        prompt_col: Name of the column containing prompt text.
        val_fraction: Proportion of each domain bucket assigned to val.
            Default ``0.20`` (20 %).
        seed: Random seed set before any sampling.  Default ``42``.
        data_dir: Directory for persisting ``val_ids.json``.
        force_resplit: If ``True``, ignore any existing ``val_ids.json``
            and recompute.

    Returns:
        A ``SplitResult`` namedtuple ``(train, val, val_ids_path)``.

    Raises:
        ValueError: If ``'id'`` column is absent from ``df``.
        ValueError: If ``val_fraction`` is outside ``(0, 1)``.
    """
    if "id" not in df.columns:
        raise ValueError("DataFrame must contain an 'id' column.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")

    data_dir.mkdir(parents=True, exist_ok=True)
    val_ids_path: Path = data_dir / _VAL_IDS_FILENAME

    if val_ids_path.exists() and not force_resplit:
        logger.info("Reloading existing val split from %s", val_ids_path)
        val_ids: list[str] = _load_val_ids(val_ids_path)
    else:
        logger.info(
            "Computing new stratified split (seed=%d, val_fraction=%.2f).",
            seed,
            val_fraction,
        )
        val_ids = _stratified_sample(df, prompt_col, val_fraction, seed)
        _persist_val_ids(val_ids, val_ids_path)
        logger.info("Val IDs persisted to %s (%d rows).", val_ids_path, len(val_ids))

    val_id_set: set[str] = set(val_ids)
    val_mask: pd.Series = df["id"].astype(str).isin(val_id_set)

    val_df: pd.DataFrame = df[val_mask].reset_index(drop=True)
    train_df: pd.DataFrame = df[~val_mask].reset_index(drop=True)

    logger.info(
        "Split complete — train: %d rows, val: %d rows.", len(train_df), len(val_df)
    )
    return SplitResult(train=train_df, val=val_df, val_ids_path=val_ids_path)


def load_split(
    df: pd.DataFrame,
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> SplitResult:
    """Reload a previously persisted split without recomputing.

    Convenience wrapper around ``make_split`` with ``force_resplit=False``.
    Raises if ``val_ids.json`` does not exist (i.e., ``make_split`` has
    never been called for this data directory).

    Args:
        df: Full training DataFrame (same as used in ``make_split``).
        data_dir: Directory where ``val_ids.json`` was written.

    Returns:
        A ``SplitResult`` namedtuple ``(train, val, val_ids_path)``.

    Raises:
        FileNotFoundError: If ``val_ids.json`` does not exist in
            ``data_dir``.
    """
    val_ids_path: Path = data_dir / _VAL_IDS_FILENAME
    if not val_ids_path.exists():
        raise FileNotFoundError(
            f"No persisted split found at {val_ids_path}. "
            "Call make_split() first."
        )
    return make_split(df, data_dir=data_dir, force_resplit=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stratified_sample(
    df: pd.DataFrame,
    prompt_col: str,
    val_fraction: float,
    seed: int,
) -> list[str]:
    """Sample val IDs stratified by inferred domain.

    Args:
        df: Full DataFrame with at minimum an ``'id'`` column.
        prompt_col: Column containing prompt text (may be absent; falls
            back to treating everything as domain ``'other'``).
        val_fraction: Fraction of each stratum to assign to val.
        seed: Random seed — set via ``random.seed`` before any sampling.

    Returns:
        List of string IDs assigned to the val split.
    """
    # Seed before ANY sampling — reproducibility mandate
    random.seed(seed)

    has_prompt: bool = prompt_col in df.columns

    # Group IDs by domain
    buckets: defaultdict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        row_id: str = str(row["id"])
        if has_prompt:
            domain: str = infer_domain(str(row[prompt_col]))
        else:
            domain = "other"
        buckets[domain].append(row_id)

    val_ids: list[str] = []
    for domain, ids in sorted(buckets.items()):  # sorted for determinism
        random.shuffle(ids)
        n_val: int = max(1, round(len(ids) * val_fraction))
        val_ids.extend(ids[:n_val])
        logger.debug(
            "Domain '%s': %d total, %d assigned to val.", domain, len(ids), n_val
        )

    return val_ids


def _persist_val_ids(val_ids: list[str], path: Path) -> None:
    """Write val IDs to a JSON file (sorted for deterministic diffs).

    Args:
        val_ids: List of ID strings.
        path: Destination file path.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sorted(val_ids), fh, indent=2)


def _load_val_ids(path: Path) -> list[str]:
    """Load val IDs from a persisted JSON file.

    Args:
        path: Path to the JSON file written by ``_persist_val_ids``.

    Returns:
        List of ID strings.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is malformed.
    """
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
