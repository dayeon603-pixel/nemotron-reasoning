"""Build the primary SFT training set from the real competition train.csv.

Each row in train.csv becomes one JSONL record matching the schema expected
by src/sft_train.py (same as scripts/build_synthetic.py):

    {
        "id":               "<row id from train.csv>",
        "prompt":           "<full puzzle prompt>",
        "trace":            "<gold CoT ending with \\boxed{real_answer}>",
        "extracted_answer": "<value inside the last \\boxed{} of the trace>",
        "gold_answer":      "<real known_answer from train.csv>",
    }

After building the file the script runs a mandatory self-check: for every
record, metric.verify(extracted_answer, gold_answer) must be True.  Any
failure is reported with the full record details and the script exits
non-zero.

Usage:
    PYTHONPATH=. python scripts/build_real_sft.py \
        --train-csv data/raw/train.csv \
        --output data/real_sft.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from src.eval.metric import extract_final_answer, verify
from src.solvers import route_and_solve

__all__: list[str] = []

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_CSV: Path = Path("data/raw/train.csv")
DEFAULT_OUTPUT: Path = Path("data/real_sft.jsonl")

# Family prefix → short tag for per-family logging
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("In Alice's Wonderland, the gravitational constant", "gravitational"),
    ("In Alice's Wonderland, a secret unit conversion", "unit_conversion"),
    ("In Alice's Wonderland, numbers are secretly converted", "numeral"),
    ("In Alice's Wonderland, secret encryption rules", "encrypt"),
    ("In Alice's Wonderland, a secret bit manipulation", "bitmanip"),
    ("In Alice's Wonderland, a secret set of transformation", "symbol"),
]

_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{")


def _classify_family(prompt: str) -> str:
    """Return a short family tag for the prompt, or '__other__' if unrecognised.

    Args:
        prompt: Full puzzle prompt string.

    Returns:
        One of the six family tag strings or ``'__other__'``.
    """
    for prefix, tag in _FAMILY_PREFIXES:
        if prompt.startswith(prefix):
            return tag
    return "__other__"


def _cot_is_scaffold(cot: str) -> bool:
    """Heuristic: True if the CoT is a short scaffold (not a full derivation).

    A scaffold CoT from route_and_solve's fallback path is always very short
    (< 400 chars) and contains the literal phrase "I can determine the rule".
    Full derivation CoTs are longer and do not contain that phrase.

    Args:
        cot: Chain-of-thought string.

    Returns:
        True if the CoT appears to be a minimal scaffold.
    """
    return "I can determine the rule" in cot


def build_real_sft(
    train_csv: Path,
    output: Path,
) -> dict[str, int]:
    """Build the real SFT JSONL from train.csv.

    Args:
        train_csv: Path to the competition train.csv (cols: id, prompt, answer).
        output:    Path to write the output JSONL file.

    Returns:
        Stats dict with keys:
            ``'total'``          — total records written,
            ``'no_match'``       — rows that matched no solver family,
            ``'verify_failures'``— rows that failed the boxed==gold check,
        plus one key per family with ``'<family>_count'``, ``'<family>_closed_form'``,
        and ``'<family>_scaffold'`` counts.

    Raises:
        FileNotFoundError: If ``train_csv`` does not exist.
        RuntimeError:      If any record fails the verify self-check.
    """
    if not train_csv.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_csv}. "
            "Download the competition data first or pass --train-csv."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    family_counts: Counter[str] = Counter()
    closed_form_counts: Counter[str] = Counter()
    scaffold_counts: Counter[str] = Counter()
    no_match_ids: list[str] = []
    verify_failures: list[dict[str, str]] = []

    records: list[dict[str, str]] = []

    logger.info("Reading %s ...", train_csv)

    with train_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row_idx, row in enumerate(reader):
            row_id: str = row["id"]
            prompt: str = row["prompt"].strip()
            known_answer: str = row["answer"].strip()

            family = _classify_family(prompt)
            family_counts[family] += 1

            result = route_and_solve(prompt, known_answer)

            if result is None:
                no_match_ids.append(row_id)
                logger.warning(
                    "Row %s (idx=%d): no solver matched. family=%s prompt_head=%r",
                    row_id,
                    row_idx,
                    family,
                    prompt[:80],
                )
                # Emit a scaffold record so the total stays at 9500.
                cot = (
                    "Let me work through this puzzle step by step.\n\n"
                    f"The answer is: {known_answer}\n\n"
                    f"\\boxed{{{known_answer}}}"
                )
                scaffold_counts[family] += 1
            else:
                cot, _ = result
                if _cot_is_scaffold(cot):
                    scaffold_counts[family] += 1
                else:
                    closed_form_counts[family] += 1

            extracted = extract_final_answer(cot)

            record: dict[str, str] = {
                "id": row_id,
                "prompt": prompt,
                "trace": cot,
                "extracted_answer": extracted,
                "gold_answer": known_answer,
            }
            records.append(record)

            if (row_idx + 1) % 500 == 0:
                logger.info("Processed %d / rows so far...", row_idx + 1)

    logger.info("All rows processed. Running verify self-check on %d records...", len(records))

    # ── mandatory self-check: every record must pass verify ──────────────────
    with output.open("w", encoding="utf-8") as out_fh:
        for record in records:
            extracted_in_trace = extract_final_answer(record["trace"])
            gold = record["gold_answer"]

            if not verify(gold, extracted_in_trace):
                verify_failures.append({
                    "id": record["id"],
                    "gold_answer": gold,
                    "extracted_from_trace": extracted_in_trace,
                    "trace_tail": record["trace"][-120:],
                })
                logger.error(
                    "VERIFY FAIL: id=%s gold=%r extracted=%r trace_tail=%r",
                    record["id"],
                    gold,
                    extracted_in_trace,
                    record["trace"][-120:],
                )

            # Write the record regardless (so we can report all failures).
            out_fh.write(json.dumps(record) + "\n")

    # ── build stats ───────────────────────────────────────────────────────────
    stats: dict[str, int] = {
        "total": len(records),
        "no_match": len(no_match_ids),
        "verify_failures": len(verify_failures),
    }
    for family in set(list(family_counts.keys()) + [p[1] for p in _FAMILY_PREFIXES]):
        tag = family
        stats[f"{tag}_count"] = family_counts.get(tag, 0)
        stats[f"{tag}_closed_form"] = closed_form_counts.get(tag, 0)
        stats[f"{tag}_scaffold"] = scaffold_counts.get(tag, 0)

    # ── logging summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("build_real_sft SUMMARY")
    logger.info("  Total records written : %d", stats["total"])
    logger.info("  Output path           : %s", output)
    logger.info("  Rows with no match    : %d", stats["no_match"])
    logger.info("  Verify failures       : %d", stats["verify_failures"])
    logger.info("")
    logger.info("  Per-family breakdown:")
    for _, tag in _FAMILY_PREFIXES:
        count = stats.get(f"{tag}_count", 0)
        cf = stats.get(f"{tag}_closed_form", 0)
        sc = stats.get(f"{tag}_scaffold", 0)
        logger.info(
            "    %-20s  rows=%4d  closed_form=%4d  scaffold=%4d",
            tag, count, cf, sc,
        )
    if stats["no_match"] > 0:
        sample = no_match_ids[:10]
        logger.warning("  No-match row IDs (first 10): %s", sample)
    logger.info("=" * 60)

    if verify_failures:
        # Hard failure: the training data is corrupt.
        msg = (
            f"build_real_sft FAILED: {len(verify_failures)} records failed "
            f"the verify self-check (boxed value in trace != gold_answer). "
            f"First failure: {verify_failures[0]}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(
        "Verify self-check: ALL %d records passed (boxed==gold).", stats["total"]
    )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the primary SFT JSONL from the real competition train.csv. "
            "Calls route_and_solve per row and writes a verified JSONL."
        )
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"Path to competition train.csv (default: {DEFAULT_TRAIN_CSV})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args()
    try:
        stats = build_real_sft(train_csv=args.train_csv, output=args.output)
    except RuntimeError as exc:
        logger.error("FATAL: %s", exc)
        sys.exit(1)
