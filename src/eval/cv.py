"""Cross-validation scoring harness for the Nemotron Reasoning Challenge.

Computes the official overall accuracy (fraction correct via ``verify``),
a per-domain accuracy breakdown, and a bootstrap 95 % CI on overall
accuracy.  All results are returned as a typed dataclass and logged in
a human-readable table — never printed.

Domain inference uses keyword heuristics applied to the *prompt* text.
Priority order (first match wins):
    binary  — keywords: 'binary', 'bit', 'bitwise', 'xor', 'and ', 'or ',
                        'shift', 'two\'s complement'
    cipher  — keywords: 'cipher', 'encrypt', 'decrypt', 'caesar',
                        'vigenere', 'rot', 'substitution'
    algebra — keywords: 'equation', 'solve', 'x =', 'x=', 'variable',
                        'linear', 'quadratic', 'polynomial'
    roman   — keywords: 'roman', 'numeral', 'xiv', 'xlii', 'mcm'
    other   — catch-all

IMPORTANT: domain inference operates on the *prompt* field, not on the
model prediction or the ground truth answer.  This avoids any look-ahead
leakage from the answer into the domain bucket.

Backtests and CV scores are hypotheses, not proofs.  Bootstrap CIs tell
you the uncertainty in your point estimate given this particular val split;
they do not account for distribution shift between val and private test.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.eval.metric import verify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Seed for bootstrap resampling — set once here, never overridden silently.
_BOOTSTRAP_SEED: int = 42
_N_BOOTSTRAP: int = 1000
_CI_LEVEL: float = 0.95

# Domain keyword heuristics — order within each tuple does not matter;
# the domain ORDER in _DOMAIN_KEYWORDS determines priority (first match wins).
_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "binary",
        (
            "binary",
            "bit",
            "bitwise",
            "xor",
            " and ",
            " or ",
            "shift",
            "two's complement",
            "twos complement",
            "base-2",
            "base 2",
        ),
    ),
    (
        "cipher",
        (
            "cipher",
            "encrypt",
            "decrypt",
            "caesar",
            "vigenere",
            "rot13",
            "rot ",
            "substitution",
            "plaintext",
            "ciphertext",
        ),
    ),
    (
        "algebra",
        (
            "equation",
            "solve",
            "x =",
            "x=",
            "variable",
            "linear",
            "quadratic",
            "polynomial",
            "coefficient",
            "expression",
        ),
    ),
    (
        "roman",
        (
            "roman",
            "numeral",
            "xiv",
            "xlii",
            "mcm",
            "lxxx",
            "mmxx",
            "dcc",
        ),
    ),
]

_DOMAIN_OTHER: str = "other"

# Required columns in each DataFrame
_PRED_REQUIRED: tuple[str, ...] = ("id", "prediction")
_SOL_REQUIRED: tuple[str, ...] = ("id", "answer")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainStats:
    """Per-domain accuracy statistics.

    Attributes:
        domain: Domain label (e.g. ``'binary'``, ``'algebra'``).
        n_total: Number of examples in this domain bucket.
        n_correct: Number of examples verified correct.
        accuracy: ``n_correct / n_total``, or ``float('nan')`` if
            ``n_total == 0``.
    """

    domain: str
    n_total: int
    n_correct: int
    accuracy: float


@dataclass(frozen=True)
class CVResult:
    """Full cross-validation result for one evaluation run.

    Attributes:
        overall_accuracy: Fraction of rows where ``verify`` returns
            ``True``.  Matches the official leaderboard ``score()``.
        n_total: Total number of evaluated rows.
        n_correct: Total correct rows.
        ci_lower: Lower bound of bootstrap 95 % CI on overall accuracy.
        ci_upper: Upper bound of bootstrap 95 % CI on overall accuracy.
        n_bootstrap: Number of bootstrap resamples used.
        domain_stats: Per-domain breakdown, sorted by domain name.
        missing_ids: IDs present in solutions but absent in predictions
            (counted as wrong).
    """

    overall_accuracy: float
    n_total: int
    n_correct: int
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    domain_stats: list[DomainStats] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------


def infer_domain(prompt: str) -> str:
    """Infer the problem domain from prompt text using keyword heuristics.

    Operates on lower-cased prompt text.  Priority order is fixed by
    ``_DOMAIN_KEYWORDS``; first match wins.  Returns ``'other'`` if no
    keyword matches.

    Args:
        prompt: Raw prompt string (question text).

    Returns:
        One of ``'binary'``, ``'cipher'``, ``'algebra'``, ``'roman'``,
        or ``'other'``.

    Raises:
        Nothing.

    Notes:
        # LOOKAHEAD RISK: None — domain is inferred from the prompt
        # only, which is available at prediction time.
    """
    lower: str = prompt.lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return domain
    return _DOMAIN_OTHER


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    correct_flags: Sequence[bool],
    n_boot: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
    ci_level: float = _CI_LEVEL,
) -> tuple[float, float]:
    """Compute a percentile bootstrap CI on accuracy.

    Uses simple percentile bootstrap (no bias correction).  Adequate for
    accuracy in [0, 1] when n >= 50; interpret with caution for smaller
    samples.

    Args:
        correct_flags: Boolean sequence where ``True`` == correct.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        ci_level: Confidence level, e.g. ``0.95``.

    Returns:
        ``(lower, upper)`` percentile bounds.

    Raises:
        ValueError: If ``correct_flags`` is empty.
    """
    if len(correct_flags) == 0:
        raise ValueError("correct_flags must not be empty.")

    rng = np.random.default_rng(seed)
    arr: np.ndarray = np.asarray(correct_flags, dtype=np.float64)
    n: int = len(arr)

    boot_means: np.ndarray = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = arr[indices].mean()

    alpha: float = 1.0 - ci_level
    lower: float = float(np.percentile(boot_means, 100 * alpha / 2))
    upper: float = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def run_cv(
    predictions: pd.DataFrame,
    solutions: pd.DataFrame,
    prompt_col: str = "prompt",
    n_boot: int = _N_BOOTSTRAP,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
) -> CVResult:
    """Compute official accuracy + per-domain breakdown + bootstrap CI.

    Replicates the official ``score()`` function exactly for overall
    accuracy: for each (id, answer) in solutions, look up the
    corresponding prediction, call ``verify(answer, prediction)``, and
    compute the fraction correct.  IDs missing from predictions are
    counted as incorrect.

    Domain breakdown requires a ``prompt`` column in *either* the
    solutions or the predictions DataFrame (checked in that order).
    If no prompt column is available, all rows are bucketed as ``'other'``.

    Args:
        predictions: DataFrame with at minimum columns ``['id',
            'prediction']``.  Extra columns are ignored.
        solutions: DataFrame with at minimum columns ``['id', 'answer']``.
            Optionally includes a ``prompt`` column used for domain
            inference.
        prompt_col: Name of the column containing prompt text.
        n_boot: Number of bootstrap resamples for the CI.
        bootstrap_seed: Seed passed to the bootstrap RNG.

    Returns:
        A ``CVResult`` dataclass.  See class docstring for field
        descriptions.

    Raises:
        ValueError: If required columns are missing from either DataFrame.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for col in _PRED_REQUIRED:
        if col not in predictions.columns:
            raise ValueError(
                f"predictions DataFrame missing required column '{col}'. "
                f"Got: {list(predictions.columns)}"
            )
    for col in _SOL_REQUIRED:
        if col not in solutions.columns:
            raise ValueError(
                f"solutions DataFrame missing required column '{col}'. "
                f"Got: {list(solutions.columns)}"
            )

    # ------------------------------------------------------------------
    # Build fast lookup: id -> prediction string
    # ------------------------------------------------------------------
    pred_map: dict[str, str] = {
        str(row["id"]): str(row["prediction"])
        for _, row in predictions.iterrows()  # iterrows acceptable here — building dict once
    }

    # ------------------------------------------------------------------
    # Determine prompt source for domain inference
    # ------------------------------------------------------------------
    has_prompt: bool = prompt_col in solutions.columns
    if not has_prompt and prompt_col in predictions.columns:
        # Merge prompt from predictions into solutions for domain lookup
        prompt_map: dict[str, str] = {
            str(row["id"]): str(row[prompt_col])
            for _, row in predictions.iterrows()
        }
    else:
        prompt_map = {}

    # ------------------------------------------------------------------
    # Evaluate every row in solutions (vectorised where possible)
    # ------------------------------------------------------------------
    correct_flags: list[bool] = []
    missing_ids: list[str] = []
    domain_correct: dict[str, int] = {}
    domain_total: dict[str, int] = {}

    for _, sol_row in solutions.iterrows():
        sol_id: str = str(sol_row["id"])
        stored_answer: str = str(sol_row["answer"])

        # Domain inference
        if has_prompt:
            domain = infer_domain(str(sol_row[prompt_col]))
        elif sol_id in prompt_map:
            domain = infer_domain(prompt_map[sol_id])
        else:
            domain = _DOMAIN_OTHER

        domain_total[domain] = domain_total.get(domain, 0) + 1
        domain_correct.setdefault(domain, 0)

        if sol_id not in pred_map:
            missing_ids.append(sol_id)
            correct_flags.append(False)
            logger.warning("ID %s missing from predictions — counted as wrong.", sol_id)
            continue

        predicted: str = pred_map[sol_id]
        is_correct: bool = verify(stored_answer, predicted)
        correct_flags.append(is_correct)
        if is_correct:
            domain_correct[domain] += 1

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    n_total: int = len(correct_flags)
    n_correct: int = sum(correct_flags)
    overall_accuracy: float = n_correct / n_total if n_total > 0 else float("nan")

    if n_total == 0:
        raise ValueError("solutions DataFrame produced zero evaluable rows.")

    ci_lower, ci_upper = _bootstrap_ci(correct_flags, n_boot=n_boot, seed=bootstrap_seed)

    domain_stats: list[DomainStats] = sorted(
        [
            DomainStats(
                domain=d,
                n_total=domain_total[d],
                n_correct=domain_correct.get(d, 0),
                accuracy=(
                    domain_correct.get(d, 0) / domain_total[d]
                    if domain_total[d] > 0
                    else float("nan")
                ),
            )
            for d in domain_total
        ],
        key=lambda s: s.domain,
    )

    result = CVResult(
        overall_accuracy=overall_accuracy,
        n_total=n_total,
        n_correct=n_correct,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_boot,
        domain_stats=domain_stats,
        missing_ids=missing_ids,
    )

    _log_report(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_report(result: CVResult) -> None:
    """Log a human-readable CV report at INFO level.

    Args:
        result: A populated ``CVResult`` instance.

    Returns:
        None.
    """
    sep = "-" * 52
    logger.info(sep)
    logger.info("CV RESULT SUMMARY")
    logger.info(sep)
    logger.info(
        "Overall accuracy : %.4f  (%d / %d correct)",
        result.overall_accuracy,
        result.n_correct,
        result.n_total,
    )
    logger.info(
        "Bootstrap 95%% CI : [%.4f, %.4f]  (n_boot=%d)",
        result.ci_lower,
        result.ci_upper,
        result.n_bootstrap,
    )
    if result.missing_ids:
        logger.warning(
            "%d IDs missing from predictions (counted wrong): %s",
            len(result.missing_ids),
            result.missing_ids[:10],
        )
    logger.info(sep)
    logger.info("PER-DOMAIN BREAKDOWN")
    logger.info("%-12s  %6s  %7s  %8s", "Domain", "N", "Correct", "Accuracy")
    logger.info("-" * 40)
    for ds in result.domain_stats:
        logger.info(
            "%-12s  %6d  %7d  %8.4f",
            ds.domain,
            ds.n_total,
            ds.n_correct,
            ds.accuracy,
        )
    logger.info(sep)
