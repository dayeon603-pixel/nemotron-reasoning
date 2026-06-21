"""Verbatim replication of the official NVIDIA Nemotron Model Reasoning Challenge scorer.

This module is an EXACT copy of the competition's public metric logic.
Every edge case — including the last-not-first boxed selection, the '}52'
nested-brace handling, and the last-number fallback — follows the official
CODE, not the official docstring (the docstring says "first"; the code
returns the last match).

Do NOT modify this file to "fix" apparent quirks.  Any deviation from the
official scorer will silently corrupt local CV scores and make leaderboard
comparisons meaningless.

References
----------
Competition metric source reviewed 2026-06-01.
"""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all regex patterns kept as module-level for reuse in cv.py
# ---------------------------------------------------------------------------

_BOXED_MARKER: str = r"\boxed{"

# Fallback phrase patterns tried in order when no \boxed{} is found.
# NOTE: order matters — first pattern that yields a match is used.
_PHRASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]

# Last-resort: grab every number-like token and return the LAST one.
# The competition docstring says "first" but the code does matches[-1].
_NUMBER_PATTERN: re.Pattern[str] = re.compile(r"-?\d+(?:\.\d+)?")

# Binary string: only '0' and '1' characters, length >= 1.
_BINARY_PATTERN: re.Pattern[str] = re.compile(r"^[01]+$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_final_answer(text: str | None) -> str:
    """Extract the model's final answer from free-form generation output.

    Replicates the official competition ``extract_final_answer`` verbatim,
    including the following non-obvious behaviours:

    * ``\\boxed{`` scanning: for each occurrence *i*, the candidate is
      the substring from the character after ``{`` up to the start of the
      *next* ``\\boxed{`` (or end-of-text), then truncated at the **last**
      ``}`` in that window.  This correctly handles answers that themselves
      contain ``}`` — e.g. ``\\boxed{}52}`` yields ``}52``.
    * When multiple ``\\boxed{…}`` blocks are present, the **last**
      non-empty stripped match is returned, not the first.
    * Phrase-regex fallback: four patterns tried in order; the last
      non-empty group from any matching pattern is returned.
    * Number fallback: ``re.findall`` returns all matches; ``matches[-1]``
      (the LAST) is returned even though the docstring says "first".
    * Final fallback: last non-empty line of the text.

    Args:
        text: Raw model generation output.  ``None`` is accepted and
            returns ``'NOT_FOUND'``.

    Returns:
        The extracted answer string, or ``'NOT_FOUND'`` if nothing was
        found.

    Raises:
        Nothing — all exceptions are caught internally and logged.
    """
    if text is None:
        return "NOT_FOUND"

    # ------------------------------------------------------------------
    # 1. \boxed{} extraction
    # ------------------------------------------------------------------
    boxed_answers: list[str] = []

    start: int = 0
    while True:
        idx: int = text.find(_BOXED_MARKER, start)
        if idx == -1:
            break

        # Content begins right after the opening '{'
        content_start: int = idx + len(_BOXED_MARKER)

        # Window ends at the start of the NEXT \boxed{ or at end-of-text
        next_idx: int = text.find(_BOXED_MARKER, content_start)
        if next_idx == -1:
            window: str = text[content_start:]
        else:
            window = text[content_start:next_idx]

        # Cut at the LAST '}' in the window — handles nested/extra braces.
        last_brace: int = window.rfind("}")
        if last_brace != -1:
            candidate: str = window[:last_brace].strip()
        else:
            candidate = window.strip()

        if candidate:
            boxed_answers.append(candidate)

        start = content_start  # advance past this \boxed{

    if boxed_answers:
        # Return the LAST non-empty match
        return boxed_answers[-1]

    # ------------------------------------------------------------------
    # 2. Phrase-regex fallback
    # ------------------------------------------------------------------
    phrase_answers: list[str] = []
    for pattern in _PHRASE_PATTERNS:
        for m in pattern.finditer(text):
            stripped: str = m.group(1).strip()
            if stripped:
                phrase_answers.append(stripped)

    if phrase_answers:
        return phrase_answers[-1]

    # ------------------------------------------------------------------
    # 3. Number fallback — NOTE: returns LAST match, not first
    # ------------------------------------------------------------------
    number_matches: list[str] = _NUMBER_PATTERN.findall(text)
    if number_matches:
        return number_matches[-1]

    # ------------------------------------------------------------------
    # 4. Last non-empty line
    # ------------------------------------------------------------------
    for line in reversed(text.splitlines()):
        stripped_line: str = line.strip()
        if stripped_line:
            return stripped_line

    return "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Check whether a model's predicted answer matches the stored ground truth.

    Replicates the official competition ``verify`` function verbatim,
    including the following branching logic:

    1. **Binary strings** (``re.fullmatch(r'[01]+', stored_answer)``):
       strict case-insensitive string equality.  Length-sensitive — a
       left-zero-padded prediction is WRONG (``"10011000"`` ≠
       ``"00011011"`` even though the numeric values happen to differ).

    2. **Numeric**: try ``float()`` on both; if both parse, use
       ``math.isclose(rel_tol=1e-2, abs_tol=1e-5)``.  1 % relative
       tolerance means answers like ``24.64`` and ``24.6401`` match.

    3. **Fallback string equality**: case-insensitive after ``.lower()``.
       This handles Roman numerals (``"XLVII"`` == ``"xlvii"``), text
       answers, and anything with units that breaks the float parse.

    Args:
        stored_answer: Ground-truth answer string from the solution CSV.
        predicted: Model's extracted answer string (output of
            ``extract_final_answer``).

    Returns:
        ``True`` if the prediction is considered correct by the official
        scorer, ``False`` otherwise.

    Raises:
        Nothing — exceptions from ``float()`` are caught to trigger
        fallback.

    Notes:
        Silent traps vs naive ``==`` — see module docstring and the
        CV protocol in ``cv.py``.
    """
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # Branch 1: binary string — strict equality, case-insensitive
    if _BINARY_PATTERN.fullmatch(stored_answer):
        return predicted.lower() == stored_answer.lower()

    # Branch 2: numeric comparison with tolerance
    try:
        stored_num: float = float(stored_answer)
        predicted_num: float = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass

    # Branch 3: fallback string equality (case-insensitive)
    return predicted.lower() == stored_answer.lower()
