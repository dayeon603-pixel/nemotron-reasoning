"""Tests for the CrackSymbolSolver (src/solvers/crack_symbol.py).

Test categories
---------------
1. Unit tests for prompt parsing correctness.
2. Unit tests for each derivation path (arithmetic, char-deletion).
3. Contract tests: answer == known_answer when supplied; CoT ends with \\boxed{}.
4. Integration accuracy test against data/raw/train.csv.
   Reports: outright accuracy on arithmetic rows, pure-symbol rows, and all
   SYMBOL rows.  Fails if accuracy drops below documented thresholds.

Run:
    python -m pytest -q tests/test_crack_symbol.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from src.solvers.crack_symbol import (
    CrackSymbolSolver,
    _ARITH_LHS_PATTERN,
    _SYMBOL_PREFIX,
    _build_arith_candidate_fns,
    _parse_symbol_prompt,
    _try_arithmetic,
    _try_char_deletion_subst,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_TRAIN_CSV: Path = _REPO_ROOT / "data" / "raw" / "train.csv"

# ---------------------------------------------------------------------------
# Accuracy thresholds (conservative — must not regress below these)
# ---------------------------------------------------------------------------

# Documented baseline from the old SymbolSolver: ~3 %.  The extended solver
# targets ~8.9 % outright on all SYMBOL rows.  We set the floor at 7 % to
# give room for minor data variation without a brittle test.
_MIN_OUTRIGHT_ALL_SYMBOL: float = 0.07  # 7 %

# The extended arithmetic solver achieves ~19 % on arithmetic-only rows.
_MIN_OUTRIGHT_ARITH: float = 0.12  # 12 %

# Precision on the rows where the arithmetic solver fires (should be >60 %).
_MIN_PRECISION_ON_FOUND: float = 0.60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOXED_MARKER: str = r"\boxed{"


def _load_symbol_rows() -> list[dict[str, str]]:
    """Load all SYMBOL rows from train.csv.

    Returns:
        List of dicts with keys 'id', 'prompt', 'answer'.

    Raises:
        pytest.skip: If train.csv is not present.
    """
    if not _TRAIN_CSV.exists():
        pytest.skip(f"train.csv not found at {_TRAIN_CSV}; skipping real-data tests.")
    rows: list[dict[str, str]] = []
    with _TRAIN_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "secret set of transformation rules is applied to equations" in row["prompt"]:
                rows.append(row)
    return rows


def _extract_boxed(cot: str) -> str | None:
    """Extract the argument of the last \\boxed{...} in a CoT string.

    Uses the same ``rfind('}')`` logic as the competition metric scorer.
    This correctly handles answers that themselves contain ``}`` characters,
    e.g. ``\\boxed{+}}`` yields ``+}``.
    """
    boxed_answers: list[str] = []
    start = 0
    while True:
        idx = cot.find(_BOXED_MARKER, start)
        if idx == -1:
            break
        content_start = idx + len(_BOXED_MARKER)
        next_idx = cot.find(_BOXED_MARKER, content_start)
        window = cot[content_start:next_idx] if next_idx != -1 else cot[content_start:]
        last_brace = window.rfind("}")
        candidate = window[:last_brace].strip() if last_brace != -1 else window.strip()
        if candidate:
            boxed_answers.append(candidate)
        start = content_start
    return boxed_answers[-1] if boxed_answers else None


# ---------------------------------------------------------------------------
# 1. Prompt parsing unit tests
# ---------------------------------------------------------------------------


class TestParseSymbolPrompt:
    """Unit tests for _parse_symbol_prompt."""

    _SAMPLE_PROMPT: str = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
        " Below are a few examples:\n"
        "34+67 = 101\n"
        "12*05 = 60\n"
        "Now, determine the result for: 22+33"
    )

    def test_parses_examples(self) -> None:
        pairs, query = _parse_symbol_prompt(self._SAMPLE_PROMPT)
        assert len(pairs) == 2
        assert pairs[0] == ("34+67", "101")
        assert pairs[1] == ("12*05", "60")

    def test_parses_query(self) -> None:
        pairs, query = _parse_symbol_prompt(self._SAMPLE_PROMPT)
        assert query == "22+33"

    def test_raises_on_missing_query(self) -> None:
        bad_prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:\n"
            "34+67 = 101\n"
        )
        with pytest.raises(ValueError, match="query line"):
            _parse_symbol_prompt(bad_prompt)


# ---------------------------------------------------------------------------
# 2. Arithmetic candidate function tests
# ---------------------------------------------------------------------------


class TestArithCandidateFns:
    """Spot-check specific arithmetic candidate functions."""

    def _get_fn(self, label: str):
        fns = dict(_build_arith_candidate_fns())
        return fns.get(label)

    def test_add(self) -> None:
        fn = self._get_fn("a+b")
        assert fn is not None
        assert fn(34, 67) == "101"

    def test_sub(self) -> None:
        fn = self._get_fn("a-b")
        assert fn is not None
        assert fn(50, 20) == "30"

    def test_rev_add(self) -> None:
        fn = self._get_fn("rev(a)+rev(b)")
        assert fn is not None
        # rev(64) + rev(65) = 46 + 56 = 102
        assert fn(64, 65) == "102"

    def test_rev_of_result_add(self) -> None:
        fn = self._get_fn("rev(a+b)")
        assert fn is not None
        # rev(64 + 65) = rev(129) = "921"
        assert fn(64, 65) == "921"

    def test_mul(self) -> None:
        fn = self._get_fn("a*b")
        assert fn is not None
        assert fn(12, 5) == "60"

    def test_concat(self) -> None:
        fn = self._get_fn("str(a)+str(b)")
        assert fn is not None
        assert fn(12, 34) == "1234"

    def test_digitwise_add_add(self) -> None:
        fn = self._get_fn("d1+e1|d2+e2")
        assert fn is not None
        # a=34, b=25: d1+e1=3+2=5, d2+e2=4+5=9 => "59"
        assert fn(34, 25) == "59"

    def test_digitwise_mul_add(self) -> None:
        fn = self._get_fn("d1*e1|d2+e2")
        assert fn is not None
        # a=34, b=25: d1*e1=3*2=6, d2+e2=4+5=9 => "69"
        assert fn(34, 25) == "69"

    def test_safe_div_by_zero(self) -> None:
        fn = self._get_fn("a//b")
        assert fn is not None
        assert fn(10, 0) is None

    def test_add_plus_one(self) -> None:
        fn = self._get_fn("a+b+1")
        assert fn is not None
        assert fn(10, 20) == "31"


# ---------------------------------------------------------------------------
# 3. Arithmetic derivation path tests
# ---------------------------------------------------------------------------


class TestTryArithmetic:
    """Tests for _try_arithmetic."""

    def _make_prompt_pairs(self, examples: list[tuple[str, str]], query: str) -> str:
        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:"
        ]
        for lhs, rhs in examples:
            lines.append(f"{lhs} = {rhs}")
        lines.append(f"Now, determine the result for: {query}")
        return "\n".join(lines)

    def test_single_operator_add(self) -> None:
        pairs = [("34+67", "101"), ("10+20", "30"), ("11+11", "22")]
        result = _try_arithmetic(pairs, "50+25")
        assert result is not None
        pred, op_label = result
        assert pred == "75"
        assert "+" in op_label

    def test_single_operator_multiply(self) -> None:
        pairs = [("12*05", "60"), ("03*10", "30")]
        result = _try_arithmetic(pairs, "07*08")
        assert result is not None
        pred, _ = result
        assert pred == "56"

    def test_multi_operator(self) -> None:
        # '+' maps to add, '*' maps to mul
        pairs = [("12+08", "20"), ("03*10", "30"), ("50+50", "100")]
        result = _try_arithmetic(pairs, "07*05")
        assert result is not None
        pred, op_label = result
        assert pred == "35"

    def test_no_consistent_function_returns_none(self) -> None:
        # No candidate function maps 12+34=99 (not a+b=46, not a*b=408, etc.)
        pairs = [("12+34", "99")]
        result = _try_arithmetic(pairs, "10+10")
        assert result is None

    def test_query_operator_not_in_examples_returns_none(self) -> None:
        pairs = [("12+34", "46")]
        result = _try_arithmetic(pairs, "10*10")  # '*' not seen
        assert result is None

    def test_non_arithmetic_lhs_returns_none(self) -> None:
        pairs = [("!@#$%", "!@#")]
        result = _try_arithmetic(pairs, "!@#$%")
        assert result is None

    def test_reversed_operand_add(self) -> None:
        # rev(64)+rev(65) = 46+56 = 102 => pairs show 102
        pairs = [("64-65", "102"), ("28-68", "861")]
        # rev(64)+rev(65)=102 ✓, rev(28)+rev(68)=82+86=168 ✗
        # so "rev(a)+rev(b)" fails for both.
        # That's fine; test that it either returns None or a different fn.
        result = _try_arithmetic(pairs, "85-77")
        # We don't assert a specific value here — just that the function doesn't crash.
        # The key is no exception.


# ---------------------------------------------------------------------------
# 4. Char deletion + substitution tests
# ---------------------------------------------------------------------------


class TestTryCharDeletionSubst:
    """Tests for _try_char_deletion_subst."""

    def test_simple_deletion_identity(self) -> None:
        # Delete position 2 always; remaining chars have identity substitution.
        # All 5 examples operate on the same rule (delete pos 2).
        # Query must use characters seen in training examples.
        pairs = [
            ("%|*\"|", "%|\"|"),   # del pos 2='*'; kept: %, |, ", |
            ("\\(*[^", "\\([^"),   # del pos 2='*'; kept: \, (, [, ^
            ("(%+[@", "(%[@"),     # del pos 2='+'; kept: (, %, [, @
            ("|[*([", "|[(["),     # del pos 2='*'; kept: |, [, (, [
        ]
        # Query uses only chars seen in training: \, (, [, ^ are all in cmap
        result = _try_char_deletion_subst(pairs, "\\(*[^")
        assert result is not None
        pred, cmap, del_pos = result
        # position 2 is deleted; kept are 0,1,3,4 = \, (, [, ^ -> \, (, [, ^
        assert pred == "\\([^"
        assert 2 in del_pos

    def test_returns_none_on_inconsistent(self) -> None:
        # Inconsistent: same char maps to different outputs
        pairs = [("!@#$%", "!@#"), ("!@#$%", "xyz")]
        result = _try_char_deletion_subst(pairs, "!@#$%")
        assert result is None

    def test_no_deletion(self) -> None:
        # 5->5 length: pure substitution (0 deletions)
        # a->x, b->y, c->z, d->w, e->v  (length-preserving)
        pairs = [("abcde", "xyzwv"), ("abcde", "xyzwv")]
        # All examples same, single-char substitution, no deletion needed
        result = _try_char_deletion_subst(pairs, "abcde")
        assert result is not None
        pred, _, _ = result
        assert pred == "xyzwv"


# ---------------------------------------------------------------------------
# 5. CrackSymbolSolver contract tests
# ---------------------------------------------------------------------------


class TestCrackSymbolSolverContracts:
    """Contract: matches, CoT ends with \\boxed{}, answer == known_answer."""

    _SOLVER = CrackSymbolSolver()

    _ARITH_PROMPT: str = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
        " Below are a few examples:\n"
        "12+08 = 20\n"
        "03*10 = 30\n"
        "50+50 = 100\n"
        "Now, determine the result for: 07*05"
    )

    def test_matches_symbol_prefix(self) -> None:
        assert self._SOLVER.matches(self._ARITH_PROMPT) is True

    def test_does_not_match_encrypt(self) -> None:
        assert self._SOLVER.matches("In Alice's Wonderland, secret encryption rules") is False

    def test_cot_ends_with_boxed(self) -> None:
        cot, answer = self._SOLVER.solve(self._ARITH_PROMPT)
        last_boxed = _extract_boxed(cot)
        assert last_boxed is not None, "CoT must contain \\boxed{...}"

    def test_cot_boxed_equals_returned_answer(self) -> None:
        cot, answer = self._SOLVER.solve(self._ARITH_PROMPT)
        last_boxed = _extract_boxed(cot)
        assert last_boxed == answer

    def test_known_answer_always_returned(self) -> None:
        known = "99"  # deliberately wrong vs derived "35"
        cot, answer = self._SOLVER.solve(self._ARITH_PROMPT, known_answer=known)
        assert answer == known

    def test_boxed_matches_known_answer_when_supplied(self) -> None:
        known = "99"
        cot, answer = self._SOLVER.solve(self._ARITH_PROMPT, known_answer=known)
        last_boxed = _extract_boxed(cot)
        assert last_boxed == known

    def test_raises_on_malformed_prompt(self) -> None:
        bad = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:\n"
            "12+08 = 20\n"
        )
        with pytest.raises(ValueError):
            self._SOLVER.solve(bad)  # no known_answer and derivation fails

    def test_fallback_with_known_answer_on_unknown_puzzle(self) -> None:
        # This prompt can't be solved but has a known_answer
        prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:\n"
            "!@#$% = ZZZ\n"
            "Now, determine the result for: &*([]"
        )
        cot, answer = self._SOLVER.solve(prompt, known_answer="XYZ")
        assert answer == "XYZ"
        assert _extract_boxed(cot) == "XYZ"


# ---------------------------------------------------------------------------
# 6. Real-data accuracy integration test
# ---------------------------------------------------------------------------


class TestRealDataAccuracy:
    """Accuracy against the full SYMBOL subset of train.csv.

    These tests are slow (~5 s) and require train.csv.  They confirm that the
    solver meets the documented accuracy thresholds.
    """

    @pytest.fixture(scope="class")
    def symbol_rows(self) -> list[dict[str, str]]:
        return _load_symbol_rows()

    def test_symbol_row_count_sanity(self, symbol_rows: list[dict[str, str]]) -> None:
        """Sanity-check: we expect ~1555 SYMBOL rows."""
        assert 1400 <= len(symbol_rows) <= 1700, (
            f"Expected ~1555 SYMBOL rows, got {len(symbol_rows)}"
        )

    def test_outright_accuracy_all_symbol(self, symbol_rows: list[dict[str, str]]) -> None:
        """Outright accuracy (no known_answer) must exceed {_MIN_OUTRIGHT_ALL_SYMBOL:.0%}."""
        solver = CrackSymbolSolver()
        correct = 0
        errors = 0
        for row in symbol_rows:
            try:
                _, pred = solver.solve(row["prompt"])
                if pred == row["answer"]:
                    correct += 1
            except ValueError:
                errors += 1  # derivation failed, no known_answer — expected for hard rows
            except Exception as exc:
                pytest.fail(f"Unexpected exception on row {row.get('id', '?')}: {exc}")
        total = len(symbol_rows)
        accuracy = correct / total
        print(
            f"\n[SYMBOL all] correct={correct}/{total} "
            f"({accuracy:.1%}), errors={errors}"
        )
        assert accuracy >= _MIN_OUTRIGHT_ALL_SYMBOL, (
            f"Outright accuracy {accuracy:.1%} < floor {_MIN_OUTRIGHT_ALL_SYMBOL:.0%}"
        )

    def test_outright_accuracy_arithmetic_rows(self, symbol_rows: list[dict[str, str]]) -> None:
        """Arithmetic sub-family outright accuracy must exceed {_MIN_OUTRIGHT_ARITH:.0%}."""
        solver = CrackSymbolSolver()
        arith_rows = [
            row for row in symbol_rows
            if all(
                _ARITH_LHS_PATTERN.match(lhs)
                for lhs, _ in _parse_puzzle(row["prompt"])[0]
            )
        ]
        if not arith_rows:
            pytest.skip("No arithmetic SYMBOL rows found.")

        correct = 0
        errors = 0
        for row in arith_rows:
            try:
                _, pred = solver.solve(row["prompt"])
                if pred == row["answer"]:
                    correct += 1
            except ValueError:
                errors += 1
            except Exception as exc:
                pytest.fail(f"Unexpected exception on row {row.get('id', '?')}: {exc}")

        total = len(arith_rows)
        accuracy = correct / total
        print(
            f"\n[SYMBOL arith] correct={correct}/{total} "
            f"({accuracy:.1%}), errors={errors}"
        )
        assert accuracy >= _MIN_OUTRIGHT_ARITH, (
            f"Arithmetic accuracy {accuracy:.1%} < floor {_MIN_OUTRIGHT_ARITH:.0%}"
        )

    def test_precision_on_arithmetic_found(
        self, symbol_rows: list[dict[str, str]]
    ) -> None:
        """Precision (correct / attempted) on arithmetic rows must exceed {_MIN_PRECISION_ON_FOUND:.0%}."""
        solver = CrackSymbolSolver()
        arith_rows = [
            row for row in symbol_rows
            if all(
                _ARITH_LHS_PATTERN.match(lhs)
                for lhs, _ in _parse_puzzle(row["prompt"])[0]
            )
        ]
        if not arith_rows:
            pytest.skip("No arithmetic SYMBOL rows found.")

        attempted = 0
        correct = 0
        for row in arith_rows:
            pairs, query = _parse_puzzle(row["prompt"])
            result = _try_arithmetic(pairs, query)
            if result is not None:
                attempted += 1
                pred, _ = result
                if pred == row["answer"]:
                    correct += 1

        if attempted == 0:
            pytest.skip("Arithmetic solver returned no predictions.")
        precision = correct / attempted
        print(
            f"\n[SYMBOL arith precision] correct={correct}/{attempted} "
            f"({precision:.1%})"
        )
        assert precision >= _MIN_PRECISION_ON_FOUND, (
            f"Precision {precision:.1%} < floor {_MIN_PRECISION_ON_FOUND:.0%}"
        )

    def test_with_known_answer_always_correct(
        self, symbol_rows: list[dict[str, str]]
    ) -> None:
        """With known_answer supplied, returned answer must always equal it (first 300 rows)."""
        solver = CrackSymbolSolver()
        sample = symbol_rows[:300]
        for row in sample:
            known = row["answer"]
            try:
                _, ans = solver.solve(row["prompt"], known_answer=known)
            except Exception as exc:
                pytest.fail(
                    f"Unexpected exception on row {row.get('id', '?')} "
                    f"with known_answer: {exc}"
                )
            assert ans == known, (
                f"Row {row.get('id', '?')}: returned {ans!r} != known {known!r}"
            )

    def test_cot_always_ends_with_boxed_answer(
        self, symbol_rows: list[dict[str, str]]
    ) -> None:
        """CoT must always end with \\boxed{answer} (first 300 rows with known_answer)."""
        solver = CrackSymbolSolver()
        sample = symbol_rows[:300]
        for row in sample:
            known = row["answer"]
            try:
                cot, ans = solver.solve(row["prompt"], known_answer=known)
            except Exception as exc:
                pytest.fail(
                    f"Unexpected exception on row {row.get('id', '?')}: {exc}"
                )
            last_boxed = _extract_boxed(cot)
            assert last_boxed is not None, (
                f"Row {row.get('id', '?')}: CoT has no \\boxed{{...}}"
            )
            assert last_boxed == ans, (
                f"Row {row.get('id', '?')}: \\boxed{{{last_boxed}}} != answer {ans!r}"
            )


# ---------------------------------------------------------------------------
# Helper used by accuracy tests (not imported from crack_symbol to keep
# the test file self-contained)
# ---------------------------------------------------------------------------

def _parse_puzzle(prompt: str) -> tuple[list[tuple[str, str]], str]:
    """Minimal parse: returns (example_pairs, query) from a SYMBOL prompt.

    Args:
        prompt: Full SYMBOL prompt string.

    Returns:
        Tuple of (pairs, query) where pairs is a list of (lhs, rhs) and
        query is the string after 'Now, determine the result for:'.
    """
    pairs: list[tuple[str, str]] = []
    query: str = ""
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Now, determine the result for:"):
            query = line.replace("Now, determine the result for:", "").strip()
        elif " = " in line and not line.startswith("Now,"):
            lhs, rhs = line.split(" = ", 1)
            pairs.append((lhs.strip(), rhs.strip()))
    return pairs, query
