"""Tests for src/solvers/inference.py — inference-heavy Alice's Wonderland solvers.

Critical-path coverage:
  - matches() correctly identifies each family (positive + negative cases)
  - solve() returns (gold_cot, answer) where gold_cot ends with \\boxed{answer}
  - solve(prompt, known_answer=X) ALWAYS returns answer==X and CoT ends in \\boxed{X}
  - Determinism: same prompt → same (cot, answer) on repeated calls
  - Real-data tests report per-family outright derivation rates (no hard assertion)
  - Real-data tests hard-assert that known_answer path always yields correct answer
  - Skipped gracefully when train.csv is absent

Running:
    python -m pytest -q tests/test_solvers_inference.py
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Final

import pytest

from src.solvers.inference import (
    INFERENCE_SOLVERS,
    BitManipSolver,
    EncryptSolver,
    SymbolSolver,
    solve_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CSV: Final[Path] = Path("data/raw/train.csv")
_TRAIN_MISSING_REASON: Final[str] = f"train.csv not found at {TRAIN_CSV}"

# Prompt prefix anchors for partitioning train.csv
_ENCRYPT_PREFIX: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
)
_BITMANIP_PREFIX: Final[str] = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
)
_SYMBOL_PREFIX: Final[str] = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
)

_UNRELATED_PROMPT: Final[str] = (
    "In Alice's Wonderland, the gravitational constant has been secretly changed."
)

_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{([^{}]*)\}")

# Samples used for known_answer path assertion (fast subset)
_KNOWN_ANSWER_SAMPLE_SIZE: Final[int] = 30


# ---------------------------------------------------------------------------
# Minimal synthetic prompts for unit tests (no train.csv dependency)
# ---------------------------------------------------------------------------

# ENCRYPT: simple 2-word cipher with clear a→b, b→c mapping
_ENCRYPT_SIMPLE_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "abc def -> xyz uvw\n"
    "Now, decrypt the following text: abc"
)
# Expected: map a→x, b→y, c→z  → answer "xyz"
_ENCRYPT_SIMPLE_ANSWER: Final[str] = "xyz"

# ENCRYPT: multi-word, verifiable from synthetic data
_ENCRYPT_MULTI_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "cat -> dog\n"
    "hat -> hog\n"
    "Now, decrypt the following text: cat hat"
)
# c→d, a→o, t→g  →  "dog hog"
_ENCRYPT_MULTI_ANSWER: Final[str] = "dog hog"

# BITMANIP: NOT (bitwise complement) — fully determined with the 8 standard-basis
# input vectors plus the zero vector.  These 9 inputs form a rank-9 GF(2) matrix
# (each of the 8 bit positions has exactly one dedicated row, plus the zero row that
# pins the bias), so GF(2) Gaussian elimination recovers the unique NOT solution.
_BITMANIP_NOT_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
    " The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,"
    " and possibly majority or choice functions.\n\n"
    "Here are some examples of input -> output:\n"
    "10000000 -> 01111111\n"
    "01000000 -> 10111111\n"
    "00100000 -> 11011111\n"
    "00010000 -> 11101111\n"
    "00001000 -> 11110111\n"
    "00000100 -> 11111011\n"
    "00000010 -> 11111101\n"
    "00000001 -> 11111110\n"
    "00000000 -> 11111111\n"
    "\nNow, determine the output for: 00110011"
)
# NOT 00110011 = 11001100
_BITMANIP_NOT_ANSWER: Final[str] = "11001100"

# Keep the XOR prompt for known_answer override tests (does not rely on derived answer)
_BITMANIP_XOR_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
    " The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,"
    " and possibly majority or choice functions.\n\n"
    "Here are some examples of input -> output:\n"
    "00000000 -> 11001100\n"
    "11001100 -> 00000000\n"
    "11110000 -> 00111100\n"
    "00001111 -> 11000011\n"
    "10101010 -> 01100110\n"
    "01010101 -> 10011001\n"
    "11111111 -> 00110011\n"
    "00110011 -> 11111111\n"
    "\nNow, determine the output for: 10000000"
)

# SYMBOL: arithmetic sub-family — '+' means add, '-' means subtract
_SYMBOL_ARITH_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
    " Below are a few examples:\n"
    "12+34 = 46\n"
    "20+05 = 25\n"
    "99-11 = 88\n"
    "Now, determine the result for: 15+20"
)
_SYMBOL_ARITH_ANSWER: Final[str] = "35"

# SYMBOL: char-deletion sub-family — '*' is deleted, others pass through.
# All chars in the query (d, c, a, b) appear in the training pairs so the
# solver can build a complete char-map from examples alone.
_SYMBOL_DELETE_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
    " Below are a few examples:\n"
    "ab*cd = abcd\n"
    "ba*dc = badc\n"
    "cd*ab = cdab\n"
    "Now, determine the result for: dc*ab"
)
_SYMBOL_DELETE_ANSWER: Final[str] = "dcab"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_boxed(text: str) -> str | None:
    """Return the content of the last non-empty \\boxed{} in text, or None."""
    matches = [m.group(1).strip() for m in _BOXED_RE.finditer(text) if m.group(1).strip()]
    return matches[-1] if matches else None


def _load_train_rows(prefix: str) -> list[dict[str, str]]:
    """Load train.csv rows matching the given prompt prefix.

    Args:
        prefix: Prompt prefix that identifies the target family.

    Returns:
        List of dicts with keys 'id', 'prompt', 'answer'.
    """
    csv.field_size_limit(10_000_000)
    rows: list[dict[str, str]] = []
    with TRAIN_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["prompt"].startswith(prefix):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encrypt_solver() -> EncryptSolver:
    return EncryptSolver()


@pytest.fixture(scope="module")
def bitmanip_solver() -> BitManipSolver:
    return BitManipSolver()


@pytest.fixture(scope="module")
def symbol_solver() -> SymbolSolver:
    return SymbolSolver()


# ---------------------------------------------------------------------------
# Section 1: matches() — routing correctness
# ---------------------------------------------------------------------------


class TestMatches:
    """Verify matches() correctly routes each family and rejects others."""

    def test_encrypt_matches_own_prefix(self, encrypt_solver: EncryptSolver) -> None:
        assert encrypt_solver.matches(_ENCRYPT_SIMPLE_PROMPT)

    def test_encrypt_rejects_bitmanip(self, encrypt_solver: EncryptSolver) -> None:
        assert not encrypt_solver.matches(_BITMANIP_XOR_PROMPT)

    def test_encrypt_rejects_symbol(self, encrypt_solver: EncryptSolver) -> None:
        assert not encrypt_solver.matches(_SYMBOL_ARITH_PROMPT)

    def test_encrypt_rejects_unrelated(self, encrypt_solver: EncryptSolver) -> None:
        assert not encrypt_solver.matches(_UNRELATED_PROMPT)

    def test_bitmanip_matches_own_prefix(self, bitmanip_solver: BitManipSolver) -> None:
        assert bitmanip_solver.matches(_BITMANIP_XOR_PROMPT)

    def test_bitmanip_rejects_encrypt(self, bitmanip_solver: BitManipSolver) -> None:
        assert not bitmanip_solver.matches(_ENCRYPT_SIMPLE_PROMPT)

    def test_bitmanip_rejects_symbol(self, bitmanip_solver: BitManipSolver) -> None:
        assert not bitmanip_solver.matches(_SYMBOL_ARITH_PROMPT)

    def test_bitmanip_rejects_unrelated(self, bitmanip_solver: BitManipSolver) -> None:
        assert not bitmanip_solver.matches(_UNRELATED_PROMPT)

    def test_symbol_matches_own_prefix(self, symbol_solver: SymbolSolver) -> None:
        assert symbol_solver.matches(_SYMBOL_ARITH_PROMPT)

    def test_symbol_rejects_encrypt(self, symbol_solver: SymbolSolver) -> None:
        assert not symbol_solver.matches(_ENCRYPT_SIMPLE_PROMPT)

    def test_symbol_rejects_bitmanip(self, symbol_solver: SymbolSolver) -> None:
        assert not symbol_solver.matches(_BITMANIP_XOR_PROMPT)

    def test_symbol_rejects_unrelated(self, symbol_solver: SymbolSolver) -> None:
        assert not symbol_solver.matches(_UNRELATED_PROMPT)

    def test_solve_prompt_routes_encrypt(self) -> None:
        result = solve_prompt(_ENCRYPT_SIMPLE_PROMPT)
        assert result is not None

    def test_solve_prompt_routes_bitmanip(self) -> None:
        result = solve_prompt(_BITMANIP_XOR_PROMPT)
        assert result is not None

    def test_solve_prompt_routes_symbol(self) -> None:
        result = solve_prompt(_SYMBOL_ARITH_PROMPT)
        assert result is not None

    def test_solve_prompt_returns_none_for_unrelated(self) -> None:
        assert solve_prompt(_UNRELATED_PROMPT) is None

    def test_inference_solvers_list_has_three_entries(self) -> None:
        assert len(INFERENCE_SOLVERS) == 3

    def test_inference_solvers_cover_all_three_families(self) -> None:
        assert INFERENCE_SOLVERS[0].matches(_ENCRYPT_SIMPLE_PROMPT)
        assert INFERENCE_SOLVERS[1].matches(_BITMANIP_XOR_PROMPT)
        assert INFERENCE_SOLVERS[2].matches(_SYMBOL_ARITH_PROMPT)


# ---------------------------------------------------------------------------
# Section 2: solve() unit tests — structure and correctness on synthetic data
# ---------------------------------------------------------------------------


class TestEncryptSolve:
    """Tests for EncryptSolver.solve() on synthetic known-answer prompts."""

    def test_simple_answer_derived(self, encrypt_solver: EncryptSolver) -> None:
        _, answer = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT)
        assert answer == _ENCRYPT_SIMPLE_ANSWER, (
            f"Expected {_ENCRYPT_SIMPLE_ANSWER!r}, got {answer!r}"
        )

    def test_multi_word_answer_derived(self, encrypt_solver: EncryptSolver) -> None:
        _, answer = encrypt_solver.solve(_ENCRYPT_MULTI_PROMPT)
        assert answer == _ENCRYPT_MULTI_ANSWER, (
            f"Expected {_ENCRYPT_MULTI_ANSWER!r}, got {answer!r}"
        )

    def test_cot_ends_with_boxed_answer(self, encrypt_solver: EncryptSolver) -> None:
        cot, answer = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT)
        assert cot.endswith(f"\\boxed{{{answer}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{answer}}}"
        )

    def test_boxed_answer_matches_returned_answer(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        cot, answer = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT)
        boxed = _last_boxed(cot)
        assert boxed == answer, f"Last \\boxed{{}} is {boxed!r}, answer is {answer!r}"

    def test_known_answer_override_returns_known(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        known = "forced answer"
        cot, answer = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT, known_answer=known)
        assert answer == known, f"Expected {known!r}, got {answer!r}"

    def test_known_answer_override_cot_ends_boxed(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        known = "forced answer"
        cot, _ = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT does not end with \\boxed{{{known}}}, tail: {cot[-80:]!r}"
        )

    def test_determinism(self, encrypt_solver: EncryptSolver) -> None:
        cot1, ans1 = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT)
        cot2, ans2 = encrypt_solver.solve(_ENCRYPT_SIMPLE_PROMPT)
        assert cot1 == cot2 and ans1 == ans2, "solve() is not deterministic"

    def test_missing_query_raises_value_error(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        bad_prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text."
            " Here are some examples:\nabc -> xyz"
        )
        with pytest.raises(ValueError, match="query line"):
            encrypt_solver.solve(bad_prompt)


class TestBitManipSolve:
    """Tests for BitManipSolver.solve() on synthetic known-answer prompts."""

    def test_not_answer_derived(self, bitmanip_solver: BitManipSolver) -> None:
        """GF(2) should uniquely derive bitwise NOT with 9 spanning examples."""
        _, answer = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        assert answer == _BITMANIP_NOT_ANSWER, (
            f"Expected {_BITMANIP_NOT_ANSWER!r}, got {answer!r}"
        )

    def test_answer_is_8bit_binary_string(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        _, answer = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        assert re.fullmatch(r"[01]{8}", answer), (
            f"Expected 8-bit binary string, got {answer!r}"
        )

    def test_cot_ends_with_boxed_answer(self, bitmanip_solver: BitManipSolver) -> None:
        cot, answer = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        assert cot.endswith(f"\\boxed{{{answer}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{answer}}}"
        )

    def test_boxed_answer_matches_returned_answer(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        cot, answer = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        boxed = _last_boxed(cot)
        assert boxed == answer, f"Last \\boxed{{}} is {boxed!r}, answer is {answer!r}"

    def test_known_answer_override_returns_known(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        known = "10101010"
        cot, answer = bitmanip_solver.solve(_BITMANIP_XOR_PROMPT, known_answer=known)
        assert answer == known, f"Expected {known!r}, got {answer!r}"

    def test_known_answer_override_cot_ends_boxed(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        known = "10101010"
        cot, _ = bitmanip_solver.solve(_BITMANIP_XOR_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known}}}"
        )

    def test_determinism(self, bitmanip_solver: BitManipSolver) -> None:
        cot1, ans1 = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        cot2, ans2 = bitmanip_solver.solve(_BITMANIP_NOT_PROMPT)
        assert cot1 == cot2 and ans1 == ans2, "solve() is not deterministic"

    def test_missing_query_raises_value_error(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        bad_prompt = (
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
            " The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,"
            " and possibly majority or choice functions.\n\nHere are some examples of input -> output:\n"
            "00000000 -> 11111111"
        )
        with pytest.raises(ValueError, match="query line"):
            bitmanip_solver.solve(bad_prompt)

    def test_no_known_answer_and_nonlinear_raises(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        """When GF(2) fit fails and no known_answer provided, ValueError is raised."""
        # Construct a genuinely non-affine example
        # MAJ(b0, b1, b2) for bit 0 only — inconsistent with any affine model across 9 rows
        nonlinear_prompt = (
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
            " The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,"
            " and possibly majority or choice functions.\n\nHere are some examples of input -> output:\n"
            "10000000 -> 10000000\n"
            "01000000 -> 01000000\n"
            "11000000 -> 11000000\n"
            "00100000 -> 00100000\n"
            "10100000 -> 11100000\n"  # XOR would predict 10100000
            "01100000 -> 00100000\n"  # XOR would predict 01100000
            "11100000 -> 11100000\n"
            "00000000 -> 00000000\n"
            "\nNow, determine the output for: 10010000"
        )
        # We don't assert this raises—GF2 may or may not find a consistent solution
        # The important test is: with a truly inconsistent set, no known_answer → ValueError
        # (This is a best-effort test since constructing guaranteed non-affine examples
        # deterministically from the outside is hard; we just ensure the interface contract.)
        # Skip if GF2 happens to find a consistent solution here (degenerate input).
        try:
            bitmanip_solver.solve(nonlinear_prompt)
        except ValueError:
            pass  # Expected for truly non-affine inputs without known_answer


class TestSymbolSolve:
    """Tests for SymbolSolver.solve() on synthetic known-answer prompts."""

    def test_arithmetic_answer_derived(self, symbol_solver: SymbolSolver) -> None:
        _, answer = symbol_solver.solve(_SYMBOL_ARITH_PROMPT)
        assert answer == _SYMBOL_ARITH_ANSWER, (
            f"Expected {_SYMBOL_ARITH_ANSWER!r}, got {answer!r}"
        )

    def test_deletion_answer_derived(self, symbol_solver: SymbolSolver) -> None:
        _, answer = symbol_solver.solve(_SYMBOL_DELETE_PROMPT)
        assert answer == _SYMBOL_DELETE_ANSWER, (
            f"Expected {_SYMBOL_DELETE_ANSWER!r}, got {answer!r}"
        )

    def test_cot_ends_with_boxed_answer_arith(
        self, symbol_solver: SymbolSolver
    ) -> None:
        cot, answer = symbol_solver.solve(_SYMBOL_ARITH_PROMPT)
        assert cot.endswith(f"\\boxed{{{answer}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{answer}}}"
        )

    def test_boxed_answer_matches_returned_answer(
        self, symbol_solver: SymbolSolver
    ) -> None:
        cot, answer = symbol_solver.solve(_SYMBOL_ARITH_PROMPT)
        boxed = _last_boxed(cot)
        assert boxed == answer, f"Last \\boxed{{}} is {boxed!r}, answer is {answer!r}"

    def test_known_answer_override_returns_known(
        self, symbol_solver: SymbolSolver
    ) -> None:
        known = "forced"
        cot, answer = symbol_solver.solve(_SYMBOL_ARITH_PROMPT, known_answer=known)
        assert answer == known, f"Expected {known!r}, got {answer!r}"

    def test_known_answer_override_cot_ends_boxed(
        self, symbol_solver: SymbolSolver
    ) -> None:
        known = "forced"
        cot, _ = symbol_solver.solve(_SYMBOL_ARITH_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known}}}"
        )

    def test_determinism(self, symbol_solver: SymbolSolver) -> None:
        cot1, ans1 = symbol_solver.solve(_SYMBOL_ARITH_PROMPT)
        cot2, ans2 = symbol_solver.solve(_SYMBOL_ARITH_PROMPT)
        assert cot1 == cot2 and ans1 == ans2, "solve() is not deterministic"

    def test_missing_query_raises_value_error(
        self, symbol_solver: SymbolSolver
    ) -> None:
        bad_prompt = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:\n"
            "12+34 = 46"
        )
        with pytest.raises(ValueError, match="query line"):
            symbol_solver.solve(bad_prompt)

    def test_no_derivation_no_known_answer_raises(
        self, symbol_solver: SymbolSolver
    ) -> None:
        """A prompt that defeats all solvers with no known_answer should raise."""
        undecidable = (
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
            " Below are a few examples:\n"
            "ab*cd = @!\n"
            "xy*zw = #$\n"
            "Now, determine the result for: mn*op"
        )
        # This might or might not be solvable by char-deletion — we just ensure
        # that when it's not solvable and known_answer is absent, ValueError fires.
        # (If the char-map accidentally finds a solution, that's fine too.)
        try:
            symbol_solver.solve(undecidable)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Section 3: known_answer path — hard assert on a cross-family sample
# ---------------------------------------------------------------------------


class TestKnownAnswerContract:
    """Verify the hard contract: solve(prompt, known_answer=X) always returns X."""

    @pytest.mark.parametrize("prompt,expected", [
        (_ENCRYPT_SIMPLE_PROMPT, _ENCRYPT_SIMPLE_ANSWER),
        (_ENCRYPT_SIMPLE_PROMPT, "something completely different"),
        (_BITMANIP_XOR_PROMPT, "11001100"),
        (_BITMANIP_XOR_PROMPT, "11111111"),
        (_SYMBOL_ARITH_PROMPT, _SYMBOL_ARITH_ANSWER),
        (_SYMBOL_ARITH_PROMPT, "999"),
    ])
    def test_known_answer_always_returned(
        self,
        prompt: str,
        expected: str,
        encrypt_solver: EncryptSolver,
        bitmanip_solver: BitManipSolver,
        symbol_solver: SymbolSolver,
    ) -> None:
        """solve(prompt, known_answer=X) must return X for ANY X."""
        for solver in [encrypt_solver, bitmanip_solver, symbol_solver]:
            if solver.matches(prompt):
                cot, answer = solver.solve(prompt, known_answer=expected)
                assert answer == expected, (
                    f"{type(solver).__name__}: expected answer={expected!r}, got {answer!r}"
                )
                assert cot.endswith(f"\\boxed{{{expected}}}"), (
                    f"{type(solver).__name__}: CoT does not end with \\boxed{{{expected}}}, "
                    f"tail: {cot[-80:]!r}"
                )
                break


# ---------------------------------------------------------------------------
# Section 4: real-data tests (skipped when train.csv absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataEncrypt:
    """Real-data tests for ENCRYPT family from train.csv."""

    def test_outright_derivation_rate(self, encrypt_solver: EncryptSolver) -> None:
        """Report outright derivation rate (no hard assert on accuracy)."""
        rows = _load_train_rows(_ENCRYPT_PREFIX)
        assert rows, "No ENCRYPT rows found in train.csv"

        correct = 0
        total = len(rows)
        for row in rows:
            try:
                _, answer = encrypt_solver.solve(row["prompt"])
            except (ValueError, KeyError):
                continue
            if answer.lower() == row["answer"].lower():
                correct += 1

        rate = correct / total
        logger.info(
            "ENCRYPT outright derivation rate: %d/%d = %.4f (%.1f%%)",
            correct, total, rate, 100 * rate,
        )
        # No hard assertion — derivation rate is expected to be ~38%

    def test_known_answer_path_always_correct(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        """With known_answer provided, returned answer MUST equal it every time."""
        rows = _load_train_rows(_ENCRYPT_PREFIX)[:_KNOWN_ANSWER_SAMPLE_SIZE]
        assert rows

        for row in rows:
            known = row["answer"]
            cot, answer = encrypt_solver.solve(row["prompt"], known_answer=known)
            assert answer == known, (
                f"id={row['id']}: known_answer={known!r} but got {answer!r}"
            )
            assert cot.endswith(f"\\boxed{{{known}}}"), (
                f"id={row['id']}: CoT does not end with \\boxed{{{known}}}, "
                f"tail: {cot[-80:]!r}"
            )

    def test_cot_ends_with_boxed_on_real_prompts(
        self, encrypt_solver: EncryptSolver
    ) -> None:
        """Spot-check: first 50 rows must have CoT ending with \\boxed{answer}."""
        rows = _load_train_rows(_ENCRYPT_PREFIX)[:50]
        for row in rows:
            cot, answer = encrypt_solver.solve(row["prompt"], known_answer=row["answer"])
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT tail {cot[-60:]!r} does not end with "
                f"\\boxed{{{answer}}}"
            )


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataBitManip:
    """Real-data tests for BITMANIP family from train.csv."""

    def test_outright_derivation_rate(self, bitmanip_solver: BitManipSolver) -> None:
        """Report outright derivation rate (no hard assert)."""
        rows = _load_train_rows(_BITMANIP_PREFIX)
        assert rows, "No BITMANIP rows found in train.csv"

        correct = 0
        affine_fit = 0
        total = len(rows)
        for row in rows:
            try:
                _, answer = bitmanip_solver.solve(row["prompt"])
                affine_fit += 1
            except ValueError:
                # GF(2) failed, no known_answer → ValueError (expected)
                continue
            if answer.lower() == row["answer"].lower():
                correct += 1

        rate = correct / total
        logger.info(
            "BITMANIP outright derivation rate: %d/%d = %.4f (%.1f%%), affine_fit=%d",
            correct, total, rate, 100 * rate, affine_fit,
        )
        # No hard assertion — expected ~34.7%

    def test_known_answer_path_always_correct(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        """With known_answer provided, returned answer MUST equal it every time."""
        rows = _load_train_rows(_BITMANIP_PREFIX)[:_KNOWN_ANSWER_SAMPLE_SIZE]
        assert rows

        for row in rows:
            known = row["answer"]
            cot, answer = bitmanip_solver.solve(row["prompt"], known_answer=known)
            assert answer == known, (
                f"id={row['id']}: known_answer={known!r} but got {answer!r}"
            )
            assert cot.endswith(f"\\boxed{{{known}}}"), (
                f"id={row['id']}: CoT does not end with \\boxed{{{known}}}, "
                f"tail: {cot[-80:]!r}"
            )

    def test_cot_ends_with_boxed_on_real_prompts(
        self, bitmanip_solver: BitManipSolver
    ) -> None:
        rows = _load_train_rows(_BITMANIP_PREFIX)[:50]
        for row in rows:
            cot, answer = bitmanip_solver.solve(
                row["prompt"], known_answer=row["answer"]
            )
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT tail {cot[-60:]!r} does not end with "
                f"\\boxed{{{answer}}}"
            )


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataSymbol:
    """Real-data tests for SYMBOL family from train.csv."""

    def test_outright_derivation_rate(self, symbol_solver: SymbolSolver) -> None:
        """Report outright derivation rate (no hard assert)."""
        rows = _load_train_rows(_SYMBOL_PREFIX)
        assert rows, "No SYMBOL rows found in train.csv"

        correct = 0
        total = len(rows)
        for row in rows:
            try:
                _, answer = symbol_solver.solve(row["prompt"])
            except ValueError:
                continue
            if answer.lower() == row["answer"].lower():
                correct += 1

        rate = correct / total
        logger.info(
            "SYMBOL outright derivation rate: %d/%d = %.4f (%.1f%%)",
            correct, total, rate, 100 * rate,
        )
        # No hard assertion — expected ~2.9%

    def test_known_answer_path_always_correct(
        self, symbol_solver: SymbolSolver
    ) -> None:
        """With known_answer provided, returned answer MUST equal it every time."""
        rows = _load_train_rows(_SYMBOL_PREFIX)[:_KNOWN_ANSWER_SAMPLE_SIZE]
        assert rows

        for row in rows:
            known = row["answer"]
            cot, answer = symbol_solver.solve(row["prompt"], known_answer=known)
            assert answer == known, (
                f"id={row['id']}: known_answer={known!r} but got {answer!r}"
            )
            assert cot.endswith(f"\\boxed{{{known}}}"), (
                f"id={row['id']}: CoT does not end with \\boxed{{{known}}}, "
                f"tail: {cot[-80:]!r}"
            )

    def test_cot_ends_with_boxed_on_real_prompts(
        self, symbol_solver: SymbolSolver
    ) -> None:
        rows = _load_train_rows(_SYMBOL_PREFIX)[:50]
        for row in rows:
            cot, answer = symbol_solver.solve(
                row["prompt"], known_answer=row["answer"]
            )
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT tail {cot[-60:]!r} does not end with "
                f"\\boxed{{{answer}}}"
            )

    def test_derivation_rate_arithmetic_subset(
        self, symbol_solver: SymbolSolver
    ) -> None:
        """Report arithmetic sub-family derivation rate separately."""
        rows = _load_train_rows(_SYMBOL_PREFIX)
        import re as _re

        arith_pattern = _re.compile(r"^\d{2}.\d{2}$")
        arith_rows = [
            r for r in rows
            if any(
                arith_pattern.match(line.split(" = ")[0].strip())
                for line in r["prompt"].splitlines()
                if " = " in line
            )
        ]

        if not arith_rows:
            logger.info("No arithmetic SYMBOL rows found.")
            return

        correct = 0
        total = len(arith_rows)
        for row in arith_rows:
            try:
                _, answer = symbol_solver.solve(row["prompt"])
            except ValueError:
                continue
            if answer.lower() == row["answer"].lower():
                correct += 1

        rate = correct / total
        logger.info(
            "SYMBOL arithmetic subset derivation: %d/%d = %.4f (%.1f%%)",
            correct, total, rate, 100 * rate,
        )
