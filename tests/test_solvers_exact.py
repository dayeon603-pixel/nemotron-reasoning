"""Tests for src/solvers/exact.py — closed-form Alice's Wonderland solvers.

Critical-path coverage:
  - matches() correctly identifies each family (positive and negative cases)
  - solve() returns (gold_cot, answer) where gold_cot ends with \\boxed{answer}
  - Real-data accuracy: 100% verify() on every row of each family in train.csv
  - generate() is deterministic (same seed → same output)
  - generate() is seed-sensitive (different seeds → different outputs)
  - Every generated Example's gold_cot ends with \\boxed{answer}
  - Every generated answer round-trips through src.eval.metric.verify
  - solve_prompt() routes correctly and returns None for unknown families
  - Parsing edge cases: 3-obs and 5-obs gravitational/unit prompts

Real-data tests are skipped (with a clear message) when train.csv is absent.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Final

import pytest

from src.eval.metric import extract_final_answer, verify
from src.generators.common import Example
from src.solvers.exact import (
    EXACT_SOLVERS,
    GravitationalSolver,
    NumeralSolver,
    UnitConversionSolver,
    _int_to_roman,
    solve_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CSV: Final[Path] = Path("data/raw/train.csv")
SEED_A: Final[int] = 42
SEED_B: Final[int] = 99
N_SYNTHETIC: Final[int] = 20

_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{([^{}]*)\}")

# Real-prompt prefixes used to partition train.csv
_GRAV_PREFIX: Final[str] = (
    "In Alice's Wonderland, the gravitational constant has been secretly changed"
)
_UNIT_PREFIX: Final[str] = (
    "In Alice's Wonderland, a secret unit conversion is applied to measurements"
)
_NUMERAL_PREFIX: Final[str] = (
    "In Alice's Wonderland, numbers are secretly converted into a different numeral system"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_boxed(text: str) -> str | None:
    """Return the last non-empty \\boxed{} content, or None."""
    matches = [m.group(1).strip() for m in _BOXED_RE.finditer(text) if m.group(1).strip()]
    return matches[-1] if matches else None


def _load_train_rows(prefix: str) -> list[dict[str, str]]:
    """Load all train.csv rows matching the given prompt prefix.

    Args:
        prefix: The prompt prefix string that identifies the target family.

    Returns:
        List of dicts with keys 'id', 'prompt', 'answer'.

    Raises:
        FileNotFoundError: Not raised — callers must check TRAIN_CSV.exists() first.
    """
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
def grav_solver() -> GravitationalSolver:
    return GravitationalSolver()


@pytest.fixture(scope="module")
def unit_solver() -> UnitConversionSolver:
    return UnitConversionSolver()


@pytest.fixture(scope="module")
def numeral_solver() -> NumeralSolver:
    return NumeralSolver()


# ---------------------------------------------------------------------------
# Section 1: matches() — routing correctness
# ---------------------------------------------------------------------------

class TestMatches:
    """Verify that matches() correctly identifies each family."""

    _GRAV_PROMPT: str = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 2.0s, distance = 20.0 m\n"
        "Now, determine the falling distance for t = 3.0s given d = 0.5*g*t^2."
    )
    _UNIT_PROMPT: str = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:\n"
        "10.00 m becomes 12.50\n"
        "Now, convert the following measurement: 8.00 m"
    )
    _NUMERAL_PROMPT: str = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:\n"
        "5 -> V\n"
        "Now, write the number 10 in the Wonderland numeral system."
    )
    _UNRELATED_PROMPT: str = "In Alice's Wonderland, a secret cipher is used on text."

    def test_grav_matches_own_family(self, grav_solver: GravitationalSolver) -> None:
        assert grav_solver.matches(self._GRAV_PROMPT)

    def test_grav_rejects_unit_prompt(self, grav_solver: GravitationalSolver) -> None:
        assert not grav_solver.matches(self._UNIT_PROMPT)

    def test_grav_rejects_numeral_prompt(self, grav_solver: GravitationalSolver) -> None:
        assert not grav_solver.matches(self._NUMERAL_PROMPT)

    def test_unit_matches_own_family(self, unit_solver: UnitConversionSolver) -> None:
        assert unit_solver.matches(self._UNIT_PROMPT)

    def test_unit_rejects_grav_prompt(self, unit_solver: UnitConversionSolver) -> None:
        assert not unit_solver.matches(self._GRAV_PROMPT)

    def test_numeral_matches_own_family(self, numeral_solver: NumeralSolver) -> None:
        assert numeral_solver.matches(self._NUMERAL_PROMPT)

    def test_numeral_rejects_grav_prompt(self, numeral_solver: NumeralSolver) -> None:
        assert not numeral_solver.matches(self._GRAV_PROMPT)

    def test_solve_prompt_routes_grav(self) -> None:
        result = solve_prompt(self._GRAV_PROMPT)
        assert result is not None
        _, answer = result
        # answer should be a float string
        assert re.fullmatch(r"-?\d+\.\d+", answer), (
            f"Expected float string, got {answer!r}"
        )

    def test_solve_prompt_routes_unit(self) -> None:
        result = solve_prompt(self._UNIT_PROMPT)
        assert result is not None
        _, answer = result
        assert re.fullmatch(r"-?\d+\.\d+", answer), (
            f"Expected float string, got {answer!r}"
        )

    def test_solve_prompt_routes_numeral(self) -> None:
        result = solve_prompt(self._NUMERAL_PROMPT)
        assert result is not None
        _, answer = result
        assert re.fullmatch(r"[IVXLCDM]+", answer), (
            f"Expected Roman numeral, got {answer!r}"
        )

    def test_solve_prompt_returns_none_for_unknown(self) -> None:
        assert solve_prompt(self._UNRELATED_PROMPT) is None

    def test_exact_solvers_list_has_three_entries(self) -> None:
        assert len(EXACT_SOLVERS) == 3

    def test_exact_solvers_cover_all_three_prefixes(self) -> None:
        assert EXACT_SOLVERS[0].matches(self._GRAV_PROMPT)
        assert EXACT_SOLVERS[1].matches(self._UNIT_PROMPT)
        assert EXACT_SOLVERS[2].matches(self._NUMERAL_PROMPT)


# ---------------------------------------------------------------------------
# Section 2: solve() unit tests — known-answer fixtures
# ---------------------------------------------------------------------------

class TestGravitationalSolve:
    """Known-answer tests for GravitationalSolver.solve()."""

    # g = 15.9, t_query = 4.41 → d = 0.5 * 15.9 * 4.41^2 ≈ 154.62
    _PROMPT_5OBS: str = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 1.37s, distance = 14.92 m\n"
        "For t = 4.27s, distance = 144.96 m\n"
        "For t = 3.28s, distance = 85.54 m\n"
        "For t = 3.67s, distance = 107.09 m\n"
        "For t = 1.78s, distance = 25.19 m\n"
        "Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2."
    )
    _EXPECTED_5OBS: str = "154.62"

    # g ≈ 12.587, t_query = 3.82
    _PROMPT_3OBS: str = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 4.74s, distance = 141.41 m\n"
        "For t = 3.71s, distance = 86.63 m\n"
        "For t = 1.75s, distance = 19.27 m\n"
        "Now, determine the falling distance for t = 3.82s given d = 0.5*g*t^2."
    )
    _EXPECTED_3OBS: str = "91.84"

    def test_5obs_answer(self, grav_solver: GravitationalSolver) -> None:
        _, answer = grav_solver.solve(self._PROMPT_5OBS)
        assert verify(self._EXPECTED_5OBS, answer), (
            f"Expected {self._EXPECTED_5OBS!r}, got {answer!r}"
        )

    def test_3obs_answer(self, grav_solver: GravitationalSolver) -> None:
        _, answer = grav_solver.solve(self._PROMPT_3OBS)
        assert verify(self._EXPECTED_3OBS, answer), (
            f"Expected {self._EXPECTED_3OBS!r}, got {answer!r}"
        )

    def test_cot_ends_with_boxed_answer(self, grav_solver: GravitationalSolver) -> None:
        cot, answer = grav_solver.solve(self._PROMPT_5OBS)
        assert cot.endswith(f"\\boxed{{{answer}}}"), (
            f"CoT tail {cot[-60:]!r} does not end with \\boxed{{{answer}}}"
        )

    def test_extracted_answer_verifies(self, grav_solver: GravitationalSolver) -> None:
        cot, answer = grav_solver.solve(self._PROMPT_5OBS)
        extracted = extract_final_answer(cot)
        assert verify(answer, extracted), (
            f"verify failed: gold={answer!r} extracted={extracted!r}"
        )

    def test_answer_is_float_string_2dp(self, grav_solver: GravitationalSolver) -> None:
        _, answer = grav_solver.solve(self._PROMPT_5OBS)
        assert re.fullmatch(r"\d+\.\d{2}", answer), (
            f"Expected 2-decimal float string, got {answer!r}"
        )

    def test_parse_error_on_missing_observations(
        self, grav_solver: GravitationalSolver
    ) -> None:
        with pytest.raises(ValueError, match="no observation pairs"):
            grav_solver.solve(
                "In Alice's Wonderland, the gravitational constant has been secretly changed. "
                "Now, determine the falling distance for t = 3.0s given d = 0.5*g*t^2."
            )

    def test_parse_error_on_missing_query(self, grav_solver: GravitationalSolver) -> None:
        with pytest.raises(ValueError, match="query time not found"):
            grav_solver.solve(
                "In Alice's Wonderland, the gravitational constant has been secretly changed. "
                "Here are some example observations:\n"
                "For t = 2.0s, distance = 20.0 m"
            )


class TestUnitConversionSolve:
    """Known-answer tests for UnitConversionSolver.solve()."""

    # k ≈ 0.6636, x_query = 25.09 → 16.65
    _PROMPT_5PAIR: str = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:\n"
        "10.08 m becomes 6.69\n"
        "17.83 m becomes 11.83\n"
        "35.85 m becomes 23.79\n"
        "17.06 m becomes 11.32\n"
        "31.54 m becomes 20.93\n"
        "Now, convert the following measurement: 25.09 m"
    )
    _EXPECTED_5PAIR: str = "16.65"

    # 3 pairs
    _PROMPT_3PAIR: str = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:\n"
        "10.00 m becomes 8.17\n"
        "20.00 m becomes 16.34\n"
        "30.00 m becomes 24.50\n"
        "Now, convert the following measurement: 13.00 m"
    )

    def test_5pair_answer(self, unit_solver: UnitConversionSolver) -> None:
        _, answer = unit_solver.solve(self._PROMPT_5PAIR)
        assert verify(self._EXPECTED_5PAIR, answer), (
            f"Expected {self._EXPECTED_5PAIR!r}, got {answer!r}"
        )

    def test_3pair_answer_verifies(self, unit_solver: UnitConversionSolver) -> None:
        _, answer = unit_solver.solve(self._PROMPT_3PAIR)
        # Answer should be parseable float
        float(answer)

    def test_cot_ends_with_boxed_answer(self, unit_solver: UnitConversionSolver) -> None:
        cot, answer = unit_solver.solve(self._PROMPT_5PAIR)
        assert cot.endswith(f"\\boxed{{{answer}}}")

    def test_extracted_answer_verifies(self, unit_solver: UnitConversionSolver) -> None:
        cot, answer = unit_solver.solve(self._PROMPT_5PAIR)
        extracted = extract_final_answer(cot)
        assert verify(answer, extracted)

    def test_answer_is_float_string(self, unit_solver: UnitConversionSolver) -> None:
        _, answer = unit_solver.solve(self._PROMPT_5PAIR)
        assert re.fullmatch(r"-?\d+\.\d+", answer), (
            f"Expected float string, got {answer!r}"
        )

    def test_parse_error_on_missing_pairs(
        self, unit_solver: UnitConversionSolver
    ) -> None:
        with pytest.raises(ValueError, match="no conversion pairs"):
            unit_solver.solve(
                "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
                "Now, convert the following measurement: 10.00 m"
            )

    def test_parse_error_on_missing_query(
        self, unit_solver: UnitConversionSolver
    ) -> None:
        with pytest.raises(ValueError, match="query measurement not found"):
            unit_solver.solve(
                "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
                "For example:\n"
                "10.00 m becomes 8.00"
            )


class TestNumeralSolve:
    """Known-answer tests for NumeralSolver.solve()."""

    _PROMPT_4EX: str = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:\n"
        "11 -> XI\n"
        "15 -> XV\n"
        "94 -> XCIV\n"
        "19 -> XIX\n"
        "Now, write the number 38 in the Wonderland numeral system."
    )
    _EXPECTED_4EX: str = "XXXVIII"

    _PROMPT_3EX: str = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:\n"
        "4 -> IV\n"
        "42 -> XLII\n"
        "59 -> LIX\n"
        "Now, write the number 100 in the Wonderland numeral system."
    )
    _EXPECTED_3EX: str = "C"

    def test_4ex_answer(self, numeral_solver: NumeralSolver) -> None:
        _, answer = numeral_solver.solve(self._PROMPT_4EX)
        assert verify(self._EXPECTED_4EX, answer), (
            f"Expected {self._EXPECTED_4EX!r}, got {answer!r}"
        )

    def test_3ex_answer(self, numeral_solver: NumeralSolver) -> None:
        _, answer = numeral_solver.solve(self._PROMPT_3EX)
        assert verify(self._EXPECTED_3EX, answer), (
            f"Expected {self._EXPECTED_3EX!r}, got {answer!r}"
        )

    def test_cot_ends_with_boxed_answer(self, numeral_solver: NumeralSolver) -> None:
        cot, answer = numeral_solver.solve(self._PROMPT_4EX)
        assert cot.endswith(f"\\boxed{{{answer}}}")

    def test_extracted_answer_verifies(self, numeral_solver: NumeralSolver) -> None:
        cot, answer = numeral_solver.solve(self._PROMPT_4EX)
        extracted = extract_final_answer(cot)
        assert verify(answer, extracted), (
            f"verify failed: gold={answer!r} extracted={extracted!r}"
        )

    def test_answer_is_uppercase_roman(self, numeral_solver: NumeralSolver) -> None:
        _, answer = numeral_solver.solve(self._PROMPT_4EX)
        assert re.fullmatch(r"[IVXLCDM]+", answer), (
            f"Expected uppercase Roman numeral, got {answer!r}"
        )

    def test_cot_contains_all_examples_verified(self, numeral_solver: NumeralSolver) -> None:
        cot, _ = numeral_solver.solve(self._PROMPT_4EX)
        assert "All examples verified." in cot

    def test_parse_error_on_missing_query(self, numeral_solver: NumeralSolver) -> None:
        with pytest.raises(ValueError, match="query integer not found"):
            numeral_solver.solve(
                "In Alice's Wonderland, numbers are secretly converted into a different "
                "numeral system. Some examples are given below:\n"
                "5 -> V"
            )

    # ── Roman codec corner cases ──────────────────────────────────────────────

    @pytest.mark.parametrize("n,expected", [
        (1, "I"), (4, "IV"), (9, "IX"), (14, "XIV"), (40, "XL"),
        (90, "XC"), (399, "CCCXCIX"), (1994, "MCMXCIV"), (3999, "MMMCMXCIX"),
    ])
    def test_int_to_roman_known_values(self, n: int, expected: str) -> None:
        assert _int_to_roman(n) == expected, (
            f"_int_to_roman({n}) = {_int_to_roman(n)!r}, expected {expected!r}"
        )

    def test_int_to_roman_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            _int_to_roman(0)

    def test_int_to_roman_rejects_4000(self) -> None:
        with pytest.raises(ValueError):
            _int_to_roman(4000)


# ---------------------------------------------------------------------------
# Section 3: generate() — synthetic data correctness
# ---------------------------------------------------------------------------

class TestGravitationalGenerate:
    """Tests for GravitationalSolver.generate()."""

    def test_returns_n_examples(self, grav_solver: GravitationalSolver) -> None:
        assert len(grav_solver.generate(N_SYNTHETIC, SEED_A)) == N_SYNTHETIC

    def test_all_are_example_instances(self, grav_solver: GravitationalSolver) -> None:
        for ex in grav_solver.generate(N_SYNTHETIC, SEED_A):
            assert isinstance(ex, Example)

    def test_gold_cot_ends_with_boxed_answer(self, grav_solver: GravitationalSolver) -> None:
        for ex in grav_solver.generate(N_SYNTHETIC, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} does not end with {expected_tail!r}"
            )

    def test_answer_verifies_against_itself(self, grav_solver: GravitationalSolver) -> None:
        for ex in grav_solver.generate(N_SYNTHETIC, SEED_A):
            assert verify(ex.answer, ex.answer), (
                f"verify(gold, gold) failed for {ex.answer!r}"
            )

    def test_extracted_answer_verifies(self, grav_solver: GravitationalSolver) -> None:
        for ex in grav_solver.generate(N_SYNTHETIC, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self, grav_solver: GravitationalSolver) -> None:
        a = grav_solver.generate(N_SYNTHETIC, SEED_A)
        b = grav_solver.generate(N_SYNTHETIC, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_examples(
        self, grav_solver: GravitationalSolver
    ) -> None:
        a = grav_solver.generate(N_SYNTHETIC, SEED_A)
        b = grav_solver.generate(N_SYNTHETIC, SEED_B)
        assert {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_wonderland_prompt_contains_grav_prefix(
        self, grav_solver: GravitationalSolver
    ) -> None:
        for ex in grav_solver.generate(5, SEED_A):
            assert "gravitational constant" in ex.prompt

    def test_answer_is_2dp_float_string(self, grav_solver: GravitationalSolver) -> None:
        for ex in grav_solver.generate(N_SYNTHETIC, SEED_A):
            assert re.fullmatch(r"\d+\.\d{2}", ex.answer), (
                f"Expected 2-decimal float string, got {ex.answer!r}"
            )

    def test_prompt_matches_answer_via_solver(
        self, grav_solver: GravitationalSolver
    ) -> None:
        """solve() on a generated prompt must reproduce the generated answer."""
        for ex in grav_solver.generate(5, SEED_A):
            _, predicted = grav_solver.solve(ex.prompt)
            assert verify(ex.answer, predicted), (
                f"Round-trip failed: generated={ex.answer!r} re-solved={predicted!r}"
            )


class TestUnitConversionGenerate:
    """Tests for UnitConversionSolver.generate()."""

    def test_returns_n_examples(self, unit_solver: UnitConversionSolver) -> None:
        assert len(unit_solver.generate(N_SYNTHETIC, SEED_A)) == N_SYNTHETIC

    def test_all_are_example_instances(self, unit_solver: UnitConversionSolver) -> None:
        for ex in unit_solver.generate(N_SYNTHETIC, SEED_A):
            assert isinstance(ex, Example)

    def test_gold_cot_ends_with_boxed_answer(self, unit_solver: UnitConversionSolver) -> None:
        for ex in unit_solver.generate(N_SYNTHETIC, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} does not end with {expected_tail!r}"
            )

    def test_answer_verifies_against_itself(self, unit_solver: UnitConversionSolver) -> None:
        for ex in unit_solver.generate(N_SYNTHETIC, SEED_A):
            assert verify(ex.answer, ex.answer)

    def test_extracted_answer_verifies(self, unit_solver: UnitConversionSolver) -> None:
        for ex in unit_solver.generate(N_SYNTHETIC, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self, unit_solver: UnitConversionSolver) -> None:
        a = unit_solver.generate(N_SYNTHETIC, SEED_A)
        b = unit_solver.generate(N_SYNTHETIC, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_examples(
        self, unit_solver: UnitConversionSolver
    ) -> None:
        a = unit_solver.generate(N_SYNTHETIC, SEED_A)
        b = unit_solver.generate(N_SYNTHETIC, SEED_B)
        assert {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_prompt_contains_unit_prefix(self, unit_solver: UnitConversionSolver) -> None:
        for ex in unit_solver.generate(5, SEED_A):
            assert "unit conversion" in ex.prompt

    def test_prompt_matches_answer_via_solver(
        self, unit_solver: UnitConversionSolver
    ) -> None:
        """solve() on a generated prompt must reproduce the generated answer."""
        for ex in unit_solver.generate(5, SEED_A):
            _, predicted = unit_solver.solve(ex.prompt)
            assert verify(ex.answer, predicted), (
                f"Round-trip failed: generated={ex.answer!r} re-solved={predicted!r}"
            )


class TestNumeralGenerate:
    """Tests for NumeralSolver.generate()."""

    def test_returns_n_examples(self, numeral_solver: NumeralSolver) -> None:
        assert len(numeral_solver.generate(N_SYNTHETIC, SEED_A)) == N_SYNTHETIC

    def test_all_are_example_instances(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(N_SYNTHETIC, SEED_A):
            assert isinstance(ex, Example)

    def test_gold_cot_ends_with_boxed_answer(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(N_SYNTHETIC, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} does not end with {expected_tail!r}"
            )

    def test_answer_verifies_against_itself(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(N_SYNTHETIC, SEED_A):
            assert verify(ex.answer, ex.answer)

    def test_extracted_answer_verifies(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(N_SYNTHETIC, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self, numeral_solver: NumeralSolver) -> None:
        a = numeral_solver.generate(N_SYNTHETIC, SEED_A)
        b = numeral_solver.generate(N_SYNTHETIC, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_examples(
        self, numeral_solver: NumeralSolver
    ) -> None:
        a = numeral_solver.generate(N_SYNTHETIC, SEED_A)
        b = numeral_solver.generate(N_SYNTHETIC, SEED_B)
        assert {ex.answer for ex in a} != {ex.answer for ex in b} or \
               {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_answer_is_uppercase_roman(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(N_SYNTHETIC, SEED_A):
            assert re.fullmatch(r"[IVXLCDM]+", ex.answer), (
                f"Expected uppercase Roman numeral, got {ex.answer!r}"
            )

    def test_prompt_contains_numeral_prefix(self, numeral_solver: NumeralSolver) -> None:
        for ex in numeral_solver.generate(5, SEED_A):
            assert "numeral system" in ex.prompt

    def test_prompt_matches_answer_via_solver(
        self, numeral_solver: NumeralSolver
    ) -> None:
        """solve() on a generated prompt must reproduce the generated answer."""
        for ex in numeral_solver.generate(5, SEED_A):
            _, predicted = numeral_solver.solve(ex.prompt)
            assert verify(ex.answer, predicted), (
                f"Round-trip failed: generated={ex.answer!r} re-solved={predicted!r}"
            )


# ---------------------------------------------------------------------------
# Section 4: real-data accuracy (skipped when train.csv is absent)
# ---------------------------------------------------------------------------

_TRAIN_MISSING_REASON: str = f"train.csv not found at {TRAIN_CSV}"


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataGravitational:
    """Verify 100% verify() accuracy on every gravitational row in train.csv."""

    def test_full_accuracy(self, grav_solver: GravitationalSolver) -> None:
        rows = _load_train_rows(_GRAV_PREFIX)
        assert rows, "No gravitational rows found in train.csv"

        correct = 0
        total = len(rows)
        failures: list[tuple[str, str, str]] = []

        for row in rows:
            _, answer = grav_solver.solve(row["prompt"])
            if verify(row["answer"], answer):
                correct += 1
            else:
                failures.append((row["id"], row["answer"], answer))

        accuracy = correct / total
        logger.info(
            "GRAVITATIONAL real-data accuracy: %d/%d = %.4f",
            correct, total, accuracy,
        )

        if failures:
            sample = failures[:5]
            msg = f"Failures (showing first 5 of {len(failures)}): {sample}"
            logger.warning(msg)

        assert accuracy == 1.0, (
            f"GRAVITATIONAL: {correct}/{total} correct (expected 1.0). "
            f"First failure: id={failures[0][0]} gold={failures[0][1]!r} "
            f"pred={failures[0][2]!r}"
        )

    def test_cot_ends_with_boxed_on_real_prompts(
        self, grav_solver: GravitationalSolver
    ) -> None:
        """Spot-check: first 50 rows must have CoT ending with \\boxed{answer}."""
        rows = _load_train_rows(_GRAV_PREFIX)[:50]
        for row in rows:
            cot, answer = grav_solver.solve(row["prompt"])
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT does not end with \\boxed{{{answer}}}"
            )


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataUnitConversion:
    """Verify 100% verify() accuracy on every unit-conversion row in train.csv."""

    def test_full_accuracy(self, unit_solver: UnitConversionSolver) -> None:
        rows = _load_train_rows(_UNIT_PREFIX)
        assert rows, "No unit-conversion rows found in train.csv"

        correct = 0
        total = len(rows)
        failures: list[tuple[str, str, str]] = []

        for row in rows:
            _, answer = unit_solver.solve(row["prompt"])
            if verify(row["answer"], answer):
                correct += 1
            else:
                failures.append((row["id"], row["answer"], answer))

        accuracy = correct / total
        logger.info(
            "UNIT_CONVERSION real-data accuracy: %d/%d = %.4f",
            correct, total, accuracy,
        )

        if failures:
            sample = failures[:5]
            logger.warning("Failures (showing first 5 of %d): %s", len(failures), sample)

        assert accuracy == 1.0, (
            f"UNIT_CONVERSION: {correct}/{total} correct (expected 1.0). "
            f"First failure: id={failures[0][0]} gold={failures[0][1]!r} "
            f"pred={failures[0][2]!r}"
        )

    def test_cot_ends_with_boxed_on_real_prompts(
        self, unit_solver: UnitConversionSolver
    ) -> None:
        rows = _load_train_rows(_UNIT_PREFIX)[:50]
        for row in rows:
            cot, answer = unit_solver.solve(row["prompt"])
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT does not end with \\boxed{{{answer}}}"
            )


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataNumeral:
    """Verify 100% verify() accuracy on every numeral row in train.csv."""

    def test_full_accuracy(self, numeral_solver: NumeralSolver) -> None:
        rows = _load_train_rows(_NUMERAL_PREFIX)
        assert rows, "No numeral rows found in train.csv"

        correct = 0
        total = len(rows)
        failures: list[tuple[str, str, str]] = []

        for row in rows:
            _, answer = numeral_solver.solve(row["prompt"])
            if verify(row["answer"], answer):
                correct += 1
            else:
                failures.append((row["id"], row["answer"], answer))

        accuracy = correct / total
        logger.info(
            "NUMERAL real-data accuracy: %d/%d = %.4f",
            correct, total, accuracy,
        )

        if failures:
            sample = failures[:5]
            logger.warning("Failures (showing first 5 of %d): %s", len(failures), sample)

        assert accuracy == 1.0, (
            f"NUMERAL: {correct}/{total} correct (expected 1.0). "
            f"First failure: id={failures[0][0]} gold={failures[0][1]!r} "
            f"pred={failures[0][2]!r}"
        )
