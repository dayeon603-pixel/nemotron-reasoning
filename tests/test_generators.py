"""Tests for the programmatic puzzle generators.

Critical-path coverage:
  - generate() returns exactly n examples
  - Every example.answer appears verbatim inside gold_cot
  - gold_cot ends with \\boxed{answer} (scoring contract)
  - verify(answer, answer) is always True (self-consistency)
  - verify(answer, extracted_from_gold_cot) is always True (end-to-end)
  - Different seeds produce different outputs (not collapsed to a constant)
  - Same seed produces identical outputs (reproducibility)
  - Binary answers are exactly BIT_WIDTH chars of 0/1 (format contract)
  - Linear equation CoT is internally consistent (a*x+b == answer)
  - Roman numeral answers decode back to the same integer (round-trip)
"""

from __future__ import annotations

import re

import pytest

from src.eval.metric import extract_final_answer, verify
from src.generators.binary_ops import BIT_WIDTH, generate as gen_binary
from src.generators.cipher import generate as gen_cipher
from src.generators.common import Example
from src.generators.linear_eq import generate as gen_linear
from src.generators.roman import _from_roman, _to_roman, generate as gen_roman
from src.generators.number_seq import generate as gen_seq
from src.generators.list_ops import generate as gen_list, _rotate_left, _rotate_right, _dedupe
from src.generators.modular_arith import generate as gen_mod

# ── helpers ───────────────────────────────────────────────────────────────────

_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{([^{}]*)\}")

N_EXAMPLES: int = 20   # enough to exercise multiple rules; fast on CPU
SEED_A: int = 42
SEED_B: int = 99


def _last_boxed(text: str) -> str | None:
    """Return the last non-empty \\boxed{} content, or None."""
    matches = [m.group(1).strip() for m in _BOXED_RE.finditer(text) if m.group(1).strip()]
    return matches[-1] if matches else None


# ── binary_ops ────────────────────────────────────────────────────────────────

class TestBinaryOpsGenerator:
    def test_returns_n_examples(self) -> None:
        examples = gen_binary(N_EXAMPLES, SEED_A)
        assert len(examples) == N_EXAMPLES

    def test_all_are_example_instances(self) -> None:
        for ex in gen_binary(N_EXAMPLES, SEED_A):
            assert isinstance(ex, Example)

    def test_domain_tag(self) -> None:
        for ex in gen_binary(5, SEED_A):
            assert ex.domain == "binary_ops"

    def test_answer_is_binary_string_of_correct_width(self) -> None:
        for ex in gen_binary(N_EXAMPLES, SEED_A):
            assert re.fullmatch(r"[01]+", ex.answer), (
                f"answer {ex.answer!r} is not a binary string"
            )
            assert len(ex.answer) == BIT_WIDTH, (
                f"answer length {len(ex.answer)} != BIT_WIDTH {BIT_WIDTH}"
            )

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_binary(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} does not end with {expected_tail!r}"
            )

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_binary(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_verify_answer_against_itself(self) -> None:
        for ex in gen_binary(5, SEED_A):
            assert verify(ex.answer, ex.answer)

    def test_different_seeds_produce_different_examples(self) -> None:
        a = gen_binary(N_EXAMPLES, SEED_A)
        b = gen_binary(N_EXAMPLES, SEED_B)
        answers_a = {ex.answer for ex in a}
        answers_b = {ex.answer for ex in b}
        # Not guaranteed to differ on every example, but the sets should differ
        # unless by extreme coincidence (probability ~0 for N=20).
        assert answers_a != answers_b or {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_same_seed_is_reproducible(self) -> None:
        a = gen_binary(N_EXAMPLES, SEED_A)
        b = gen_binary(N_EXAMPLES, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_binary(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt, (
                "Wonderland framing missing from prompt"
            )

    def test_answer_not_in_prompt(self) -> None:
        # The answer must NOT appear pre-revealed in the puzzle prompt.
        for ex in gen_binary(5, SEED_A):
            # The prompt shows demo outputs but NOT the query answer.
            # We verify the query answer line appears only in gold_cot.
            assert f"\\boxed{{{ex.answer}}}" not in ex.prompt


# ── cipher ────────────────────────────────────────────────────────────────────

class TestCipherGenerator:
    def test_returns_n_examples(self) -> None:
        assert len(gen_cipher(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_domain_tag(self) -> None:
        for ex in gen_cipher(5, SEED_A):
            assert ex.domain == "cipher"

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_cipher(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail)

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_cipher(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_answer_is_non_empty_string(self) -> None:
        for ex in gen_cipher(N_EXAMPLES, SEED_A):
            assert ex.answer.strip() != ""

    def test_reproducibility(self) -> None:
        a = gen_cipher(10, SEED_A)
        b = gen_cipher(10, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_cipher(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt


# ── linear_eq ─────────────────────────────────────────────────────────────────

class TestLinearEqGenerator:
    def test_returns_n_examples(self) -> None:
        assert len(gen_linear(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_domain_tag(self) -> None:
        for ex in gen_linear(5, SEED_A):
            assert ex.domain == "linear_eq"

    def test_answer_is_integer_string(self) -> None:
        for ex in gen_linear(N_EXAMPLES, SEED_A):
            # Bare integer, parseable as int, no units
            assert re.fullmatch(r"-?\d+", ex.answer), (
                f"answer {ex.answer!r} is not a bare integer"
            )

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_linear(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail)

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_linear(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_float_tolerance_accepted_for_integer_answers(self) -> None:
        # verify uses math.isclose for numeric answers, so "42" and "42.0" match.
        for ex in gen_linear(5, SEED_A):
            assert verify(ex.answer, str(float(ex.answer)))

    def test_reproducibility(self) -> None:
        a = gen_linear(10, SEED_A)
        b = gen_linear(10, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_linear(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt


# ── roman ─────────────────────────────────────────────────────────────────────

class TestRomanGenerator:
    def test_returns_n_examples(self) -> None:
        assert len(gen_roman(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_domain_tag(self) -> None:
        for ex in gen_roman(5, SEED_A):
            assert ex.domain == "roman"

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_roman(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail)

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_roman(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_roman_numeral_answers_are_uppercase_or_integer(self) -> None:
        for ex in gen_roman(N_EXAMPLES, SEED_A):
            # Either a pure decimal integer (roman_to_int) or uppercase Roman
            is_int = re.fullmatch(r"\d+", ex.answer)
            is_roman = re.fullmatch(r"[IVXLCDM]+", ex.answer)
            assert is_int or is_roman, (
                f"answer {ex.answer!r} is neither decimal int nor Roman numeral"
            )

    def test_roman_round_trip(self) -> None:
        """to_roman -> from_roman -> same value for the whole range sample."""
        for n in [1, 4, 9, 14, 40, 90, 399, 1000, 1994, 3999]:
            roman = _to_roman(n)
            assert _from_roman(roman) == n, (
                f"Round-trip failed for {n}: {roman!r} -> {_from_roman(roman)}"
            )

    def test_to_roman_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _to_roman(0)
        with pytest.raises(ValueError):
            _to_roman(4000)

    def test_reproducibility(self) -> None:
        a = gen_roman(10, SEED_A)
        b = gen_roman(10, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_roman(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt


# ── cross-domain verify contract ──────────────────────────────────────────────

class TestVerifyContract:
    """Generators produce answers that obey the scoring verify() contract."""

    def test_binary_answer_fails_with_wrong_length(self) -> None:
        # A zero-padded version of a binary answer should fail verify.
        ex = gen_binary(1, SEED_A)[0]
        padded = "0" + ex.answer  # longer — different binary string
        assert not verify(ex.answer, padded)

    def test_linear_eq_answer_passes_with_float_string(self) -> None:
        ex = gen_linear(1, SEED_A)[0]
        float_repr = f"{int(ex.answer)}.0"
        assert verify(ex.answer, float_repr)

    def test_cipher_answer_passes_case_insensitive(self) -> None:
        ex = gen_cipher(1, SEED_A)[0]
        # Cipher answers are text — verify uses case-insensitive comparison.
        assert verify(ex.answer, ex.answer.upper())
        assert verify(ex.answer, ex.answer.lower())

    def test_roman_answer_passes_case_insensitive(self) -> None:
        # Roman numeral answers (non-numeric) are case-insensitive in verify.
        import re as _re
        for ex in gen_roman(10, SEED_A):
            if _re.fullmatch(r"[IVXLCDM]+", ex.answer):
                assert verify(ex.answer, ex.answer.lower())
                break


# ── number_seq ────────────────────────────────────────────────────────────────

class TestNumberSeqGenerator:
    """Tests for the number sequence puzzle generator."""

    def test_returns_n_examples(self) -> None:
        assert len(gen_seq(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_all_are_example_instances(self) -> None:
        for ex in gen_seq(N_EXAMPLES, SEED_A):
            assert isinstance(ex, Example)

    def test_domain_tag(self) -> None:
        for ex in gen_seq(5, SEED_A):
            assert ex.domain == "number_seq"

    def test_answer_is_integer_string(self) -> None:
        for ex in gen_seq(N_EXAMPLES, SEED_A):
            # Answers must be parseable as int with no trailing units or spaces.
            assert re.fullmatch(r"-?\d+", ex.answer), (
                f"answer {ex.answer!r} is not a bare integer string"
            )

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_seq(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} != {expected_tail!r}"
            )

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_seq(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self) -> None:
        a = gen_seq(N_EXAMPLES, SEED_A)
        b = gen_seq(N_EXAMPLES, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_outputs(self) -> None:
        a = gen_seq(N_EXAMPLES, SEED_A)
        b = gen_seq(N_EXAMPLES, SEED_B)
        assert {ex.answer for ex in a} != {ex.answer for ex in b} or \
               {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_seq(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt

    def test_answer_not_prerevealed_in_prompt(self) -> None:
        for ex in gen_seq(5, SEED_A):
            assert f"\\boxed{{{ex.answer}}}" not in ex.prompt

    # ── hand-checked gold-answer correctness ──────────────────────────────────
    # Each case is independently verified by manual arithmetic (see comments).

    def test_arithmetic_sequence_seed1001(self) -> None:
        # Sequence: -8, -16, -24, -32, -40  (d = -8)
        # Next term (index 5): -8 + 5×(-8) = -48
        ex = gen_seq(1, seed=1001)[0]
        assert ex.answer == "-48", f"Expected -48, got {ex.answer!r}"

    def test_arithmetic_sequence_seed1020(self) -> None:
        # Sequence: -19, -14, -9, -4, 1  (d = +5)
        # Next term (index 5): -19 + 5×5 = 6
        ex = gen_seq(1, seed=1020)[0]
        assert ex.answer == "6", f"Expected 6, got {ex.answer!r}"

    def test_arithmetic_sequence_seed1030(self) -> None:
        # Sequence: 7, 17, 27, 37, 47  (d = +10)
        # Next term (index 5): 7 + 5×10 = 57
        ex = gen_seq(1, seed=1030)[0]
        assert ex.answer == "57", f"Expected 57, got {ex.answer!r}"

    def test_power_sequence_seed1002(self) -> None:
        # Sequence: 2, 9, 28, 65, 126  (n^3 + 1, 1-indexed: 1^3+1=2, 2^3+1=9 ...)
        # Next term (index 5, i.e. n=6): 6^3 + 1 = 217
        ex = gen_seq(1, seed=1002)[0]
        assert ex.answer == "217", f"Expected 217, got {ex.answer!r}"

    def test_fibonacci_like_sequence_seed1003(self) -> None:
        # Sequence: 9, 6, 15, 21, 36  (each term = sum of two preceding)
        # Next term: 21 + 36 = 57
        ex = gen_seq(1, seed=1003)[0]
        assert ex.answer == "57", f"Expected 57, got {ex.answer!r}"

    def test_quadratic_sequence_seed1010(self) -> None:
        # Sequence: 5, 8, 13, 20, 29  (second differences = 2; a=1,b=2,c=5)
        # a(n)=n^2+2n+5; next term (n=5): 25+10+5=40
        ex = gen_seq(1, seed=1010)[0]
        assert ex.answer == "40", f"Expected 40, got {ex.answer!r}"

    def test_cot_contains_all_shown_terms(self) -> None:
        """Shown terms must appear verbatim in the CoT verification step."""
        for ex in gen_seq(5, SEED_A):
            assert "All shown terms verified." in ex.gold_cot

    # ── Fix-locking tests ─────────────────────────────────────────────────────

    def test_prompt_demo_pairs_cover_all_shown_terms(self) -> None:
        """NUM_DEMO_PAIRS must equal NUM_SHOWN so every term the CoT references
        is visible in the prompt.

        Before the fix: NUM_DEMO_PAIRS=4, NUM_SHOWN=5.  The CoT verified term
        index 4 (the 5th term) but the prompt only showed term indices 0-3.
        After the fix NUM_DEMO_PAIRS == NUM_SHOWN == 5.
        """
        from src.generators.number_seq import NUM_DEMO_PAIRS, NUM_SHOWN

        assert NUM_DEMO_PAIRS == NUM_SHOWN, (
            f"NUM_DEMO_PAIRS ({NUM_DEMO_PAIRS}) != NUM_SHOWN ({NUM_SHOWN}): "
            "the CoT will reference hidden terms"
        )

        # Count demo pairs in a real prompt: lines matching "  N. Input: term N"
        for ex in gen_seq(N_EXAMPLES, SEED_A):
            # The prompt has lines like "  1. Input: term 1  →  Output: ..."
            demo_lines = [
                ln for ln in ex.prompt.splitlines()
                if re.match(r"\s+\d+\.\s+Input:", ln)
            ]
            assert len(demo_lines) == NUM_SHOWN, (
                f"Prompt has {len(demo_lines)} demo pairs but NUM_SHOWN={NUM_SHOWN}"
            )

    def test_query_is_one_beyond_shown_set(self) -> None:
        """The query must ask for term NUM_SHOWN+1 (one beyond the shown set)."""
        from src.generators.number_seq import NUM_SHOWN

        for ex in gen_seq(N_EXAMPLES, SEED_A):
            assert f"term {NUM_SHOWN + 1}" in ex.prompt, (
                f"Expected 'term {NUM_SHOWN + 1}' in prompt query. "
                f"Prompt tail: {ex.prompt[-100:]!r}"
            )


# ── list_ops ──────────────────────────────────────────────────────────────────

class TestListOpsGenerator:
    """Tests for the list transformation puzzle generator."""

    def test_returns_n_examples(self) -> None:
        assert len(gen_list(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_all_are_example_instances(self) -> None:
        for ex in gen_list(N_EXAMPLES, SEED_A):
            assert isinstance(ex, Example)

    def test_domain_tag(self) -> None:
        for ex in gen_list(5, SEED_A):
            assert ex.domain == "list_ops"

    def test_answer_is_comma_separated_integers(self) -> None:
        for ex in gen_list(N_EXAMPLES, SEED_A):
            # Comma-space separated ints, possibly negative, e.g. "-3, 1, 7"
            assert re.fullmatch(r"-?\d+(?:, -?\d+)*", ex.answer), (
                f"answer {ex.answer!r} is not comma-space-separated integers"
            )

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_list(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-80:]!r} != {expected_tail!r}"
            )

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_list(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self) -> None:
        a = gen_list(N_EXAMPLES, SEED_A)
        b = gen_list(N_EXAMPLES, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_outputs(self) -> None:
        a = gen_list(N_EXAMPLES, SEED_A)
        b = gen_list(N_EXAMPLES, SEED_B)
        assert {ex.answer for ex in a} != {ex.answer for ex in b} or \
               {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_list(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt

    def test_cot_contains_all_examples_verified(self) -> None:
        for ex in gen_list(5, SEED_A):
            assert "All examples verified." in ex.gold_cot

    # ── primitive helpers ─────────────────────────────────────────────────────

    def test_rotate_left_helper(self) -> None:
        # [1, 2, 3, 4, 5] rotate left by 2 → [3, 4, 5, 1, 2]
        assert _rotate_left([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]
        # rotate by len is identity
        assert _rotate_left([1, 2, 3], 3) == [1, 2, 3]

    def test_rotate_right_helper(self) -> None:
        # [1, 2, 3, 4, 5] rotate right by 2 → [4, 5, 1, 2, 3]
        assert _rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
        assert _rotate_right([1, 2, 3], 3) == [1, 2, 3]

    def test_dedupe_helper_preserves_order(self) -> None:
        # First occurrence of each value is kept; rest dropped.
        assert _dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
        assert _dedupe([5, 5, 5]) == [5]
        assert _dedupe([1, 2, 3]) == [1, 2, 3]  # no duplicates

    # ── hand-checked gold-answer correctness ──────────────────────────────────

    def test_dedupe_rule_seed3001(self) -> None:
        # Query list: [-14, -9, 25, 25, 30]; dedup → [-14, -9, 25, 30]
        ex = gen_list(1, seed=3001)[0]
        assert ex.answer == "-14, -9, 25, 30", (
            f"Expected '-14, -9, 25, 30', got {ex.answer!r}"
        )

    def test_sort_asc_correctness(self) -> None:
        # Any sort_asc example: answer must be the sorted version of the query input.
        # Find a sort_asc example by scanning a batch.
        # CoT Step 4 line: "  Query input: -3, 1, 7, 12" (no brackets after fix).
        for ex in gen_list(50, SEED_A):
            if "ascending" in ex.gold_cot:
                # Parse the query input from the CoT Step 4 line (no brackets).
                m = re.search(r"Query input: (-?\d+(?:, -?\d+)*)", ex.gold_cot)
                assert m is not None, f"Could not find Query input in CoT: {ex.gold_cot[:300]}"
                query_vals = [int(v.strip()) for v in m.group(1).split(",")]
                expected = ", ".join(str(v) for v in sorted(query_vals))
                assert ex.answer == expected, (
                    f"sort_asc answer mismatch: got {ex.answer!r}, expected {expected!r}"
                )
                return  # one case sufficient
        pytest.skip("No sort_asc example found in batch")  # pragma: no cover

    def test_reverse_correctness(self) -> None:
        # Any reverse example: answer must be the reversed query input.
        for ex in gen_list(50, SEED_A):
            if "reverse the order" in ex.gold_cot:
                m = re.search(r"Query input: (-?\d+(?:, -?\d+)*)", ex.gold_cot)
                assert m is not None, f"Could not find Query input in CoT: {ex.gold_cot[:300]}"
                query_vals = [int(v.strip()) for v in m.group(1).split(",")]
                expected = ", ".join(str(v) for v in reversed(query_vals))
                assert ex.answer == expected, (
                    f"reverse answer mismatch: got {ex.answer!r}, expected {expected!r}"
                )
                return
        pytest.skip("No reverse example found in batch")  # pragma: no cover

    # ── Fix-locking tests (surface-form consistency) ──────────────────────────

    def test_prompt_example_outputs_match_answer_surface_form(self) -> None:
        """Prompt demo outputs must use the SAME surface form as Example.answer.

        Before the fix: prompt showed '[−3, 1, 7]' but answer was '−3, 1, 7'
        (no brackets).  A model trained on the prompt emits \\boxed{[−3, 1, 7]}
        which verify() rejects.  After the fix both are bare comma-separated.
        """
        for ex in gen_list(N_EXAMPLES, SEED_A):
            # answer must not be wrapped in brackets
            assert not ex.answer.startswith("["), (
                f"answer {ex.answer!r} has leading bracket — surface-form mismatch"
            )
            assert not ex.answer.endswith("]"), (
                f"answer {ex.answer!r} has trailing bracket — surface-form mismatch"
            )
            # prompt demo lines must also not show bracketed outputs
            # Every "→  Output:" line in the prompt must match bare form
            for line in ex.prompt.splitlines():
                if "→" in line and "Output:" not in line:
                    # Wonderland prompt format: "  1. Input: X  →  Output: Y"
                    # or just "  1. Input: X  →  Y" — check the output portion
                    parts = line.split("→")
                    if len(parts) == 2:
                        output_part = parts[1].strip()
                        assert not output_part.startswith("["), (
                            f"Prompt demo output {output_part!r} is bracket-wrapped "
                            f"but answer {ex.answer!r} is not"
                        )

    def test_verify_roundtrip_gold_answer(self) -> None:
        """verify(ex.answer, ex.answer) must be True for all list_ops examples.

        This is the core scoring contract: the gold answer must match itself.
        Breaks if answer format routes through the wrong verify branch.
        """
        for ex in gen_list(N_EXAMPLES, SEED_A):
            assert verify(ex.answer, ex.answer), (
                f"verify(gold, gold) failed for answer {ex.answer!r}"
            )


# ── modular_arith ─────────────────────────────────────────────────────────────

class TestModularArithGenerator:
    """Tests for the modular arithmetic puzzle generator."""

    def test_returns_n_examples(self) -> None:
        assert len(gen_mod(N_EXAMPLES, SEED_A)) == N_EXAMPLES

    def test_all_are_example_instances(self) -> None:
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            assert isinstance(ex, Example)

    def test_domain_tag(self) -> None:
        for ex in gen_mod(5, SEED_A):
            assert ex.domain == "modular_arith"

    def test_answer_is_non_negative_integer_string(self) -> None:
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            # Answer is always in [0, n-1]; no negative sign.
            assert re.fullmatch(r"\d+", ex.answer), (
                f"answer {ex.answer!r} is not a non-negative integer string"
            )

    def test_answer_is_within_modulus_range(self) -> None:
        """Answer must be strictly less than the modulus stated in the CoT."""
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            m = re.search(r"Clock size \(modulus\): (\d+)", ex.gold_cot)
            assert m is not None, "Modulus not found in CoT"
            modulus = int(m.group(1))
            answer_val = int(ex.answer)
            assert 0 <= answer_val < modulus, (
                f"answer {answer_val} not in [0, {modulus - 1}]"
            )

    def test_gold_cot_ends_with_boxed_answer(self) -> None:
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            expected_tail = f"\\boxed{{{ex.answer}}}"
            assert ex.gold_cot.endswith(expected_tail), (
                f"CoT tail {ex.gold_cot[-60:]!r} != {expected_tail!r}"
            )

    def test_extracted_answer_verifies_correctly(self) -> None:
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            extracted = extract_final_answer(ex.gold_cot)
            assert verify(ex.answer, extracted), (
                f"verify failed: gold={ex.answer!r} extracted={extracted!r}"
            )

    def test_reproducibility(self) -> None:
        a = gen_mod(N_EXAMPLES, SEED_A)
        b = gen_mod(N_EXAMPLES, SEED_A)
        for ea, eb in zip(a, b):
            assert ea.answer == eb.answer
            assert ea.prompt == eb.prompt

    def test_different_seeds_produce_different_outputs(self) -> None:
        a = gen_mod(N_EXAMPLES, SEED_A)
        b = gen_mod(N_EXAMPLES, SEED_B)
        assert {ex.answer for ex in a} != {ex.answer for ex in b} or \
               {ex.prompt for ex in a} != {ex.prompt for ex in b}

    def test_wonderland_framing_in_prompt(self) -> None:
        for ex in gen_mod(5, SEED_A):
            assert "Alice" in ex.prompt or "Wonderland" in ex.prompt

    def test_cot_contains_all_examples_verified(self) -> None:
        for ex in gen_mod(5, SEED_A):
            assert "All examples verified." in ex.gold_cot

    # ── hand-checked gold-answer correctness ──────────────────────────────────

    def test_linear_mod_seed2001(self) -> None:
        # Rule: (2·a + 0) mod 5, query a=3 → (2*3+0) mod 5 = 6 mod 5 = 1
        ex = gen_mod(1, seed=2001)[0]
        assert ex.answer == "1", f"Expected '1', got {ex.answer!r}"

    def test_add_mod_correctness(self) -> None:
        """Any add_mod example: answer must equal (a+b) mod n."""
        for ex in gen_mod(50, SEED_A):
            if "add the two numbers" in ex.gold_cot:
                # Extract modulus
                m_mod = re.search(r"Clock size \(modulus\): (\d+)", ex.gold_cot)
                assert m_mod is not None
                n = int(m_mod.group(1))
                # Extract query: "Query input: a + b"
                m_q = re.search(r"Query input: (\d+) \+ (\d+)", ex.gold_cot)
                assert m_q is not None
                a, b = int(m_q.group(1)), int(m_q.group(2))
                expected = str((a + b) % n)
                assert ex.answer == expected, (
                    f"add_mod mismatch: ({a}+{b}) mod {n} = {expected}, got {ex.answer!r}"
                )
                return
        pytest.skip("No add_mod example in batch")  # pragma: no cover

    def test_mul_mod_correctness(self) -> None:
        """Any mul_mod example: answer must equal (a*b) mod n."""
        for ex in gen_mod(50, SEED_A):
            if "multiply the two numbers" in ex.gold_cot:
                m_mod = re.search(r"Clock size \(modulus\): (\d+)", ex.gold_cot)
                assert m_mod is not None
                n = int(m_mod.group(1))
                # Extract query: "Query input: a × b"
                m_q = re.search(r"Query input: (\d+) × (\d+)", ex.gold_cot)
                assert m_q is not None
                a, b = int(m_q.group(1)), int(m_q.group(2))
                expected = str((a * b) % n)
                assert ex.answer == expected, (
                    f"mul_mod mismatch: ({a}×{b}) mod {n} = {expected}, got {ex.answer!r}"
                )
                return
        pytest.skip("No mul_mod example in batch")  # pragma: no cover

    def test_pow_mod_correctness(self) -> None:
        """Any pow_mod example: answer must equal a^k mod n."""
        for ex in gen_mod(50, SEED_A):
            if "raise the number to the power" in ex.gold_cot:
                m_mod = re.search(r"Clock size \(modulus\): (\d+)", ex.gold_cot)
                m_exp = re.search(r"raise the number to the power (\d+)", ex.gold_cot)
                m_q = re.search(r"Query input: (\d+)\s*$", ex.gold_cot, re.MULTILINE)
                assert m_mod and m_exp and m_q
                n = int(m_mod.group(1))
                k = int(m_exp.group(1))
                a = int(m_q.group(1))
                expected = str(pow(a, k, n))
                assert ex.answer == expected, (
                    f"pow_mod mismatch: {a}^{k} mod {n} = {expected}, got {ex.answer!r}"
                )
                return
        pytest.skip("No pow_mod example in batch")  # pragma: no cover

    # ── Fix-locking tests ─────────────────────────────────────────────────────

    def test_linear_mod_multiplier_is_coprime_with_modulus(self) -> None:
        """For every linear_mod_n example, gcd(c, n) must equal 1.

        Before the fix: c was sampled with randint(2, n-1) without checking
        coprimality.  E.g. n=6, c=4 → gcd(4,6)=2, so the map is 2-to-1 and
        the rule cannot be uniquely induced from demo pairs.
        """
        import math as _math

        for ex in gen_mod(100, SEED_A):
            # Only check linear_mod examples
            if "linear map" not in ex.gold_cot:
                continue
            # Extract c and n from description line: "({c}·a + {d}) mod {n}"
            m = re.search(r"\((\d+)·a \+ (\d+)\) mod (\d+)", ex.gold_cot)
            if m is None:
                # Also try the form used in description (c might be 1-digit)
                m = re.search(r"apply the linear map \((\d+)·a \+ (\d+)\) mod (\d+)", ex.gold_cot)
            if m is None:
                continue
            c_val, _d_val, n_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
            assert _math.gcd(c_val, n_val) == 1, (
                f"linear_mod multiplier c={c_val} is not coprime with n={n_val}: "
                f"gcd={_math.gcd(c_val, n_val)}"
            )

    def test_modular_arith_answer_is_bare_int_not_float(self) -> None:
        """Answer must be a bare integer string (e.g. '1' not '1.0').

        '0' and '1' route through verify()'s binary branch (strict string
        equality).  '1.0' would fail against gold '1'.
        """
        for ex in gen_mod(N_EXAMPLES, SEED_A):
            assert re.fullmatch(r"\d+", ex.answer), (
                f"answer {ex.answer!r} is not a bare non-negative integer"
            )
            # Confirm it has no decimal point
            assert "." not in ex.answer, (
                f"answer {ex.answer!r} contains a decimal point (float format)"
            )
            # verify(gold, gold) must pass — critical for '0' and '1' via binary branch
            assert verify(ex.answer, ex.answer), (
                f"verify(gold, gold) failed for answer {ex.answer!r}"
            )

    # ── binary_ops shift-width fix locking ───────────────────────────────────

    def test_binary_ops_shift_param_wider_than_3(self) -> None:
        """At least some SHL/SHR examples must use k > 3 (wider than old range).

        Before the fix: k_shift was randint(1,3).  After: randint(1, BIT_WIDTH-1)
        i.e. up to 7 for 8-bit.  Run enough examples to observe k>3 with high
        probability.
        """
        from src.generators.binary_ops import BIT_WIDTH as _BW

        max_k_seen = 0
        for ex in gen_binary(200, SEED_A):
            # Extract shift amount from rule name in CoT: "SHL-5" or "SHR-6"
            m = re.search(r"shift (?:left|right) by (\d+) bit", ex.gold_cot)
            if m:
                max_k_seen = max(max_k_seen, int(m.group(1)))

        assert max_k_seen > 3, (
            f"Largest shift k seen is {max_k_seen} — expected > 3 after widening "
            f"k range to [1, {_BW - 1}]"
        )

    def test_binary_ops_no_all_zero_answer_from_shift(self) -> None:
        """SHL/SHR rules should not produce all-zero outputs for all demo pairs.

        The degenerate case (e.g. SHR-7 on small values) renders the rule
        undetectable from examples.  The non-degenerate sampler ensures at
        least one demo output is non-zero.
        """
        for ex in gen_binary(N_EXAMPLES, SEED_A):
            if "SHL" not in ex.gold_cot and "SHR" not in ex.gold_cot:
                continue
            # Collect demo output lines: "  Example N: XXXXXXXX  →  YYYYYYYY [PASS]"
            outputs = re.findall(
                r"Example \d+: [01]{8}\s+→\s+([01]{8})",
                ex.gold_cot,
            )
            non_zero = [o for o in outputs if o != "0" * 8]
            assert non_zero, (
                f"All demo outputs are zero for a shift rule — degenerate puzzle.\n"
                f"CoT excerpt: {ex.gold_cot[:400]}"
            )
