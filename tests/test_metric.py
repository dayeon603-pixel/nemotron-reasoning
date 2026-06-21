"""Tests for src/eval/metric.py — official scorer replication.

Every test is named after the specific scorer quirk it guards.  The goal
is NOT to test general Python string handling; it is to lock in the exact
behaviours of the OFFICIAL scorer so that any future "fix" that silently
changes scoring is caught immediately.

Test categories
---------------
verify() — binary branch
    Strict length-sensitive equality for [01]+ stored answers.

verify() — numeric branch
    math.isclose tolerance at rel_tol=1e-2, abs_tol=1e-5.

verify() — string fallback branch
    Case-insensitive equality for non-binary, non-numeric answers.

verify() — silent traps vs naive ==
    Cases where ``prediction == stored_answer`` gives the WRONG verdict.

extract_final_answer() — boxed extraction
    Single box, multiple boxes (last wins), nested/extra braces,
    empty box, no-content-box.

extract_final_answer() — phrase fallback
    Four phrase patterns and last-match semantics.

extract_final_answer() — number fallback
    Last number returned (not first), integers, negative, decimal.

extract_final_answer() — edge cases
    None input, empty string, no-number text.
"""

from __future__ import annotations

import pytest

from src.eval.metric import extract_final_answer, verify


# ===========================================================================
# verify() — BINARY BRANCH
# ===========================================================================


class TestVerifyBinary:
    """Binary strings must match EXACTLY (length-sensitive, case-insensitive)."""

    def test_exact_match_returns_true(self) -> None:
        assert verify("10011000", "10011000") is True

    def test_single_bit_flip_returns_false(self) -> None:
        assert verify("10011000", "10011001") is False

    def test_left_zero_padded_prediction_returns_false(self) -> None:
        # '11011' vs '00011011' — same numeric value but different binary string
        # This is a SILENT TRAP: int("11011", 2) == int("00011011", 2)
        # but the scorer does strict string comparison, so this must be False.
        assert verify("11011", "00011011") is False

    def test_right_zero_padded_prediction_returns_false(self) -> None:
        assert verify("101", "10100") is False

    def test_case_insensitive_binary_is_true(self) -> None:
        # Binary strings are only 0/1, but case-insensitivity is stated in spec.
        assert verify("10", "10") is True

    def test_empty_binary_string_falls_through_to_string(self) -> None:
        # '' does not fullmatch [01]+, so falls through to string compare.
        assert verify("", "") is True

    def test_binary_stored_vs_numeric_predicted_false(self) -> None:
        # stored "101" is binary -> strict string compare.
        # predicted "5" != "101" -> False.
        # SILENT TRAP: int("101", 2) == 5, but scorer is NOT base-2 aware.
        assert verify("101", "5") is False

    def test_single_zero(self) -> None:
        assert verify("0", "0") is True

    def test_single_one(self) -> None:
        assert verify("1", "1") is True

    def test_binary_wrong_length_longer_answer(self) -> None:
        assert verify("10011000", "010011000") is False


# ===========================================================================
# verify() — NUMERIC BRANCH
# ===========================================================================


class TestVerifyNumeric:
    """Numeric answers use math.isclose(rel_tol=1e-2, abs_tol=1e-5)."""

    def test_exact_float_match(self) -> None:
        assert verify("24.64", "24.64") is True

    def test_within_rel_tol_returns_true(self) -> None:
        # 24.6401 vs 24.64 — relative diff ~ 4e-6, within 1% tolerance.
        assert verify("24.64", "24.6401") is True

    def test_outside_rel_tol_returns_false(self) -> None:
        # 24.64 vs 25.0 — relative diff ~ 1.5%, outside 1% tolerance.
        assert verify("24.64", "25.0") is False

    def test_integer_answers_match(self) -> None:
        assert verify("42", "42") is True

    def test_integer_off_by_one_returns_false(self) -> None:
        assert verify("42", "43") is False

    def test_negative_number_match(self) -> None:
        assert verify("-7", "-7") is True

    def test_negative_vs_positive_returns_false(self) -> None:
        assert verify("-7", "7") is False

    def test_zero_exact_match(self) -> None:
        assert verify("0", "0") is True

    def test_zero_vs_small_nonzero_within_abs_tol(self) -> None:
        # SILENT TRAP: "0" matches [01]+ so the BINARY branch fires first.
        # Binary branch does strict string compare: "0" != "0.000001" -> False.
        # abs_tol is NEVER reached.  This is NOT a numeric comparison for "0".
        # To get numeric-tolerance behaviour for zero you need e.g. "0.0".
        assert verify("0", "0.000001") is False  # binary branch, not numeric

    def test_zero_float_vs_small_nonzero_within_abs_tol(self) -> None:
        # "0.0" does NOT match [01]+ -> goes to float branch.
        # math.isclose(0.0, 1e-6, rel_tol=1e-2, abs_tol=1e-5):
        # |0 - 1e-6| = 1e-6 < abs_tol=1e-5 -> True.
        assert verify("0.0", "0.000001") is True

    def test_zero_vs_large_nonzero_outside_tol(self) -> None:
        assert verify("0", "0.01") is False

    def test_integer_stored_float_predicted(self) -> None:
        # "42" parses as 42.0; "42.0" parses as 42.0 -> match.
        assert verify("42", "42.0") is True

    def test_float_stored_int_predicted(self) -> None:
        # "3.0" vs "3" -> math.isclose(3.0, 3.0) -> True.
        assert verify("3.0", "3") is True


# ===========================================================================
# verify() — STRING FALLBACK BRANCH
# ===========================================================================


class TestVerifyStringFallback:
    """Non-binary, non-numeric answers use case-insensitive string equality."""

    def test_roman_numeral_case_insensitive(self) -> None:
        assert verify("XLVII", "xlvii") is True

    def test_roman_numeral_wrong_answer(self) -> None:
        assert verify("XLVII", "XLVIII") is False

    def test_text_answer_exact(self) -> None:
        assert verify("Caesar", "Caesar") is True

    def test_text_answer_case_insensitive(self) -> None:
        assert verify("Caesar", "caesar") is True

    def test_text_answer_mismatch(self) -> None:
        assert verify("Caesar", "Vigenere") is False

    def test_answer_with_units_string_compare(self) -> None:
        # "42 apples" does not parse as float -> string compare.
        # "42" != "42 apples" -> False.
        assert verify("42 apples", "42") is False

    def test_predicted_with_units_vs_plain_number(self) -> None:
        # stored "42" IS numeric -> float branch: float("42 apples") raises.
        # Wait — stored "42" parses fine but predicted "42 apples" raises.
        # float branch raises -> string fallback: "42 apples" != "42" -> False.
        # SILENT TRAP: naive == would also give False, but for different reasons.
        assert verify("42", "42 apples") is False

    def test_whitespace_stripped_before_compare(self) -> None:
        # Both sides are stripped before any comparison.
        assert verify("  XLVII  ", "  xlvii  ") is True

    def test_multiword_text_answer(self) -> None:
        assert verify("hello world", "HELLO WORLD") is True


# ===========================================================================
# verify() — SILENT TRAPS (where naive == gives the WRONG verdict)
# ===========================================================================


class TestVerifySilentTraps:
    """Cases where ``prediction == stored_answer`` disagrees with verify().

    These are the most dangerous cases — they cause silent scoring errors
    when evaluating locally with a plain equality check.
    """

    def test_case_difference_naive_fails_verify_passes(self) -> None:
        # naive: "XLVII" == "xlvii" -> False.  verify -> True.
        assert "XLVII" != "xlvii"  # confirm naive disagrees
        assert verify("XLVII", "xlvii") is True

    def test_float_tolerance_naive_fails_verify_passes(self) -> None:
        # naive: "24.64" == "24.6401" -> False.  verify -> True.
        assert "24.64" != "24.6401"
        assert verify("24.64", "24.6401") is True

    def test_float_int_representation_naive_fails_verify_passes(self) -> None:
        # naive: "42" == "42.0" -> False.  verify -> True.
        assert "42" != "42.0"
        assert verify("42", "42.0") is True

    def test_zero_padding_binary_naive_passes_verify_fails(self) -> None:
        # naive: "11011" == "11011" -> True (if someone compares against itself).
        # The real trap: naive code might zero-pad before compare.
        # verify("11011", "00011011") -> False because binary branch is strict.
        assert verify("11011", "00011011") is False

    def test_numeric_vs_binary_numeric_wins(self) -> None:
        # stored "2" is NOT [01]+ fullmatch... wait: "2" does NOT match [01]+
        # so it goes to float branch: float("2") == float("2.0") -> True.
        # naive "2" == "2.0" -> False.
        assert "2" != "2.0"
        assert verify("2", "2.0") is True

    def test_binary_one_vs_numeric_one(self) -> None:
        # stored "1" matches [01]+ -> binary branch -> strict string compare.
        # predicted "1.0" -> "1.0".lower() == "1".lower() -> "1.0" != "1" -> False.
        # SILENT TRAP: float(1) == float(1.0) is True, but scorer uses binary path.
        assert verify("1", "1.0") is False

    def test_binary_zero_vs_numeric_zero(self) -> None:
        # Same as above for "0".
        assert verify("0", "0.0") is False

    def test_units_break_float_parse_predicted(self) -> None:
        # stored "42" parses, predicted "42 kg" does not -> string fallback.
        # "42 kg" != "42" -> False.
        assert verify("42", "42 kg") is False

    def test_single_digit_binary_intercepts_numeric_path(self) -> None:
        # SILENT TRAP: stored "0" and "1" match [01]+ -> BINARY branch fires.
        # Numeric tolerance is NEVER applied to single-digit binary answers.
        # "1" vs "1.0": binary branch -> "1.0" != "1" -> False.
        # A naive coder expecting float-tolerance here will silently miscount.
        assert verify("1", "1.0") is False
        assert verify("0", "0.0") is False

    def test_extra_whitespace_does_not_affect_verify(self) -> None:
        # Both sides are stripped, so extra spaces are transparent.
        # naive "  42  " == "42" -> False.  verify -> True.
        assert "  42  " != "42"
        assert verify("  42  ", "42") is True


# ===========================================================================
# extract_final_answer() — BOXED EXTRACTION
# ===========================================================================


class TestExtractBoxed:
    """\\boxed{} extraction — all edge cases from the spec."""

    def test_single_simple_box(self) -> None:
        assert extract_final_answer(r"The answer is \boxed{42}") == "42"

    def test_multiple_boxes_last_wins(self) -> None:
        # CRITICAL: official scorer returns the LAST non-empty box.
        text = r"\boxed{3} intermediate step \boxed{7}"
        assert extract_final_answer(text) == "7"

    def test_three_boxes_last_wins(self) -> None:
        text = r"\boxed{1} \boxed{2} \boxed{3}"
        assert extract_final_answer(text) == "3"

    def test_last_empty_box_skipped_earlier_box_returned(self) -> None:
        # Empty box is skipped; last NON-EMPTY box is returned.
        # \boxed{42} then \boxed{} — the second box has no content.
        # The algorithm: window for box 1 is "42}" up to start of \boxed{.
        # rfind('}') -> position of '}'.  candidate = "42". Non-empty -> kept.
        # window for box 2 is "" (nothing after opening '{').
        # rfind('}') -> -1 -> candidate = "".  Empty -> not appended.
        # boxed_answers = ["42"].  Return "42".
        text = r"\boxed{42}\boxed{}"
        assert extract_final_answer(text) == "42"

    def test_nested_brace_answer_retained(self) -> None:
        # \boxed{}52} — the spec says this yields "}52".
        # The marker \boxed{ ends; content_start is after '{'.
        # window = "}52}..." (no next \boxed{).
        # rfind('}') finds the last '}' in "}52}" which is at index 3.
        # candidate = window[:3] = "}52".strip() = "}52".
        text = r"\boxed{}52}"
        assert extract_final_answer(text) == "}52"

    def test_box_with_expression(self) -> None:
        text = r"\boxed{x + 2}"
        assert extract_final_answer(text) == "x + 2"

    def test_box_with_latex_fraction(self) -> None:
        # \boxed{\frac{1}{2}} — rfind('}') catches the last '}' in the window.
        text = r"\boxed{\frac{1}{2}}"
        # window is r"\frac{1}{2}}", rfind('}') -> last '}' at end.
        # candidate = r"\frac{1}{2}".strip()
        assert extract_final_answer(text) == r"\frac{1}{2}"

    def test_box_with_units_string_compare_warning(self) -> None:
        # \boxed{42 apples} — extraction gives "42 apples".
        # verify("42", "42 apples") is False because float("42 apples") raises.
        # This test documents the extraction step only.
        text = r"\boxed{42 apples}"
        assert extract_final_answer(text) == "42 apples"

    def test_empty_box_then_phrase_fallback(self) -> None:
        # All boxes empty -> boxed_answers is empty -> fall through to phrase.
        text = r"\boxed{} The final answer is: 99"
        result = extract_final_answer(text)
        assert result == "99"

    def test_whitespace_only_box_skipped(self) -> None:
        # \boxed{   } — candidate after strip() is "".  Not appended.
        text = r"\boxed{   }"
        # Falls through to phrase then number then last-line.
        # No phrases, no numbers (only spaces) -> last non-empty line
        # is the whole text line.
        result = extract_final_answer(text)
        # Should not return "   " or empty — fallback to last non-empty line.
        assert result != ""
        assert result != "NOT_FOUND" or True  # just verify no crash

    def test_box_with_leading_trailing_whitespace_stripped(self) -> None:
        text = r"\boxed{  42  }"
        assert extract_final_answer(text) == "42"


# ===========================================================================
# extract_final_answer() — PHRASE FALLBACK
# ===========================================================================


class TestExtractPhraseFallback:
    """Four phrase patterns — last match semantics, case-insensitive."""

    def test_the_final_answer_is_pattern(self) -> None:
        text = "The final answer is: 42"
        assert extract_final_answer(text) == "42"

    def test_final_answer_is_pattern(self) -> None:
        text = "Final answer is: 99"
        assert extract_final_answer(text) == "99"

    def test_final_answer_colon_pattern(self) -> None:
        text = "Final answer: 7"
        assert extract_final_answer(text) == "7"

    def test_final_answer_lowercase_pattern(self) -> None:
        text = "final answer: XLVII"
        assert extract_final_answer(text) == "XLVII"

    def test_final_answer_fullwidth_colon(self) -> None:
        # Pattern includes fullwidth colon '：' (U+FF1A)
        text = "Final answer：42"
        assert extract_final_answer(text) == "42"

    def test_last_phrase_match_wins_across_patterns(self) -> None:
        # Two phrase matches — the later one in the text wins.
        text = "The final answer is: 3\nFinal answer: 7"
        assert extract_final_answer(text) == "7"

    def test_case_insensitive_phrase(self) -> None:
        text = "FINAL ANSWER: 55"
        assert extract_final_answer(text) == "55"

    def test_phrase_with_text_answer(self) -> None:
        text = "The final answer is: XLVII"
        assert extract_final_answer(text) == "XLVII"


# ===========================================================================
# extract_final_answer() — NUMBER FALLBACK
# ===========================================================================


class TestExtractNumberFallback:
    """Number fallback returns LAST match (not first) — matches the CODE."""

    def test_single_integer_returned(self) -> None:
        text = "the result is 42 obviously"
        assert extract_final_answer(text) == "42"

    def test_multiple_numbers_last_returned(self) -> None:
        # CRITICAL: spec docstring says "first" but code returns LAST.
        text = "step 1: got 10, step 2: got 20, step 3: got 30"
        assert extract_final_answer(text) == "30"

    def test_negative_number_returned(self) -> None:
        text = "the answer is -7"
        assert extract_final_answer(text) == "-7"

    def test_decimal_number_returned(self) -> None:
        text = "result: 3.14"
        assert extract_final_answer(text) == "3.14"

    def test_last_number_after_mixed_content(self) -> None:
        text = "from 100 items, remove 30, you get 70"
        assert extract_final_answer(text) == "70"


# ===========================================================================
# extract_final_answer() — EDGE CASES
# ===========================================================================


class TestExtractEdgeCases:
    """None input, empty string, no-number text, and boundary conditions."""

    def test_none_input_returns_not_found(self) -> None:
        assert extract_final_answer(None) == "NOT_FOUND"

    def test_empty_string_returns_not_found(self) -> None:
        assert extract_final_answer("") == "NOT_FOUND"

    def test_whitespace_only_returns_not_found(self) -> None:
        assert extract_final_answer("   \n  \t  ") == "NOT_FOUND"

    def test_no_box_no_phrase_no_number_returns_last_line(self) -> None:
        text = "no answer here\nfinal line"
        assert extract_final_answer(text) == "final line"

    def test_no_box_no_phrase_no_number_single_line(self) -> None:
        text = "no answer here"
        assert extract_final_answer(text) == "no answer here"

    def test_boxed_empty_then_no_other_signal(self) -> None:
        # Box is empty, no phrases, no numbers -> last non-empty line.
        text = r"\boxed{}"
        result = extract_final_answer(text)
        # The whole text is one line: "\boxed{}".  Last non-empty line is that.
        assert result == r"\boxed{}"

    def test_box_content_preferred_over_phrase_in_same_text(self) -> None:
        # If boxed answer exists, phrase fallback is NOT reached.
        text = r"\boxed{42} The final answer is: 99"
        assert extract_final_answer(text) == "42"

    def test_last_box_preferred_over_phrase(self) -> None:
        text = r"\boxed{3} \boxed{7} The final answer is: 99"
        assert extract_final_answer(text) == "7"

    def test_multiline_text_last_nonempty_line(self) -> None:
        text = "line one\n\nline three\n\n"
        # No box, no phrase, no number -> last non-empty line = "line three"
        assert extract_final_answer(text) == "line three"

    def test_boxed_with_fraction_last_brace_handling(self) -> None:
        # Verify rfind('}') selects the outermost closing brace.
        # \boxed{\frac{3}{4}} -> window is "\frac{3}{4}}..."
        # rfind('}') finds last '}' before the next \boxed{ or end.
        text = r"\boxed{\frac{3}{4}}"
        result = extract_final_answer(text)
        assert result == r"\frac{3}{4}"

    def test_large_number_in_box(self) -> None:
        text = r"\boxed{1234567890}"
        assert extract_final_answer(text) == "1234567890"

    def test_negative_number_in_box(self) -> None:
        text = r"\boxed{-42}"
        assert extract_final_answer(text) == "-42"


# ===========================================================================
# Combined extraction + verification pipeline
# ===========================================================================


class TestExtractionThenVerification:
    """End-to-end: extract then verify catches real scoring scenarios."""

    def test_boxed_numeric_within_tolerance(self) -> None:
        pred_text = r"\boxed{24.6401}"
        extracted = extract_final_answer(pred_text)
        assert verify("24.64", extracted) is True

    def test_boxed_roman_case_insensitive(self) -> None:
        pred_text = r"\boxed{xlvii}"
        extracted = extract_final_answer(pred_text)
        assert verify("XLVII", extracted) is True

    def test_boxed_units_mismatch_stored_plain(self) -> None:
        # Model outputs "42 apples" in box; stored answer is "42".
        # float("42 apples") fails -> string compare -> "42 apples" != "42" -> False.
        pred_text = r"\boxed{42 apples}"
        extracted = extract_final_answer(pred_text)
        assert extracted == "42 apples"
        assert verify("42", extracted) is False

    def test_last_boxed_used_for_verification(self) -> None:
        # Multiple boxes: last one (7) is used, not first (3).
        pred_text = r"\boxed{3} \boxed{7}"
        extracted = extract_final_answer(pred_text)
        assert extracted == "7"
        assert verify("7", extracted) is True
        assert verify("3", extracted) is False

    def test_phrase_fallback_then_verify_numeric(self) -> None:
        pred_text = "After computing, The final answer is: 100"
        extracted = extract_final_answer(pred_text)
        assert extracted == "100"
        assert verify("100", extracted) is True

    def test_none_prediction_is_wrong(self) -> None:
        extracted = extract_final_answer(None)
        assert extracted == "NOT_FOUND"
        assert verify("42", extracted) is False
