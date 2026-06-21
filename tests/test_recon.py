"""Tests for the family-taxonomy recon."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.recon.taxonomy import (
    GENERATOR_DOMAINS,
    analyze_rows,
    classify_answer,
    classify_domain,
    template_signature,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_train.csv"

# ---------------------------------------------------------------------------
# classify_answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "answer,expected",
    [
        # Multi-char all-[01] strings are binary regardless of value.
        ("10010111", "binary"),
        ("10", "binary"),
        ("01", "binary"),
        # Single-char "0" and "1" are ints, NOT binary.
        # binary_ops always emits BIT_WIDTH (8) char strings; a 1-char answer
        # can only come from modular_arith / number_seq, so int is correct.
        ("1", "int"),
        ("0", "int"),
        # Standard int and float cases.
        ("38", "int"),
        ("-12", "int"),
        ("24.64", "float"),
        # Roman numerals.
        ("XXXVIII", "roman"),
        ("xlvii", "roman"),
        # Word / phrase.
        ("wizard", "word"),
        ("cat imagines book", "phrase"),
        # Edge cases.
        ("", "empty"),
        ("3/4", "other"),
        # Comma-separated list (list_ops answer format) -> "other".
        ("-3, 1, 7, 12", "other"),
    ],
)
def test_classify_answer(answer: str, expected: str) -> None:
    assert classify_answer(answer) == expected


# ---------------------------------------------------------------------------
# classify_domain — all 7 families + uncovered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        # ── original 4 ────────────────────────────────────────────────────────
        (
            "a secret bit manipulation rule transforms 8-bit binary numbers",
            "binary_ops",
        ),
        ("secret encryption rules are used on text", "cipher"),
        ("numbers are written using a roman numeral rule", "roman"),
        ("a hidden equation rule maps an input number", "linear_eq"),
        # ── 3 new families ────────────────────────────────────────────────────
        # number_seq: uses "magical sequence" from _HINT_SEQ
        (
            "In this realm, numbers follow a hidden pattern in a magical sequence. "
            "Each term is determined by a secret rule applied to its position. "
            "Examples:  1. Input: term 1  →  Output: 3",
            "number_seq",
        ),
        # list_ops: uses "lists of numbers are transformed by a hidden structural rule"
        (
            "In this realm, lists of numbers are transformed by a hidden structural rule. "
            "Examples:  1. Input: [3, -1, 7]  →  Output: [-1, 3, 7]",
            "list_ops",
        ),
        # modular_arith: uses "circular clock that resets back to zero"
        (
            "In this realm, all arithmetic happens on a circular clock that resets "
            "back to zero after reaching a certain number. "
            "Examples:  1. Input: 3 + 4  →  Output: 0",
            "modular_arith",
        ),
        # ── uncovered ─────────────────────────────────────────────────────────
        ("In Wonderland a totally novel widget rule applies", "uncovered"),
    ],
)
def test_classify_domain(prompt: str, expected: str) -> None:
    assert classify_domain(prompt) == expected


# ---------------------------------------------------------------------------
# Cross-routing: make sure new keywords don't mis-route old families
# ---------------------------------------------------------------------------

def test_binary_ops_not_rerouted_by_new_families() -> None:
    """binary_ops prompt must not be caught by modular_arith / list_ops / number_seq."""
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms "
        "8-bit binary numbers. The transformation maps inputs to outputs."
    )
    assert classify_domain(prompt) == "binary_ops"


def test_linear_eq_not_rerouted() -> None:
    """linear_eq prompt (no clock/list/sequence keywords) stays linear_eq."""
    prompt = (
        "In this magical land, each number is transformed according to a hidden "
        "arithmetic rule of the form  f(x) = a·x + b. "
        "Solve for the query input."
    )
    assert classify_domain(prompt) == "linear_eq"


def test_cipher_not_rerouted() -> None:
    """cipher prompt stays cipher."""
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples: apple -> nccyr."
    )
    assert classify_domain(prompt) == "cipher"


# ---------------------------------------------------------------------------
# template_signature
# ---------------------------------------------------------------------------

def test_template_signature_collapses_values() -> None:
    a = "Examples: 00110100 -> 11001011. Query input: 01101000"
    b = "Examples: 11111111 -> 00000000. Query input: 10111100"
    assert template_signature(a) == template_signature(b)


# ---------------------------------------------------------------------------
# analyze_rows with the original 4-family fixture
# ---------------------------------------------------------------------------

def test_analyze_fixture() -> None:
    with FIXTURE.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    report = analyze_rows(rows)
    assert report.total == 7
    # binary_ops (2), cipher (2), roman (1), linear_eq (2)
    assert report.domains["binary_ops"].count == 2
    assert report.domains["cipher"].count == 2
    assert report.domains["linear_eq"].count == 2
    assert report.domains["roman"].count == 1
    # binary answers detected at 8-bit width
    assert report.domains["binary_ops"].binary_widths[8] == 2
    # no uncovered family in this clean fixture
    assert "uncovered" not in report.domains


# ---------------------------------------------------------------------------
# analyze_rows with synthetic rows covering all 7 families
# ---------------------------------------------------------------------------

def _make_row(
    domain_prompt: str,
    answer: str,
    row_id: str = "test",
) -> dict[str, str]:
    return {"id": row_id, "prompt": domain_prompt, "answer": answer}


def test_analyze_all_7_families() -> None:
    """All 7 generator domains must be correctly classified from synthetic rows."""
    rows = [
        _make_row(
            "a secret bit manipulation rule transforms 8-bit binary numbers",
            "10010111",
            "id_bin",
        ),
        _make_row(
            "secret encryption rules are used on text",
            "wizard",
            "id_cipher",
        ),
        _make_row(
            "numbers are written using a roman numeral rule",
            "XXXVIII",
            "id_roman",
        ),
        _make_row(
            "a hidden equation rule maps an input number",
            "23",
            "id_lineq",
        ),
        _make_row(
            "numbers follow a hidden pattern in a magical sequence",
            "89",
            "id_numseq",
        ),
        _make_row(
            "lists of numbers are transformed by a hidden structural rule",
            "-3, 1, 7",
            "id_listops",
        ),
        _make_row(
            "all arithmetic happens on a circular clock that resets back to zero",
            "4",
            "id_modarith",
        ),
    ]
    report = analyze_rows(rows)
    assert report.total == 7
    for dom in GENERATOR_DOMAINS:
        assert dom in report.domains, f"domain {dom!r} not found in report"
        assert report.domains[dom].count == 1, (
            f"expected 1 row for {dom!r}, got {report.domains[dom].count}"
        )
    assert "uncovered" not in report.domains


def test_single_char_01_classified_as_int_not_binary() -> None:
    """Rows with answer '0' or '1' from modular_arith must not bleed into binary counts."""
    rows = [
        _make_row(
            "all arithmetic happens on a circular clock that resets back to zero",
            "0",
            "id_mod0",
        ),
        _make_row(
            "all arithmetic happens on a circular clock that resets back to zero",
            "1",
            "id_mod1",
        ),
    ]
    report = analyze_rows(rows)
    st = report.domains["modular_arith"]
    # Both answers must be classified as "int", not "binary"
    assert st.answer_formats.get("int", 0) == 2, (
        f"Expected 2 int answers, got: {dict(st.answer_formats)}"
    )
    assert st.answer_formats.get("binary", 0) == 0, (
        f"'0'/'1' answers were mis-classified as binary: {dict(st.answer_formats)}"
    )
    # No binary_widths should have been recorded
    assert not st.binary_widths, (
        f"binary_widths should be empty for modular_arith '0'/'1' answers: "
        f"{dict(st.binary_widths)}"
    )


# ---------------------------------------------------------------------------
# _detect_gaps surface check
# ---------------------------------------------------------------------------

def test_detect_gaps_reports_coverage_summary() -> None:
    """gaps list must always include a COVERAGE SUMMARY line."""
    with FIXTURE.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    report = analyze_rows(rows)
    summary_msgs = [g for g in report.gaps if g.startswith("COVERAGE SUMMARY")]
    assert len(summary_msgs) == 1
    assert "covered" in summary_msgs[0]
    assert "uncovered" in summary_msgs[0]


def test_detect_gaps_uncovered_surfaced() -> None:
    """Rows that match no domain must produce an UNCOVERED FAMILY gap."""
    rows = [
        _make_row("completely unrecognised puzzle format", "42", "id_unk"),
    ]
    report = analyze_rows(rows)
    uncovered_msgs = [g for g in report.gaps if g.startswith("UNCOVERED FAMILY")]
    assert len(uncovered_msgs) == 1
    assert "1 rows" in uncovered_msgs[0]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_analyze_empty_raises() -> None:
    with pytest.raises(ValueError):
        analyze_rows([])
