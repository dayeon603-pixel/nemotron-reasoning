"""Tests for scripts/build_real_sft.py and src/solvers.__init__.route_and_solve.

Critical-path coverage:
  - route_and_solve on a synthetic prompt per family: answer == known_answer,
    trace ends with \\boxed{known_answer}.
  - The verify self-check in build_real_sft raises RuntimeError on corrupt data.
  - If data/raw/train.csv exists: run the assembler on a 200-row sample and
    assert (a) 100% records pass boxed==gold, (b) >=99% matched some family.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Final

import pytest

from src.eval.metric import extract_final_answer, verify
from src.solvers import route_and_solve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CSV: Final[Path] = Path("data/raw/train.csv")
_TRAIN_MISSING_REASON: str = f"data/raw/train.csv not found at {TRAIN_CSV}"

_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{([^{}]*)\}")

# One synthetic prompt per family, carefully matching each solver's expected
# prefix so matches() fires.

_GRAVITATIONAL_PROMPT: str = (
    "In Alice's Wonderland, the gravitational constant has been secretly changed. "
    "Here are some example observations:\n"
    "For t = 2.00s, distance = 20.00 m\n"
    "For t = 3.00s, distance = 45.00 m\n"
    "Now, determine the falling distance for t = 4.00s given d = 0.5*g*t^2."
)
# g = 10.0: d = 0.5 * 10 * 16 = 80.00
_GRAVITATIONAL_ANSWER: str = "80.00"

_UNIT_PROMPT: str = (
    "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
    "For example:\n"
    "10.00 m becomes 15.00\n"
    "20.00 m becomes 30.00\n"
    "Now, convert the following measurement: 5.00 m"
)
# k = 1.5: y = 1.5 * 5.00 = 7.50
_UNIT_ANSWER: str = "7.50"

_NUMERAL_PROMPT: str = (
    "In Alice's Wonderland, numbers are secretly converted into a different "
    "numeral system. Some examples are given below:\n"
    "1 -> I\n"
    "5 -> V\n"
    "10 -> X\n"
    "Now, write the number 42 in the Wonderland numeral system."
)
_NUMERAL_ANSWER: str = "XLII"

_ENCRYPT_PROMPT: str = (
    "In Alice's Wonderland, secret encryption rules are used on text. "
    "Here are some examples:\n"
    "dbu -> cat\n"
    "Now, decrypt the following text: dbu"
)
_ENCRYPT_ANSWER: str = "cat"

_BITMANIP_PROMPT: str = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
    "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
    "and possibly majority or choice functions.\n\n"
    "Here are some examples of input -> output:\n"
    "00000000 -> 11111111\n"
    "11111111 -> 00000000\n"
    "10101010 -> 01010101\n"
    "01010101 -> 10101010\n"
    "Now, determine the output for: 11000000"
)
# Bitwise NOT: 11000000 -> 00111111
_BITMANIP_ANSWER: str = "00111111"

_SYMBOL_PROMPT: str = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
    "Below are a few examples:\n"
    "12+34 = 46\n"
    "23+11 = 34\n"
    "Now, determine the result for: 15+20"
)
_SYMBOL_ANSWER: str = "35"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_boxed(text: str) -> str | None:
    """Return last non-empty \\boxed{} content or None."""
    matches = [m.group(1).strip() for m in _BOXED_RE.finditer(text) if m.group(1).strip()]
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Section 1: route_and_solve per family
# ---------------------------------------------------------------------------

class TestRouteAndSolvePerFamily:
    """route_and_solve must return (cot, known_answer) for each family."""

    @pytest.mark.parametrize("prompt,known_answer", [
        (_GRAVITATIONAL_PROMPT, _GRAVITATIONAL_ANSWER),
        (_UNIT_PROMPT, _UNIT_ANSWER),
        (_NUMERAL_PROMPT, _NUMERAL_ANSWER),
        (_ENCRYPT_PROMPT, _ENCRYPT_ANSWER),
        (_BITMANIP_PROMPT, _BITMANIP_ANSWER),
        (_SYMBOL_PROMPT, _SYMBOL_ANSWER),
    ])
    def test_returns_tuple_not_none(self, prompt: str, known_answer: str) -> None:
        result = route_and_solve(prompt, known_answer)
        assert result is not None, (
            f"route_and_solve returned None for known_answer={known_answer!r}. "
            f"Prompt head: {prompt[:80]!r}"
        )

    @pytest.mark.parametrize("prompt,known_answer", [
        (_GRAVITATIONAL_PROMPT, _GRAVITATIONAL_ANSWER),
        (_UNIT_PROMPT, _UNIT_ANSWER),
        (_NUMERAL_PROMPT, _NUMERAL_ANSWER),
        (_ENCRYPT_PROMPT, _ENCRYPT_ANSWER),
        (_BITMANIP_PROMPT, _BITMANIP_ANSWER),
        (_SYMBOL_PROMPT, _SYMBOL_ANSWER),
    ])
    def test_answer_equals_known_answer(self, prompt: str, known_answer: str) -> None:
        result = route_and_solve(prompt, known_answer)
        assert result is not None
        _, returned_answer = result
        assert returned_answer == known_answer, (
            f"Returned answer {returned_answer!r} != known_answer {known_answer!r}"
        )

    @pytest.mark.parametrize("prompt,known_answer", [
        (_GRAVITATIONAL_PROMPT, _GRAVITATIONAL_ANSWER),
        (_UNIT_PROMPT, _UNIT_ANSWER),
        (_NUMERAL_PROMPT, _NUMERAL_ANSWER),
        (_ENCRYPT_PROMPT, _ENCRYPT_ANSWER),
        (_BITMANIP_PROMPT, _BITMANIP_ANSWER),
        (_SYMBOL_PROMPT, _SYMBOL_ANSWER),
    ])
    def test_trace_ends_with_boxed_known_answer(
        self, prompt: str, known_answer: str
    ) -> None:
        result = route_and_solve(prompt, known_answer)
        assert result is not None
        cot, _ = result
        assert cot.endswith(f"\\boxed{{{known_answer}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known_answer}}}"
        )

    @pytest.mark.parametrize("prompt,known_answer", [
        (_GRAVITATIONAL_PROMPT, _GRAVITATIONAL_ANSWER),
        (_UNIT_PROMPT, _UNIT_ANSWER),
        (_NUMERAL_PROMPT, _NUMERAL_ANSWER),
        (_ENCRYPT_PROMPT, _ENCRYPT_ANSWER),
        (_BITMANIP_PROMPT, _BITMANIP_ANSWER),
        (_SYMBOL_PROMPT, _SYMBOL_ANSWER),
    ])
    def test_extracted_answer_verifies(self, prompt: str, known_answer: str) -> None:
        result = route_and_solve(prompt, known_answer)
        assert result is not None
        cot, _ = result
        extracted = extract_final_answer(cot)
        assert verify(known_answer, extracted), (
            f"verify failed: gold={known_answer!r} extracted={extracted!r} "
            f"trace_tail={cot[-80:]!r}"
        )

    def test_returns_none_for_unrecognised_prompt(self) -> None:
        """A prompt matching no family must return None."""
        unknown = "What is 2 + 2?"
        result = route_and_solve(unknown, "4")
        assert result is None, f"Expected None, got {result!r}"


# ---------------------------------------------------------------------------
# Section 2: verify self-check in build_real_sft
# ---------------------------------------------------------------------------

class TestBuildRealSftVerifyCheck:
    """build_real_sft's verify self-check raises RuntimeError on corrupt data."""

    def _make_csv(self, rows: list[dict[str, str]]) -> Path:
        """Write a minimal CSV to a temp file and return its Path."""
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "prompt", "answer"])
            writer.writeheader()
            writer.writerows(rows)
        return tmp

    def test_clean_data_does_not_raise(self) -> None:
        """build_real_sft completes without error when all answers are correct."""
        from scripts.build_real_sft import build_real_sft

        rows = [
            {
                "id": "grav_0001",
                "prompt": _GRAVITATIONAL_PROMPT,
                "answer": _GRAVITATIONAL_ANSWER,
            },
            {
                "id": "numeral_0001",
                "prompt": _NUMERAL_PROMPT,
                "answer": _NUMERAL_ANSWER,
            },
        ]
        csv_path = self._make_csv(rows)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out = Path(tmp.name)

        try:
            stats = build_real_sft(train_csv=csv_path, output=out)
            assert stats["total"] == 2
            assert stats["verify_failures"] == 0
        finally:
            csv_path.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    def test_corrupt_trace_triggers_runtime_error(self) -> None:
        """If a solver returns a wrong boxed value, RuntimeError must be raised."""
        import unittest.mock as mock
        from scripts.build_real_sft import build_real_sft

        # Inject a prompt that our solvers recognise but whose answer we
        # intentionally swap so the boxed value won't match the gold.
        rows = [
            {
                "id": "corrupt_0001",
                "prompt": _GRAVITATIONAL_PROMPT,
                "answer": "WRONG_GOLD_999",  # No solver will produce this
            },
        ]
        csv_path = self._make_csv(rows)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out = Path(tmp.name)

        try:
            # route_and_solve will build a scaffold CoT with \boxed{WRONG_GOLD_999}
            # so it will actually pass — we need to mock the CoT to be genuinely bad.
            # Patch route_and_solve to return a CoT with the wrong boxed value.
            bad_cot = "Some reasoning.\n\\boxed{TOTALLY_WRONG}"
            with mock.patch(
                "scripts.build_real_sft.route_and_solve",
                return_value=(bad_cot, "WRONG_GOLD_999"),
            ):
                with pytest.raises(RuntimeError, match="verify self-check"):
                    build_real_sft(train_csv=csv_path, output=out)
        finally:
            csv_path.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    def test_output_jsonl_schema(self) -> None:
        """Every record in the output JSONL must have all required fields."""
        from scripts.build_real_sft import build_real_sft

        rows = [
            {
                "id": "grav_0002",
                "prompt": _GRAVITATIONAL_PROMPT,
                "answer": _GRAVITATIONAL_ANSWER,
            },
        ]
        csv_path = self._make_csv(rows)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            out = Path(tmp.name)

        try:
            build_real_sft(train_csv=csv_path, output=out)
            records = [json.loads(line) for line in out.read_text().splitlines() if line]
            assert len(records) == 1
            rec = records[0]
            for field in ("id", "prompt", "trace", "extracted_answer", "gold_answer"):
                assert field in rec, f"Missing field {field!r} in record {rec!r}"
            assert rec["gold_answer"] == _GRAVITATIONAL_ANSWER
            assert rec["id"] == "grav_0002"
        finally:
            csv_path.unlink(missing_ok=True)
            out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 3: 200-row smoke test on the real train.csv (skipped if absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataSample:
    """Run build_real_sft on a 200-row stratified sample and check invariants."""

    _SAMPLE_N: int = 200
    _FAMILY_MATCH_THRESHOLD: float = 0.99  # >= 99% of rows must match a family

    @pytest.fixture(scope="class")
    def sample_csv_and_output(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        """Write a 200-row sample CSV and run build_real_sft on it.

        Returns:
            Tuple of (sample_csv_path, output_jsonl_path).
        """
        from scripts.build_real_sft import build_real_sft

        # Read the full train.csv and take first 200 rows (ordered by CSV order,
        # which interleaves all families due to the competition's shuffle).
        all_rows: list[dict[str, str]] = []
        with TRAIN_CSV.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                all_rows.append(
                    {"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]}
                )
                if len(all_rows) >= self._SAMPLE_N:
                    break

        tmp = tmp_path_factory.mktemp("real_sample")
        sample_csv = tmp / "sample_train.csv"
        with sample_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "prompt", "answer"])
            writer.writeheader()
            writer.writerows(all_rows)

        out_jsonl = tmp / "real_sft_sample.jsonl"
        build_real_sft(train_csv=sample_csv, output=out_jsonl)
        return sample_csv, out_jsonl

    def test_record_count_equals_sample(
        self,
        sample_csv_and_output: tuple[Path, Path],
    ) -> None:
        _, out_jsonl = sample_csv_and_output
        records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l]
        assert len(records) == self._SAMPLE_N, (
            f"Expected {self._SAMPLE_N} records, got {len(records)}"
        )

    def test_100_percent_verify(
        self,
        sample_csv_and_output: tuple[Path, Path],
    ) -> None:
        """Every record's boxed value in the trace must equal the gold answer."""
        _, out_jsonl = sample_csv_and_output
        records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l]

        failures: list[dict[str, str]] = []
        for rec in records:
            extracted = extract_final_answer(rec["trace"])
            if not verify(rec["gold_answer"], extracted):
                failures.append({
                    "id": rec["id"],
                    "gold": rec["gold_answer"],
                    "extracted": extracted,
                    "trace_tail": rec["trace"][-100:],
                })

        assert not failures, (
            f"{len(failures)}/{len(records)} records failed verify: "
            f"first failure: {failures[0]}"
        )

    def test_family_match_rate_at_least_99_pct(
        self,
        sample_csv_and_output: tuple[Path, Path],
    ) -> None:
        """At most 1% of records should have family='__other__'."""
        from scripts.build_real_sft import _classify_family

        _, out_jsonl = sample_csv_and_output
        records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l]

        no_match = sum(
            1 for rec in records if _classify_family(rec["prompt"]) == "__other__"
        )
        match_rate = (len(records) - no_match) / len(records)
        assert match_rate >= self._FAMILY_MATCH_THRESHOLD, (
            f"Family match rate {match_rate:.2%} < {self._FAMILY_MATCH_THRESHOLD:.0%}. "
            f"{no_match} / {len(records)} rows matched '__other__'."
        )

    def test_all_records_have_required_fields(
        self,
        sample_csv_and_output: tuple[Path, Path],
    ) -> None:
        _, out_jsonl = sample_csv_and_output
        records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l]
        required = {"id", "prompt", "trace", "extracted_answer", "gold_answer"}
        for rec in records:
            missing = required - rec.keys()
            assert not missing, f"Record {rec.get('id')!r} missing fields: {missing}"

    def test_trace_ends_with_boxed_for_exact_families(
        self,
        sample_csv_and_output: tuple[Path, Path],
    ) -> None:
        """For gravitational / unit_conversion / numeral rows, the trace must
        end with some \\boxed{} value that verifies against the gold.

        Note: the boxed string may differ from gold_answer by a rounding
        boundary (e.g. derived=43.44 vs gold=43.43) while still passing
        metric.verify (1% relative tolerance).  We check verify(), not
        exact string equality.
        """
        from scripts.build_real_sft import _classify_family

        exact_families = {"gravitational", "unit_conversion", "numeral"}
        _, out_jsonl = sample_csv_and_output
        records = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l]

        failures: list[str] = []
        for rec in records:
            family = _classify_family(rec["prompt"])
            if family not in exact_families:
                continue
            # Trace must contain at least one \boxed{} value.
            if not _BOXED_RE.search(rec["trace"]):
                failures.append(
                    f"id={rec['id']} family={family}: no \\boxed{{}} in trace"
                )
                continue
            # The extracted boxed value must verify against the gold.
            extracted = extract_final_answer(rec["trace"])
            if not verify(rec["gold_answer"], extracted):
                failures.append(
                    f"id={rec['id']} family={family} gold={rec['gold_answer']!r} "
                    f"extracted={extracted!r} tail={rec['trace'][-60:]!r}"
                )

        assert not failures, (
            f"{len(failures)} exact-family records failed the boxed-verify check: "
            + "; ".join(failures[:3])
        )
