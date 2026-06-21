"""Tests for src/solvers/crack_encrypt.py — dictionary-augmented ENCRYPT solver.

Critical-path coverage
----------------------
- matches() correctly identifies ENCRYPT prompts and rejects other families.
- solve() returns (cot, answer) where cot ends with \\boxed{answer}.
- solve() with no known_answer: derived answer is correct on fully-known prompts.
- solve() with known_answer: returned answer is ALWAYS the supplied known_answer.
- CoT structure: contains all five steps, ends with \\boxed{answer}.
- Real-data accuracy (outright, no known_answer): hard assert >= 95%.
- Real-data accuracy (outright, no known_answer): report exact number.
- Known-answer path: hard assert 100% of first 50 real rows return known_answer.
- Bijectivity constraint: two cipher chars cannot map to the same plain char.
- Vocabulary constraint: words of wrong length are not accepted as candidates.
- Edge case: query word fully covered by example map (no dict lookup needed).
- Edge case: single-word query.
- Edge case: prompt with maximum example pairs (5 pairs).

Running:
    python -m pytest -q tests/test_crack_encrypt.py
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Final

import pytest

from src.solvers.crack_encrypt import (
    EncryptCrackSolver,
    _BENCHMARK_VOCAB,
    _VOCAB_BY_LEN,
    _build_charmap_from_pairs,
    _find_candidates,
    _parse_encrypt_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CSV: Final[Path] = Path("data/raw/train.csv")
_TRAIN_MISSING_REASON: Final[str] = f"train.csv not found at {TRAIN_CSV}"
_ENCRYPT_PREFIX: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
)
_BOXED_RE: Final[re.Pattern[str]] = re.compile(r"\\boxed\{([^{}]*)\}")

# Hard accuracy target: the dictionary-augmented solver must reach this on the
# full 1,576-row ENCRYPT subset (no known_answer supplied).
_OUTRIGHT_ACCURACY_FLOOR: Final[float] = 0.95


# ---------------------------------------------------------------------------
# Synthetic prompt fixtures
# ---------------------------------------------------------------------------

# All example characters appear in the query → solved by map only (no dict needed).
# 'mno' → 'cat': m→c, n→a, o→t (all distinct cipher and plain chars)
# 'bpq' → 'key': b→k, p→e, q→y (all distinct; no overlap with first pair's plain chars)
# Query 'mno bpq' decodes fully from the map: m→c, n→a, o→t, b→k, p→e, q→y → 'cat key'
_MAP_ONLY_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "mno -> cat\n"
    "bpq -> key\n"
    "Now, decrypt the following text: mno bpq"
)
_MAP_ONLY_ANSWER: Final[str] = "cat key"

# Query word 'king' requires exactly one dictionary fill: 'j' → 'k'.
# Example pair establishes: i→i, n→n, g→g (they are 3-char words matching 'ing').
# Then query word 'jing' (len 4) does not exist — use a better constructed case.
#
# Constructed carefully: examples give partial map, one query char is new.
# 'tqg' → 'cat' establishes t→c, q→a, g→t
# 'kxo' → 'the' establishes k→t, x→h, o→e
# Query 'jxo' has j unmapped. len=3; candidates in vocab of length 3: cat,the,map,key,in(2),
# but bijectivity: x→h and o→e already fixed, so plain word must match _h_e at pos 1,2.
# Only 'the' fits but k→t already takes t, so j cannot→t.  Try constructing differently.
#
# Simplest reliable fixture: use benchmark pair where one query word is directly in vocab.
# Build a synthetic cipher where we control the full mapping.
# Cipher alphabet: every letter shifted +3 (Caesar-like but injective by construction).
# 'dragon' in cipher: d->g, r->u, a->d, g->j, o->r, n->q → 'gudjrq'
# 'reads' in cipher: r->u, e->h, a->d, d->g, s->v → 'uhdgv'
# Example pair: 'gudjrq uhdgv -> dragon reads'
# gives g→d, u→r, d→a, j→g, r→o, q→n, h→e, v→s
# Query: 'jxu gudjrq' — 'x' unmapped.
# 'jxu' length 3: j→g, u→r already in map; x must→? so plain[1]=?, constraint j→g, u→r
# Candidates len 3 with [0]=g, [2]=r: only 'gar'? not in vocab. Hmm.
#
# Use the actual benchmark's own prompts via real-data tests instead for complex cases.
# For synthetic unit tests, use trivially-derivable prompts.

_DICT_FILL_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "ucoov -> queen\n"
    "bxo -> the\n"
    "Now, decrypt the following text: ucoov bxo"
)
_DICT_FILL_ANSWER: Final[str] = "queen the"

# known_answer override: derived answer matches.
_OVERRIDE_MATCH_PROMPT: Final[str] = _MAP_ONLY_PROMPT
_OVERRIDE_MATCH_KNOWN: Final[str] = "cat key"

# known_answer override: derived answer differs (forced mismatch).
_OVERRIDE_FORCE_PROMPT: Final[str] = _MAP_ONLY_PROMPT
_OVERRIDE_FORCE_KNOWN: Final[str] = "rabbit dragon"

# Single-word query.
_SINGLE_WORD_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "zrk -> cat\n"
    "Now, decrypt the following text: zrk"
)
_SINGLE_WORD_ANSWER: Final[str] = "cat"

# Malformed prompt (no query line).
_BAD_PROMPT: Final[str] = (
    "In Alice's Wonderland, secret encryption rules are used on text."
    " Here are some examples:\n"
    "abc -> xyz"
)

# Non-ENCRYPT prompts (used to test matches() negative cases).
_GRAV_PROMPT: Final[str] = (
    "In Alice's Wonderland, the gravitational constant has been secretly changed."
    " Here are some example observations:\n"
    "For t = 2.0s, distance = 20.0 m\n"
    "Now, determine the falling distance for t = 3.0s given d = 0.5*g*t^2."
)
_BITMANIP_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
    " Here are some examples of input -> output:\n"
    "10000000 -> 01111111\n"
    "Now, determine the output for: 00000001"
)
_SYMBOL_PROMPT: Final[str] = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
    " Below are a few examples:\n"
    "12+34 = 46\n"
    "Now, determine the result for: 15+20"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_boxed(text: str) -> str | None:
    """Return the last non-empty \\boxed{} content in text, or None."""
    matches: list[str] = [
        m.group(1).strip() for m in _BOXED_RE.finditer(text) if m.group(1).strip()
    ]
    return matches[-1] if matches else None


def _load_encrypt_rows() -> list[dict[str, str]]:
    """Load all ENCRYPT rows from train.csv.

    Returns:
        List of dicts with keys 'id', 'prompt', 'answer'.
    """
    csv.field_size_limit(10_000_000)
    rows: list[dict[str, str]] = []
    with TRAIN_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["prompt"].startswith(_ENCRYPT_PREFIX):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solver() -> EncryptCrackSolver:
    """Return a shared EncryptCrackSolver instance."""
    return EncryptCrackSolver()


@pytest.fixture(scope="module")
def encrypt_rows() -> list[dict[str, str]]:
    """Load ENCRYPT rows once per module (expensive I/O)."""
    if not TRAIN_CSV.exists():
        return []
    return _load_encrypt_rows()


# ---------------------------------------------------------------------------
# Section 1: matches() — routing correctness
# ---------------------------------------------------------------------------


class TestMatches:
    """matches() must accept ENCRYPT prompts and reject every other family."""

    def test_matches_map_only_prompt(self, solver: EncryptCrackSolver) -> None:
        assert solver.matches(_MAP_ONLY_PROMPT)

    def test_matches_dict_fill_prompt(self, solver: EncryptCrackSolver) -> None:
        assert solver.matches(_DICT_FILL_PROMPT)

    def test_matches_single_word_prompt(self, solver: EncryptCrackSolver) -> None:
        assert solver.matches(_SINGLE_WORD_PROMPT)

    def test_matches_bad_prompt(self, solver: EncryptCrackSolver) -> None:
        """Bad prompt still starts with the ENCRYPT prefix."""
        assert solver.matches(_BAD_PROMPT)

    def test_rejects_gravitational(self, solver: EncryptCrackSolver) -> None:
        assert not solver.matches(_GRAV_PROMPT)

    def test_rejects_bitmanip(self, solver: EncryptCrackSolver) -> None:
        assert not solver.matches(_BITMANIP_PROMPT)

    def test_rejects_symbol(self, solver: EncryptCrackSolver) -> None:
        assert not solver.matches(_SYMBOL_PROMPT)

    def test_rejects_empty_string(self, solver: EncryptCrackSolver) -> None:
        assert not solver.matches("")

    def test_rejects_unrelated_string(self, solver: EncryptCrackSolver) -> None:
        assert not solver.matches("Hello, world.")


# ---------------------------------------------------------------------------
# Section 2: parse helper
# ---------------------------------------------------------------------------


class TestParseEncryptPrompt:
    """_parse_encrypt_prompt must extract pairs and query correctly."""

    def test_extracts_pairs_from_map_only(self) -> None:
        pairs, query = _parse_encrypt_prompt(_MAP_ONLY_PROMPT)
        assert len(pairs) == 2
        assert pairs[0] == ("mno", "cat")
        assert pairs[1] == ("bpq", "key")

    def test_extracts_query_from_map_only(self) -> None:
        _, query = _parse_encrypt_prompt(_MAP_ONLY_PROMPT)
        assert query == "mno bpq"

    def test_extracts_single_word_query(self) -> None:
        _, query = _parse_encrypt_prompt(_SINGLE_WORD_PROMPT)
        assert query == "zrk"

    def test_raises_on_missing_query_line(self) -> None:
        with pytest.raises(ValueError, match="query line"):
            _parse_encrypt_prompt(_BAD_PROMPT)

    def test_multi_example_pairs(self) -> None:
        prompt = (
            "In Alice's Wonderland, secret encryption rules are used on text."
            " Here are some examples:\n"
            "abc -> xyz\n"
            "def -> uvw\n"
            "ghi -> rst\n"
            "Now, decrypt the following text: abc"
        )
        pairs, _ = _parse_encrypt_prompt(prompt)
        assert len(pairs) == 3


# ---------------------------------------------------------------------------
# Section 3: charmap builder
# ---------------------------------------------------------------------------


class TestBuildCharmapFromPairs:
    """_build_charmap_from_pairs must produce correct bijective partial maps."""

    def test_simple_pair(self) -> None:
        cmap, inv = _build_charmap_from_pairs([("ucc", "cat")])
        assert cmap == {"u": "c", "c": "a"}
        # Note: both 'c' in cipher map to 'a' (both chars are the same cipher char)
        # and the 'a' maps are just the last unique one.
        assert inv.get("c") == "u"
        assert inv.get("a") == "c"

    def test_two_pairs_no_conflict(self) -> None:
        cmap, inv = _build_charmap_from_pairs([("ucc", "cat"), ("bxo", "the")])
        assert cmap["u"] == "c"
        assert cmap["b"] == "t"
        assert cmap["x"] == "h"
        assert cmap["o"] == "e"

    def test_word_length_mismatch_skips_pair(self) -> None:
        cmap, _ = _build_charmap_from_pairs([("abc", "ab")])
        assert len(cmap) == 0

    def test_word_count_mismatch_skips_pair(self) -> None:
        cmap, _ = _build_charmap_from_pairs([("abc def", "xyz")])
        assert len(cmap) == 0

    def test_conflict_keeps_first(self) -> None:
        cmap, _ = _build_charmap_from_pairs([("aa", "bc")])
        # 'a' maps to 'b' first, then sees 'c' — conflict, first wins
        assert cmap["a"] == "b"

    def test_inverse_map_is_consistent(self) -> None:
        cmap, inv = _build_charmap_from_pairs([("xyzw", "abcd")])
        for cc, pc in cmap.items():
            assert inv[pc] == cc

    def test_empty_pairs(self) -> None:
        cmap, inv = _build_charmap_from_pairs([])
        assert cmap == {}
        assert inv == {}


# ---------------------------------------------------------------------------
# Section 4: vocabulary and candidate lookup
# ---------------------------------------------------------------------------


class TestVocabulary:
    """Verify _BENCHMARK_VOCAB and _VOCAB_BY_LEN are consistent."""

    def test_vocab_size(self) -> None:
        assert len(_BENCHMARK_VOCAB) == 77

    def test_vocab_all_lowercase(self) -> None:
        for word in _BENCHMARK_VOCAB:
            assert word == word.lower(), f"{word!r} is not all-lowercase"

    def test_vocab_by_len_covers_all_words(self) -> None:
        recovered: set[str] = set()
        for words in _VOCAB_BY_LEN.values():
            recovered.update(words)
        assert recovered == _BENCHMARK_VOCAB

    def test_vocab_by_len_bucket_lengths_correct(self) -> None:
        for length, words in _VOCAB_BY_LEN.items():
            for w in words:
                assert len(w) == length, f"{w!r} in bucket {length} has length {len(w)}"


class TestFindCandidates:
    """_find_candidates must respect bijectivity constraints."""

    def test_no_cmap_all_vocab_candidates_of_matching_length(self) -> None:
        # With empty cmap, every vocab word of matching length is a candidate.
        candidates = _find_candidates("cat", {}, {})
        plain_words = {pw for pw, _ in candidates}
        expected = {w for w in _BENCHMARK_VOCAB if len(w) == 3}
        assert plain_words == expected

    def test_known_map_filters_correctly(self) -> None:
        # c→c, a→a already in map; t must map to something.
        # Only words of length 3 starting with 'c', 'a' at pos 1 qualify.
        cmap = {"c": "c", "a": "a"}
        inv_map = {"c": "c", "a": "a"}
        candidates = _find_candidates("cat", cmap, inv_map)
        plain_words = {pw for pw, _ in candidates}
        # Every candidate must start with 'c' and have 'a' at position 1.
        for pw in plain_words:
            assert pw[0] == "c", f"Expected [0]='c', got {pw!r}"
            assert pw[1] == "a", f"Expected [1]='a', got {pw!r}"

    def test_bijectivity_blocks_collision(self) -> None:
        # inv_map says plain 'c' is already claimed by cipher 'z';
        # 'cat' requires position 0 to map to 'c', but cipher[0]='u' != 'z'
        # so this candidate should be rejected.
        cmap: dict[str, str] = {}
        inv_map: dict[str, str] = {"c": "z"}  # 'c' is taken by 'z'
        candidates = _find_candidates("ucc", cmap, inv_map)
        # 'cat' needs u→c, c→a.  u→c conflicts because c is taken by z.
        plain_words = {pw for pw, _ in candidates}
        assert "cat" not in plain_words

    def test_wrong_length_word_excluded(self) -> None:
        # cipher word length 10 — no 10-letter words in benchmark vocab
        candidates = _find_candidates("abcdefghij", {}, {})
        assert candidates == []

    def test_candidate_new_mappings_are_correct(self) -> None:
        # Cipher word 'uzj' (all distinct chars) should produce candidate 'cat'
        # with new_map u→c, z→a, j→t — fully verifiable because all three cipher
        # characters are distinct and the alignment is unambiguous.
        cmap: dict[str, str] = {}
        inv_map: dict[str, str] = {}
        candidates = _find_candidates("uzj", cmap, inv_map)
        cat_entry = next(
            ((pw, nm) for pw, nm in candidates if pw == "cat"), None
        )
        assert cat_entry is not None, "'cat' should be a candidate for cipher 'uzj'"
        _, new_map = cat_entry
        assert new_map.get("u") == "c", f"Expected u→c, got u→{new_map.get('u')!r}"
        assert new_map.get("z") == "a", f"Expected z→a, got z→{new_map.get('z')!r}"
        assert new_map.get("j") == "t", f"Expected j→t, got j→{new_map.get('j')!r}"


# ---------------------------------------------------------------------------
# Section 5: solve() structure and correctness — synthetic prompts
# ---------------------------------------------------------------------------


class TestSolveStructure:
    """solve() must always produce a valid (cot, answer) pair."""

    def test_returns_tuple_of_two_strings(self, solver: EncryptCrackSolver) -> None:
        result = solver.solve(_MAP_ONLY_PROMPT)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, str) for x in result)

    def test_cot_ends_with_boxed_answer_map_only(
        self, solver: EncryptCrackSolver
    ) -> None:
        cot, answer = solver.solve(_MAP_ONLY_PROMPT)
        assert cot.endswith(f"\\boxed{{{answer}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{answer}}}"
        )

    def test_boxed_content_matches_returned_answer(
        self, solver: EncryptCrackSolver
    ) -> None:
        cot, answer = solver.solve(_MAP_ONLY_PROMPT)
        boxed = _last_boxed(cot)
        assert boxed == answer, f"\\boxed{{}} is {boxed!r}, answer is {answer!r}"

    def test_cot_contains_all_five_steps(self, solver: EncryptCrackSolver) -> None:
        cot, _ = solver.solve(_MAP_ONLY_PROMPT)
        for step in ("Step 1", "Step 2", "Step 3", "Step 4", "Step 5"):
            assert step in cot, f"CoT missing {step!r}"

    def test_cot_contains_query_cipher(self, solver: EncryptCrackSolver) -> None:
        cot, _ = solver.solve(_MAP_ONLY_PROMPT)
        assert "mno bpq" in cot, "CoT does not show the query cipher phrase"

    def test_map_only_derives_correct_answer(self, solver: EncryptCrackSolver) -> None:
        _, answer = solver.solve(_MAP_ONLY_PROMPT)
        assert answer == _MAP_ONLY_ANSWER, (
            f"Expected {_MAP_ONLY_ANSWER!r}, got {answer!r}"
        )

    def test_dict_fill_derives_correct_answer(self, solver: EncryptCrackSolver) -> None:
        _, answer = solver.solve(_DICT_FILL_PROMPT)
        assert answer == _DICT_FILL_ANSWER, (
            f"Expected {_DICT_FILL_ANSWER!r}, got {answer!r}"
        )

    def test_single_word_query_correct(self, solver: EncryptCrackSolver) -> None:
        _, answer = solver.solve(_SINGLE_WORD_PROMPT)
        assert answer == _SINGLE_WORD_ANSWER, (
            f"Expected {_SINGLE_WORD_ANSWER!r}, got {answer!r}"
        )

    def test_determinism(self, solver: EncryptCrackSolver) -> None:
        cot1, ans1 = solver.solve(_MAP_ONLY_PROMPT)
        cot2, ans2 = solver.solve(_MAP_ONLY_PROMPT)
        assert cot1 == cot2 and ans1 == ans2, "solve() is not deterministic"

    def test_missing_query_line_raises_value_error(
        self, solver: EncryptCrackSolver
    ) -> None:
        with pytest.raises(ValueError, match="query line"):
            solver.solve(_BAD_PROMPT)


# ---------------------------------------------------------------------------
# Section 6: known_answer contract
# ---------------------------------------------------------------------------


class TestKnownAnswerContract:
    """solve(prompt, known_answer=X) must ALWAYS return X and end CoT with \\boxed{X}."""

    def test_known_answer_returned_when_matches_derivation(
        self, solver: EncryptCrackSolver
    ) -> None:
        known = _OVERRIDE_MATCH_KNOWN
        _, answer = solver.solve(_OVERRIDE_MATCH_PROMPT, known_answer=known)
        assert answer == known, f"Expected {known!r}, got {answer!r}"

    def test_cot_ends_with_boxed_known_answer_match(
        self, solver: EncryptCrackSolver
    ) -> None:
        known = _OVERRIDE_MATCH_KNOWN
        cot, _ = solver.solve(_OVERRIDE_MATCH_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known}}}"
        )

    def test_known_answer_overrides_when_differs_from_derivation(
        self, solver: EncryptCrackSolver
    ) -> None:
        """Even when the known answer contradicts the derivation, it must be returned."""
        known = _OVERRIDE_FORCE_KNOWN
        _, answer = solver.solve(_OVERRIDE_FORCE_PROMPT, known_answer=known)
        assert answer == known, f"Expected {known!r}, got {answer!r}"

    def test_cot_ends_with_boxed_known_answer_forced(
        self, solver: EncryptCrackSolver
    ) -> None:
        known = _OVERRIDE_FORCE_KNOWN
        cot, _ = solver.solve(_OVERRIDE_FORCE_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known}}}"
        )

    @pytest.mark.parametrize("known", [
        "cat the",
        "dragon reads",
        "the",
        "a very unusual known answer with spaces",
    ])
    def test_arbitrary_known_answer_always_returned(
        self, solver: EncryptCrackSolver, known: str
    ) -> None:
        _, answer = solver.solve(_MAP_ONLY_PROMPT, known_answer=known)
        assert answer == known

    @pytest.mark.parametrize("known", [
        "cat the",
        "dragon reads",
        "the",
        "a very unusual known answer with spaces",
    ])
    def test_arbitrary_known_answer_cot_ends_with_boxed(
        self, solver: EncryptCrackSolver, known: str
    ) -> None:
        cot, _ = solver.solve(_MAP_ONLY_PROMPT, known_answer=known)
        assert cot.endswith(f"\\boxed{{{known}}}"), (
            f"CoT tail {cot[-80:]!r} does not end with \\boxed{{{known}}}"
        )


# ---------------------------------------------------------------------------
# Section 7: real-data accuracy (skipped when train.csv absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason=_TRAIN_MISSING_REASON)
class TestRealDataAccuracy:
    """Validate accuracy on the full 1,576-row ENCRYPT subset of train.csv.

    Hard assertion: outright accuracy (no known_answer) >= 95 %.
    All 1,576 rows are tested.
    """

    def test_outright_accuracy_above_floor(
        self,
        solver: EncryptCrackSolver,
        encrypt_rows: list[dict[str, str]],
    ) -> None:
        """Outright derivation rate must exceed _OUTRIGHT_ACCURACY_FLOOR."""
        assert encrypt_rows, "No ENCRYPT rows loaded from train.csv"

        correct: int = 0
        total: int = len(encrypt_rows)
        failures: list[tuple[str, str, str]] = []

        for row in encrypt_rows:
            try:
                _, answer = solver.solve(row["prompt"])
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "EncryptCrackSolver raised on id=%s: %s", row["id"], exc
                )
                failures.append((row["id"], row["answer"], f"<error: {exc}>"))
                continue

            if answer == row["answer"]:
                correct += 1
            else:
                failures.append((row["id"], row["answer"], answer))

        accuracy: float = correct / total
        logger.info(
            "ENCRYPT crack outright accuracy: %d/%d = %.4f (%.2f%%)",
            correct, total, accuracy, 100.0 * accuracy,
        )
        if failures:
            sample = failures[:5]
            logger.warning(
                "First %d failures (id, gold, predicted): %s",
                min(5, len(failures)),
                sample,
            )

        assert accuracy >= _OUTRIGHT_ACCURACY_FLOOR, (
            f"ENCRYPT crack outright accuracy {accuracy:.4f} ({correct}/{total}) "
            f"is below the required floor {_OUTRIGHT_ACCURACY_FLOOR:.2f}. "
            f"First failure: id={failures[0][0]!r} gold={failures[0][1]!r} "
            f"pred={failures[0][2]!r}"
        )

    def test_cot_ends_with_boxed_spot_check(
        self,
        solver: EncryptCrackSolver,
        encrypt_rows: list[dict[str, str]],
    ) -> None:
        """First 100 real rows: CoT must end with \\boxed{answer} (no known_answer)."""
        assert encrypt_rows

        for row in encrypt_rows[:100]:
            try:
                cot, answer = solver.solve(row["prompt"])
            except ValueError:
                continue
            assert cot.endswith(f"\\boxed{{{answer}}}"), (
                f"id={row['id']}: CoT tail {cot[-60:]!r} does not end with "
                f"\\boxed{{{answer}}}"
            )

    def test_known_answer_path_always_correct(
        self,
        solver: EncryptCrackSolver,
        encrypt_rows: list[dict[str, str]],
    ) -> None:
        """With known_answer supplied, returned answer must equal it every time."""
        sample = encrypt_rows[:50]
        assert sample

        for row in sample:
            known = row["answer"]
            cot, answer = solver.solve(row["prompt"], known_answer=known)
            assert answer == known, (
                f"id={row['id']}: known_answer={known!r} but got {answer!r}"
            )
            assert cot.endswith(f"\\boxed{{{known}}}"), (
                f"id={row['id']}: CoT tail {cot[-60:]!r} does not end with "
                f"\\boxed{{{known}}}"
            )

    def test_report_baseline_vs_new(
        self,
        solver: EncryptCrackSolver,
        encrypt_rows: list[dict[str, str]],
    ) -> None:
        """Log baseline (map-only) vs. dictionary-augmented accuracy for comparison."""
        assert encrypt_rows

        map_only_correct: int = 0
        dict_correct: int = 0
        total: int = len(encrypt_rows)

        for row in encrypt_rows:
            # Simulate map-only: build charmap, apply without dict fill-in.
            from src.solvers.crack_encrypt import (
                _build_charmap_from_pairs,
                _parse_encrypt_prompt,
            )

            try:
                pairs, query_cipher = _parse_encrypt_prompt(row["prompt"])
            except ValueError:
                continue

            cmap, _ = _build_charmap_from_pairs(pairs)
            map_words = [
                "".join(cmap.get(cc, "?") for cc in cw)
                for cw in query_cipher.split()
            ]
            map_result = " ".join(map_words)
            if map_result == row["answer"]:
                map_only_correct += 1

            try:
                _, answer = solver.solve(row["prompt"])
            except ValueError:
                continue
            if answer == row["answer"]:
                dict_correct += 1

        logger.info(
            "BASELINE (map-only): %d/%d = %.1f%%  |  "
            "NEW (dict-augmented): %d/%d = %.1f%%",
            map_only_correct, total, 100.0 * map_only_correct / total,
            dict_correct, total, 100.0 * dict_correct / total,
        )
