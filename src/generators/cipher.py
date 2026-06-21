"""Text substitution / Caesar-style cipher puzzle generator.

Supported rule families:
  - Caesar shift (shift each letter by K, wrap A-Z / a-z separately)
  - ROT-13 (fixed Caesar k=13)
  - Atbash (A↔Z, B↔Y, ... preserving case)
  - Reverse-word (reverse each word, preserve spaces)
  - Character-swap (swap pairs: position 0↔1, 2↔3, ...; odd-length: last char stays)

Non-alpha characters pass through unchanged in all cipher rules.
Answers are exact ciphertext strings; the \\boxed{} contains the raw string.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass
from typing import Callable

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NUM_DEMO_PAIRS: int = 4

_HINT_CIPHER = (
    "In this realm, words and letters are disguised by an ancient secret cipher."
)

# Short word list used to build demo plaintexts (avoids gibberish)
_WORD_POOL: list[str] = [
    "apple", "brave", "chess", "dance", "eagle", "flame", "grace", "honey",
    "ivory", "jewel", "knave", "lunar", "magic", "noble", "orbit", "pearl",
    "queen", "river", "stone", "tower", "ultra", "vivid", "whirl", "xenon",
    "yacht", "zonal", "brook", "cedar", "drift", "ember",
]


# ── cipher implementations ────────────────────────────────────────────────────

def _caesar(text: str, k: int) -> str:
    result: list[str] = []
    for ch in text:
        if ch.isupper():
            result.append(chr((ord(ch) - ord("A") + k) % 26 + ord("A")))
        elif ch.islower():
            result.append(chr((ord(ch) - ord("a") + k) % 26 + ord("a")))
        else:
            result.append(ch)
    return "".join(result)


def _atbash(text: str) -> str:
    result: list[str] = []
    for ch in text:
        if ch.isupper():
            result.append(chr(ord("Z") - (ord(ch) - ord("A"))))
        elif ch.islower():
            result.append(chr(ord("z") - (ord(ch) - ord("a"))))
        else:
            result.append(ch)
    return "".join(result)


def _reverse_words(text: str) -> str:
    return " ".join(w[::-1] for w in text.split(" "))


def _swap_pairs(text: str) -> str:
    chars = list(text)
    for i in range(0, len(chars) - 1, 2):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


# ── rule registry ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _CipherRule:
    name: str
    fn: Callable[[str], str]
    description: str


def _build_rules(rng: random.Random) -> list[_CipherRule]:
    k = rng.randint(1, 12)
    return [
        _CipherRule(
            name=f"Caesar-{k}",
            fn=lambda t, _k=k: _caesar(t, _k),
            description=(
                f"shift each letter forward by {k} positions in the alphabet "
                f"(A→{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[k % 26]}, wrapping around), "
                "leave non-alpha unchanged"
            ),
        ),
        _CipherRule(
            name="ROT13",
            fn=lambda t: _caesar(t, 13),
            description=(
                "rotate each letter by 13 positions (ROT-13): A↔N, B↔O, etc., "
                "leave non-alpha unchanged"
            ),
        ),
        _CipherRule(
            name="Atbash",
            fn=_atbash,
            description=(
                "reverse the alphabet: A↔Z, B↔Y, C↔X, …, preserving case, "
                "leave non-alpha unchanged"
            ),
        ),
        _CipherRule(
            name="ReverseWords",
            fn=_reverse_words,
            description="reverse the characters within each word, preserve spaces",
        ),
        _CipherRule(
            name="SwapPairs",
            fn=_swap_pairs,
            description=(
                "swap adjacent character pairs (pos 0↔1, 2↔3, …); "
                "if the string length is odd the last character stays in place"
            ),
        ),
    ]


# ── plaintext sampler ─────────────────────────────────────────────────────────

def _sample_plaintext(rng: random.Random, word_count: int = 2) -> str:
    """Pick ``word_count`` words from the pool and join with a space."""
    return " ".join(rng.sample(_WORD_POOL, word_count))


# ── CoT builder ───────────────────────────────────────────────────────────────

def _build_cot(
    rule: _CipherRule,
    demo_pairs: list[tuple[str, str]],
    query_plain: str,
    answer_cipher: str,
) -> str:
    """Build a gold chain-of-thought for a cipher puzzle.

    Args:
        rule:          The chosen cipher rule.
        demo_pairs:    List of (plaintext, ciphertext) demo pairs.
        query_plain:   The query plaintext.
        answer_cipher: The expected ciphertext answer.

    Returns:
        Full CoT string ending in ``\\boxed{answer_cipher}``.
    """
    lines: list[str] = []
    lines.append("Let me work through this cipher puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (p, c) in enumerate(demo_pairs, start=1):
        lines.append(f"  Example {i}: '{p}'  →  '{c}'")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append(f"The transformation is: {rule.description}.")

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, (p, c) in enumerate(demo_pairs, start=1):
        predicted = rule.fn(p)
        status = "PASS" if predicted == c else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: apply rule to '{p}' → '{predicted}' "
            f"(expected '{c}') [{status}]"
        )
    if not all_pass:
        raise ValueError(
            f"[cipher] Rule '{rule.name}' failed to reproduce its own demo pairs."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(f"  Query input: '{query_plain}'")
    lines.append(f"  Apply '{rule.name}' → '{answer_cipher}'")

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer_cipher}}}")

    return "\n".join(lines)


# ── public entry-point ────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n cipher puzzle Examples.

    Args:
        n:    Number of examples to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of length n fully-formed Example objects.

    Raises:
        ValueError: If any generated puzzle fails internal consistency check.
    """
    rng = random.Random(seed)
    examples: list[Example] = []

    for idx in range(n):
        rules = _build_rules(rng)
        rule = rng.choice(rules)

        word_count = rng.randint(1, 3)
        demo_plains = [_sample_plaintext(rng, word_count) for _ in range(NUM_DEMO_PAIRS)]
        query_plain = _sample_plaintext(rng, word_count)

        demo_pairs = [(p, rule.fn(p)) for p in demo_plains]
        answer_cipher = rule.fn(query_plain)

        prompt = format_wonderland_prompt(
            pairs=demo_pairs,
            query_input=query_plain,
            extra_hint=_HINT_CIPHER,
        )

        gold_cot = _build_cot(rule, demo_pairs, query_plain, answer_cipher)

        expected_box = f"\\boxed{{{answer_cipher}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[cipher] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=answer_cipher,
                domain="cipher",
                gold_cot=gold_cot,
            )
        )
        logger.debug("cipher example %d: rule=%s", idx, rule.name)

    return examples
