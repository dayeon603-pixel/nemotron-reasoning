"""Roman numeral puzzle generator.

Supported rule families:
  - int_to_roman: given a decimal integer, produce its Roman numeral string
  - roman_to_int: given a Roman numeral, produce its decimal integer
  - roman_add: add two Roman numerals, return result as Roman numeral
  - roman_subtract: subtract second from first (result always > 0)

Values are drawn from [1, 3999] (standard Roman numeral range).
Answers:
  - Roman numeral output  → exact uppercase Roman numeral string (e.g. "XIV")
  - Integer output        → bare decimal integer string (e.g. "14")
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Literal

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NUM_DEMO_PAIRS: int = 4
ROMAN_MIN: int = 1
ROMAN_MAX: int = 3999

_HINT_ROMAN = (
    "In this realm, numbers speak in the ancient tongue of the Roman Empire."
)

# ── Roman numeral codec ───────────────────────────────────────────────────────

_INT_TO_ROMAN_TABLE: list[tuple[int, str]] = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"),
    (1,    "I"),
]


def _to_roman(n: int) -> str:
    """Convert a positive integer in [1, 3999] to a Roman numeral string.

    Args:
        n: Integer to convert.

    Returns:
        Uppercase Roman numeral string.

    Raises:
        ValueError: If n is outside [1, 3999].
    """
    if not (ROMAN_MIN <= n <= ROMAN_MAX):
        raise ValueError(f"Roman numeral range is [{ROMAN_MIN}, {ROMAN_MAX}]; got {n}")
    result: list[str] = []
    for value, symbol in _INT_TO_ROMAN_TABLE:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)


def _from_roman(s: str) -> int:
    """Parse a Roman numeral string to an integer.

    Args:
        s: Uppercase Roman numeral string.

    Returns:
        Integer value.

    Raises:
        ValueError: If the string contains unrecognised characters.
    """
    roman_vals: dict[str, int] = {
        "I": 1, "V": 5, "X": 10, "L": 50,
        "C": 100, "D": 500, "M": 1000,
    }
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        if ch not in roman_vals:
            raise ValueError(f"Unknown Roman numeral character: {ch!r}")
        val = roman_vals[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


# ── rule registry ─────────────────────────────────────────────────────────────

AnswerType = Literal["roman", "integer"]


@dataclass(slots=True)
class _RomanRule:
    name: str
    answer_type: AnswerType
    description: str
    # fn takes a tuple of strings (the prompt "inputs") and returns the answer string
    # For binary rules the input string is "A + B" or "A - B"
    input_fn: Callable[[int, int], tuple[str, str]]
    # returns (display_input, display_output, answer)
    answer_fn: Callable[[int, int], str]


def _build_rules() -> list[_RomanRule]:
    return [
        _RomanRule(
            name="int_to_roman",
            answer_type="roman",
            description="convert the decimal integer to its Roman numeral equivalent",
            input_fn=lambda a, _b: (str(a), _to_roman(a)),
            answer_fn=lambda a, _b: _to_roman(a),
        ),
        _RomanRule(
            name="roman_to_int",
            answer_type="integer",
            description="convert the Roman numeral to its decimal integer equivalent",
            input_fn=lambda a, _b: (_to_roman(a), str(a)),
            answer_fn=lambda a, _b: str(a),
        ),
        _RomanRule(
            name="roman_add",
            answer_type="roman",
            description="add the two Roman numerals and express the result as a Roman numeral",
            input_fn=lambda a, b: (f"{_to_roman(a)} + {_to_roman(b)}", _to_roman(a + b)),
            answer_fn=lambda a, b: _to_roman(a + b),
        ),
        _RomanRule(
            name="roman_subtract",
            answer_type="roman",
            description=(
                "subtract the second Roman numeral from the first "
                "and express the result as a Roman numeral (result is always positive)"
            ),
            input_fn=lambda a, b: (f"{_to_roman(a)} - {_to_roman(b)}", _to_roman(a - b)),
            answer_fn=lambda a, b: _to_roman(a - b),
        ),
    ]


# ── CoT builder ───────────────────────────────────────────────────────────────

def _build_cot(
    rule: _RomanRule,
    demo_pairs: list[tuple[str, str]],
    query_display: str,
    answer: str,
    demo_ab: list[tuple[int, int]],
    query_a: int,
    query_b: int,
) -> str:
    """Build a gold chain-of-thought for a Roman numeral puzzle.

    Args:
        rule:         The chosen rule.
        demo_pairs:   List of (input_display, output_display) demo pairs.
        query_display: The display form of the query input.
        answer:        The correct answer string.
        demo_ab:       Raw integer pairs used to generate demos (for verify step).
        query_a:       First integer operand of the query.
        query_b:       Second integer operand (0 for unary rules).

    Returns:
        Full CoT string ending in ``\\boxed{answer}``.
    """
    lines: list[str] = []
    lines.append("Let me work through this Roman numeral puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (inp, out) in enumerate(demo_pairs, start=1):
        lines.append(f"  Example {i}: {inp}  →  {out}")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append(f"The pattern is: {rule.description}.")

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, ((a, b), (inp, out)) in enumerate(zip(demo_ab, demo_pairs), start=1):
        predicted = rule.answer_fn(a, b)
        status = "PASS" if predicted == out else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: apply rule to '{inp}' → '{predicted}' "
            f"(expected '{out}') [{status}]"
        )
    if not all_pass:
        raise ValueError(
            f"[roman] Rule '{rule.name}' failed to reproduce its own demo pairs."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(f"  Query input: {query_display}")
    lines.append(f"  Apply '{rule.name}' → {answer}")

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


# ── public entry-point ────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n Roman numeral puzzle Examples.

    Args:
        n:    Number of examples to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of length n fully-formed Example objects.

    Raises:
        ValueError: If any generated puzzle fails internal consistency check.
    """
    rng = random.Random(seed)
    rules = _build_rules()
    examples: list[Example] = []

    for idx in range(n):
        rule = rng.choice(rules)
        is_binary = rule.name in {"roman_add", "roman_subtract"}

        if is_binary:
            # For addition: ensure sum <= ROMAN_MAX
            # For subtraction: ensure a > b (result >= 1)
            a = rng.randint(ROMAN_MIN + 1, ROMAN_MAX // 2)
            if rule.name == "roman_add":
                b = rng.randint(ROMAN_MIN, ROMAN_MAX - a)
            else:
                b = rng.randint(ROMAN_MIN, a - 1)
        else:
            a = rng.randint(ROMAN_MIN, ROMAN_MAX)
            b = 0

        # Generate demo pairs
        demo_ab: list[tuple[int, int]] = []
        for _ in range(NUM_DEMO_PAIRS):
            if is_binary:
                da = rng.randint(ROMAN_MIN + 1, ROMAN_MAX // 2)
                if rule.name == "roman_add":
                    db = rng.randint(ROMAN_MIN, ROMAN_MAX - da)
                else:
                    db = rng.randint(ROMAN_MIN, da - 1)
            else:
                da = rng.randint(ROMAN_MIN, ROMAN_MAX)
                db = 0
            demo_ab.append((da, db))

        demo_pairs: list[tuple[str, str]] = [
            rule.input_fn(da, db) for da, db in demo_ab
        ]
        query_display, _ = rule.input_fn(a, b)
        answer = rule.answer_fn(a, b)

        prompt = format_wonderland_prompt(
            pairs=demo_pairs,
            query_input=query_display,
            extra_hint=_HINT_ROMAN,
        )

        gold_cot = _build_cot(
            rule, demo_pairs, query_display, answer, demo_ab, a, b
        )

        expected_box = f"\\boxed{{{answer}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[roman] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=answer,
                domain="roman",
                gold_cot=gold_cot,
            )
        )
        logger.debug("roman example %d: rule=%s answer=%s", idx, rule.name, answer)

    return examples
