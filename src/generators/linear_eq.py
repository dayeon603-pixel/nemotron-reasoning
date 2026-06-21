"""Linear equation puzzle generator.

The hidden rule is a linear function  f(x) = a*x + b  with integer a, b.
Demonstration examples show  input→output  pairs.  The model must infer a and b,
then evaluate f(query).

Constraints:
  - a ∈ {-5, …, -1} ∪ {1, …, 5}  (non-zero)
  - b ∈ {-10, …, 10}
  - Inputs x ∈ {-20, …, 20} \ {0}, all demo+query inputs are distinct
  - Answers are bare integers (no units, no words, no sign-padding)
"""

from __future__ import annotations

import logging
import random

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NUM_DEMO_PAIRS: int = 4

_HINT_LINEAR = (
    "In this magical land, each number is transformed according to a hidden "
    "arithmetic rule of the form  f(x) = a·x + b."
)

_A_CHOICES: list[int] = list(range(-5, 0)) + list(range(1, 6))
_B_RANGE: tuple[int, int] = (-10, 10)
_X_POOL: list[int] = list(range(-20, 0)) + list(range(1, 21))


# ── CoT builder ───────────────────────────────────────────────────────────────

def _build_cot(
    a: int,
    b: int,
    demo_pairs: list[tuple[int, int]],
    query_x: int,
    answer: int,
) -> str:
    """Build a gold chain-of-thought for a linear equation puzzle.

    Args:
        a:          Slope coefficient.
        b:          Intercept.
        demo_pairs: List of (x, f(x)) demo pairs.
        query_x:    The query input.
        answer:     The correct output f(query_x).

    Returns:
        Full CoT string ending in ``\\boxed{answer}``.
    """
    lines: list[str] = []
    lines.append("Let me work through this algebraic puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (x, y) in enumerate(demo_pairs, start=1):
        lines.append(f"  Example {i}: f({x}) = {y}")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append("I will find a and b such that f(x) = a·x + b.")
    if len(demo_pairs) >= 2:
        (x1, y1), (x2, y2) = demo_pairs[0], demo_pairs[1]
        if x2 != x1:
            slope_num = y2 - y1
            slope_den = x2 - x1
            lines.append(
                f"  Using examples 1 and 2: "
                f"a = (f({x2}) − f({x1})) / ({x2} − {x1}) "
                f"= ({y2} − {y1}) / ({x2} − {x1}) "
                f"= {slope_num} / {slope_den} = {a}"
            )
            lines.append(
                f"  Then b = f({x1}) − a·{x1} = {y1} − ({a})·({x1}) = {b}"
            )
    lines.append(f"  The rule is: f(x) = {a}·x + ({b}) = {a}x {'+ ' if b >= 0 else '− '}{abs(b)}")

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, (x, y) in enumerate(demo_pairs, start=1):
        predicted = a * x + b
        status = "PASS" if predicted == y else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: f({x}) = {a}·({x}) + ({b}) = {predicted} "
            f"(expected {y}) [{status}]"
        )
    if not all_pass:
        raise ValueError(
            f"[linear_eq] Rule f(x)={a}x+{b} failed to reproduce its own demo pairs."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(
        f"  f({query_x}) = {a}·({query_x}) + ({b}) = {a * query_x} + ({b}) = {answer}"
    )

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


# ── public entry-point ────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n linear-equation puzzle Examples.

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
        a = rng.choice(_A_CHOICES)
        b = rng.randint(*_B_RANGE)

        inputs = rng.sample(_X_POOL, NUM_DEMO_PAIRS + 1)
        demo_inputs = inputs[:NUM_DEMO_PAIRS]
        query_x = inputs[NUM_DEMO_PAIRS]

        demo_pairs: list[tuple[int, int]] = [(x, a * x + b) for x in demo_inputs]
        answer = a * query_x + b

        demo_pairs_str = [(str(x), str(y)) for x, y in demo_pairs]
        prompt = format_wonderland_prompt(
            pairs=demo_pairs_str,
            query_input=str(query_x),
            extra_hint=_HINT_LINEAR,
        )

        gold_cot = _build_cot(a, b, demo_pairs, query_x, answer)

        expected_box = f"\\boxed{{{answer}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[linear_eq] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=str(answer),
                domain="linear_eq",
                gold_cot=gold_cot,
            )
        )
        logger.debug(
            "linear_eq example %d: a=%d b=%d query_x=%d answer=%d",
            idx, a, b, query_x, answer,
        )

    return examples
