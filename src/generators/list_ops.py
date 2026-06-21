"""List / permutation transform puzzle generator.

Each puzzle presents a hidden structural transformation applied to a list of
integers; the model must infer the rule from examples and apply it to a query.

Supported rule families:
  - sort_asc:   sort in ascending order
  - sort_desc:  sort in descending order
  - reverse:    reverse the list order
  - rotate_left_k:  rotate elements left by k positions (cyclic)
  - rotate_right_k: rotate elements right by k positions (cyclic)
  - dedupe:     remove duplicate values, preserve first occurrence order

Lists contain 4–6 distinct integers drawn from [-30, 30].
Answers are comma-space-separated integer strings, e.g. "-3, 1, 7, 12".
This format is unambiguous and directly verifiable.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NUM_DEMO_PAIRS: int = 4
LIST_LEN_MIN: int = 4
LIST_LEN_MAX: int = 6
ELEM_MIN: int = -30
ELEM_MAX: int = 30

_HINT_LIST = (
    "In this realm, lists of numbers are transformed by a hidden structural rule."
)


# ── list formatting helpers ───────────────────────────────────────────────────

def _fmt(lst: list[int]) -> str:
    """Format a list as a comma-space-separated string."""
    return ", ".join(str(x) for x in lst)


# ── rule dataclass ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _ListRule:
    """A named list transformation rule.

    Attributes:
        name:        Short identifier.
        description: Human-readable explanation for the CoT.
        fn:          Callable that maps input list → output list.
    """

    name: str
    description: str
    fn: Callable[[list[int]], list[int]]


# ── rule builders ─────────────────────────────────────────────────────────────

def _rotate_left(lst: list[int], k: int) -> list[int]:
    """Rotate list left by k positions."""
    if not lst:
        return lst
    k = k % len(lst)
    return lst[k:] + lst[:k]


def _rotate_right(lst: list[int], k: int) -> list[int]:
    """Rotate list right by k positions."""
    if not lst:
        return lst
    k = k % len(lst)
    return lst[-k:] + lst[:-k] if k else lst[:]


def _dedupe(lst: list[int]) -> list[int]:
    """Remove duplicates, preserving first-occurrence order."""
    seen: set[int] = set()
    result: list[int] = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def _build_rules(rng: random.Random) -> list[_ListRule]:
    """Construct the candidate rule pool (rotation param sampled fresh)."""
    k = rng.randint(1, 3)
    return [
        _ListRule(
            name="sort_asc",
            description="sort the list in ascending (smallest to largest) order",
            fn=lambda lst: sorted(lst),
        ),
        _ListRule(
            name="sort_desc",
            description="sort the list in descending (largest to smallest) order",
            fn=lambda lst: sorted(lst, reverse=True),
        ),
        _ListRule(
            name="reverse",
            description="reverse the order of the list",
            fn=lambda lst: lst[::-1],
        ),
        _ListRule(
            name=f"rotate_left_{k}",
            description=f"rotate the list left by {k} position(s): the first {k} element(s) move to the end",
            fn=lambda lst, _k=k: _rotate_left(lst, _k),
        ),
        _ListRule(
            name=f"rotate_right_{k}",
            description=f"rotate the list right by {k} position(s): the last {k} element(s) move to the front",
            fn=lambda lst, _k=k: _rotate_right(lst, _k),
        ),
        _ListRule(
            name="dedupe",
            description="remove duplicate values, keeping only the first occurrence of each",
            fn=_dedupe,
        ),
    ]


# ── list sampler ──────────────────────────────────────────────────────────────

def _sample_list(rng: random.Random, length: int, with_duplicates: bool = False) -> list[int]:
    """Sample a list of integers.

    Args:
        rng:             Random generator.
        length:          Number of elements.
        with_duplicates: If True, allow repeated values (for dedupe rule).

    Returns:
        List of integers drawn from [ELEM_MIN, ELEM_MAX].
    """
    if with_duplicates:
        pool = rng.sample(range(ELEM_MIN, ELEM_MAX + 1), length - 1)
        dup_idx = rng.randint(0, length - 2)
        pool.append(pool[dup_idx])
        rng.shuffle(pool)
        return pool
    return rng.sample(range(ELEM_MIN, ELEM_MAX + 1), length)


# ── CoT builder ───────────────────────────────────────────────────────────────

def _build_cot(
    rule: _ListRule,
    demo_inputs: list[list[int]],
    demo_outputs: list[list[int]],
    query_input: list[int],
    answer_list: list[int],
) -> str:
    """Build a gold chain-of-thought for a list-transform puzzle.

    Args:
        rule:         The chosen transformation rule.
        demo_inputs:  Raw input lists for each demo pair.
        demo_outputs: Transformed output lists for each demo pair.
        query_input:  The query input list.
        answer_list:  The correct output list.

    Returns:
        Full CoT string ending in ``\\boxed{answer}``.
    """
    answer = _fmt(answer_list)
    lines: list[str] = []
    lines.append("Let me work through this list transformation puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs), start=1):
        lines.append(f"  Example {i}: {_fmt(inp)}  →  {_fmt(out)}")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append(f"The transformation is: {rule.description}.")

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs), start=1):
        predicted = rule.fn(inp)
        predicted_str = _fmt(predicted)
        out_str = _fmt(out)
        status = "PASS" if predicted_str == out_str else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: apply rule to {_fmt(inp)} → {predicted_str} "
            f"(expected {out_str}) [{status}]"
        )
    if not all_pass:
        raise ValueError(
            f"[list_ops] Rule '{rule.name}' failed to reproduce its own demo pairs."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(f"  Query input: {_fmt(query_input)}")
    lines.append(f"  Apply '{rule.name}' → {answer}")

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


# ── public entry-point ────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n list-transform puzzle Examples.

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

        needs_duplicates = rule.name == "dedupe"
        list_len = rng.randint(LIST_LEN_MIN, LIST_LEN_MAX)

        # Generate NUM_DEMO_PAIRS + 1 input lists (last is query)
        all_inputs: list[list[int]] = [
            _sample_list(rng, list_len, with_duplicates=needs_duplicates)
            for _ in range(NUM_DEMO_PAIRS + 1)
        ]
        demo_inputs = all_inputs[:NUM_DEMO_PAIRS]
        query_input = all_inputs[NUM_DEMO_PAIRS]

        demo_outputs = [rule.fn(inp) for inp in demo_inputs]
        answer_list = rule.fn(query_input)
        answer = _fmt(answer_list)

        # Prompt: show list as bare comma-separated string (no brackets) so the
        # surface form matches Example.answer exactly.  A model trained on these
        # demos will emit \boxed{-3, 1, 7, 12} which verify() accepts against
        # the stored gold "-3, 1, 7, 12" via case-insensitive string equality.
        demo_pairs: list[tuple[str, str]] = [
            (_fmt(inp), _fmt(out))
            for inp, out in zip(demo_inputs, demo_outputs)
        ]
        prompt = format_wonderland_prompt(
            pairs=demo_pairs,
            query_input=_fmt(query_input),
            extra_hint=_HINT_LIST,
        )

        gold_cot = _build_cot(rule, demo_inputs, demo_outputs, query_input, answer_list)

        expected_box = f"\\boxed{{{answer}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[list_ops] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=answer,
                domain="list_ops",
                gold_cot=gold_cot,
            )
        )
        logger.debug("list_ops example %d: rule=%s answer=%r", idx, rule.name, answer)

    return examples
