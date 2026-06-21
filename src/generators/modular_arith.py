"""Modular arithmetic / clock-arithmetic puzzle generator.

Each puzzle presents input→output pairs where the hidden rule is a modular
arithmetic operation applied to one or two operands.  The model must infer
the modulus and operation from examples, then apply them to a query.

Supported rule families:
  - add_mod_n:    (a + b) mod n
  - sub_mod_n:    (a - b) mod n  (result always in [0, n-1])
  - mul_mod_n:    (a * b) mod n
  - pow_mod_n:    a^k mod n      (fixed exponent k, unary-looking)
  - linear_mod_n: (c * a + d) mod n  (affine map mod n)

Operand values are integers in [0, n-1] so the puzzle always presents
numbers within the clock face.  Answers are bare non-negative integer strings
in [0, n-1].  The modulus n is in {5, 6, 7, 8, 10, 11, 12, 13} so that
puzzles stay recognisable and the clock metaphor is natural.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NUM_DEMO_PAIRS: int = 4
_MODULI: list[int] = [5, 6, 7, 8, 10, 11, 12, 13]

_HINT_MOD = (
    "In this realm, all arithmetic happens on a circular clock that resets "
    "back to zero after reaching a certain number."
)


# ── rule dataclass ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _ModRule:
    """One modular arithmetic rule.

    Attributes:
        name:        Short identifier for logging.
        modulus:     The modulus n.
        is_binary:   True if the rule takes two operands (a, b).
        description: Human-readable explanation used in CoT.
        input_fn:    Given (a, b) produce the display input string and output string.
        answer_fn:   Given (a, b) produce the answer integer.
    """

    name: str
    modulus: int
    is_binary: bool
    description: str
    input_fn: Callable[[int, int], tuple[str, str]]
    answer_fn: Callable[[int, int], int]


# ── rule builders ─────────────────────────────────────────────────────────────

def _coprime_multiplier(rng: random.Random, n: int) -> int:
    """Return a random integer c in [2, n-1] with gcd(c, n) == 1.

    This ensures the linear map a -> (c*a + d) mod n is a bijection, so
    the rule is uniquely inducible from the demo pairs.

    Args:
        rng: Random generator.
        n:   Modulus (must be >= 3 so the range [2, n-1] is non-empty for
             some coprime; for n=2 the only option would be 1 which is
             excluded, but _MODULI starts at 5 so this is safe).

    Returns:
        A value c in [2, n-1] coprime with n.

    Raises:
        ValueError: If no coprime in [2, n-1] exists (cannot happen for n>=5).
    """
    candidates = [c for c in range(2, n) if math.gcd(c, n) == 1]
    if not candidates:
        raise ValueError(f"No coprime multiplier in [2, {n - 1}] for n={n}")
    return rng.choice(candidates)


def _build_rules(rng: random.Random, n: int) -> list[_ModRule]:
    """Construct candidate rules for modulus n."""
    k = rng.randint(2, 4)           # exponent for pow_mod
    # c must be coprime with n so linear_mod_n is a bijection and the rule is
    # uniquely inducible from the demo pairs (non-coprime c creates collisions).
    c = _coprime_multiplier(rng, n)
    d = rng.randint(0, n - 1)       # additive offset for linear_mod

    rules: list[_ModRule] = [
        _ModRule(
            name=f"add_mod_{n}",
            modulus=n,
            is_binary=True,
            description=(
                f"add the two numbers then take the result modulo {n} "
                f"(like clock arithmetic with {n} hours: (a + b) mod {n})"
            ),
            input_fn=lambda a, b, _n=n: (
                f"{a} + {b}",
                str((a + b) % _n),
            ),
            answer_fn=lambda a, b, _n=n: (a + b) % _n,
        ),
        _ModRule(
            name=f"sub_mod_{n}",
            modulus=n,
            is_binary=True,
            description=(
                f"subtract the second from the first then take the result modulo {n}: "
                f"(a − b) mod {n}"
            ),
            input_fn=lambda a, b, _n=n: (
                f"{a} - {b}",
                str((a - b) % _n),
            ),
            answer_fn=lambda a, b, _n=n: (a - b) % _n,
        ),
        _ModRule(
            name=f"mul_mod_{n}",
            modulus=n,
            is_binary=True,
            description=(
                f"multiply the two numbers then take the result modulo {n}: "
                f"(a × b) mod {n}"
            ),
            input_fn=lambda a, b, _n=n: (
                f"{a} × {b}",
                str((a * b) % _n),
            ),
            answer_fn=lambda a, b, _n=n: (a * b) % _n,
        ),
        _ModRule(
            name=f"pow{k}_mod_{n}",
            modulus=n,
            is_binary=False,
            description=(
                f"raise the number to the power {k} then take the result modulo {n}: "
                f"a^{k} mod {n}"
            ),
            input_fn=lambda a, _b, _n=n, _k=k: (
                str(a),
                str(pow(a, _k, _n)),
            ),
            answer_fn=lambda a, _b, _n=n, _k=k: pow(a, _k, _n),
        ),
        _ModRule(
            name=f"linear_mod_{n}(c={c},d={d})",
            modulus=n,
            is_binary=False,
            description=(
                f"apply the linear map ({c}·a + {d}) mod {n} to the number"
            ),
            input_fn=lambda a, _b, _n=n, _c=c, _d=d: (
                str(a),
                str((_c * a + _d) % _n),
            ),
            answer_fn=lambda a, _b, _n=n, _c=c, _d=d: (_c * a + _d) % _n,
        ),
    ]
    return rules


# ── operand sampler ───────────────────────────────────────────────────────────

def _sample_operands(
    rng: random.Random,
    n: int,
    count: int,
    is_binary: bool,
) -> list[tuple[int, int]]:
    """Sample count distinct (a, b) pairs from [0, n-1]^2 or [0, n-1] for unary.

    Args:
        rng:       Random generator.
        n:         Modulus.
        count:     Number of pairs needed.
        is_binary: If True sample a and b independently; else b=0.

    Returns:
        List of (a, b) tuples.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = count * 20
    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1) if is_binary else 0
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    if len(pairs) < count:
        # Fallback: allow repeats (highly unlikely but never crash)
        while len(pairs) < count:
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1) if is_binary else 0
            pairs.append((a, b))
    return pairs


# ── CoT builder ───────────────────────────────────────────────────────────────

def _build_cot(
    rule: _ModRule,
    demo_pairs: list[tuple[str, str]],
    demo_ab: list[tuple[int, int]],
    query_display: str,
    query_a: int,
    query_b: int,
    answer: int,
) -> str:
    """Build a gold chain-of-thought for a modular arithmetic puzzle.

    Args:
        rule:          The chosen modular rule.
        demo_pairs:    List of (input_display, output_display) for demos.
        demo_ab:       Raw (a, b) integer pairs for verification.
        query_display: Display string of the query input.
        query_a:       First operand of the query.
        query_b:       Second operand of the query (0 for unary rules).
        answer:        The correct answer integer.

    Returns:
        Full CoT string ending in ``\\boxed{answer}``.
    """
    n = rule.modulus
    lines: list[str] = []
    lines.append("Let me work through this modular arithmetic puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (inp, out) in enumerate(demo_pairs, start=1):
        lines.append(f"  Example {i}: {inp}  →  {out}")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append(f"The transformation is: {rule.description}.")
    lines.append(f"  Clock size (modulus): {n}.")

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, ((a, b), (inp, out)) in enumerate(zip(demo_ab, demo_pairs), start=1):
        predicted = rule.answer_fn(a, b)
        status = "PASS" if str(predicted) == out else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: apply rule to '{inp}' → {predicted} "
            f"(expected {out}) [{status}]"
        )
    if not all_pass:
        raise ValueError(
            f"[modular_arith] Rule '{rule.name}' failed to reproduce its own demo pairs."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(f"  Query input: {query_display}")
    lines.append(f"  Apply '{rule.name}': result = {answer}")

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


# ── public entry-point ────────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n modular arithmetic puzzle Examples.

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
        modulus = rng.choice(_MODULI)
        rules = _build_rules(rng, modulus)
        rule = rng.choice(rules)

        all_ab = _sample_operands(
            rng, modulus, NUM_DEMO_PAIRS + 1, rule.is_binary
        )
        demo_ab = all_ab[:NUM_DEMO_PAIRS]
        query_a, query_b = all_ab[NUM_DEMO_PAIRS]

        demo_pairs: list[tuple[str, str]] = [
            rule.input_fn(a, b) for a, b in demo_ab
        ]
        query_display, _ = rule.input_fn(query_a, query_b)
        answer_int = rule.answer_fn(query_a, query_b)
        answer = str(answer_int)

        prompt = format_wonderland_prompt(
            pairs=demo_pairs,
            query_input=query_display,
            extra_hint=_HINT_MOD,
        )

        gold_cot = _build_cot(
            rule, demo_pairs, demo_ab,
            query_display, query_a, query_b, answer_int,
        )

        expected_box = f"\\boxed{{{answer}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[modular_arith] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=answer,
                domain="modular_arith",
                gold_cot=gold_cot,
            )
        )
        logger.debug(
            "modular_arith example %d: rule=%s modulus=%d answer=%s",
            idx, rule.name, modulus, answer,
        )

    return examples
