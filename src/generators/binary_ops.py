"""Binary bit-manipulation puzzle generator.

Each puzzle presents N input→output pairs where the hidden rule is one of:
  - bitwise NOT (flip all bits)
  - left rotate by K positions
  - right rotate by K positions
  - XOR with a fixed mask
  - left shift by K (truncated, zero-filled right)
  - right shift by K (truncated, zero-filled left)

All values are zero-padded to a fixed BIT_WIDTH.  The answer is always a
zero-padded binary string (no spaces) of exactly BIT_WIDTH characters.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable

from src.generators.common import Example, format_wonderland_prompt

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
BIT_WIDTH: int = 8          # all puzzles use 8-bit values
NUM_DEMO_PAIRS: int = 4     # demonstration examples per puzzle
NUM_QUERY: int = 1          # single query per puzzle

# Wonderland-flavour hint injected per rule family
_HINT_BINARY = (
    "In this land, numbers wear binary costumes, "
    "and a secret bitwise spell transforms each one."
)

# ── rule definitions ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class _BinaryRule:
    name: str
    fn: Callable[[int, int], int]    # (value, param) -> result
    param: int                       # rule parameter (e.g., shift amount)
    description: str                 # human-readable rule name for CoT


def _rotate_left(value: int, k: int, width: int = BIT_WIDTH) -> int:
    k = k % width
    mask = (1 << width) - 1
    return ((value << k) | (value >> (width - k))) & mask


def _rotate_right(value: int, k: int, width: int = BIT_WIDTH) -> int:
    k = k % width
    mask = (1 << width) - 1
    return ((value >> k) | (value << (width - k))) & mask


def _bit_not(value: int, _: int, width: int = BIT_WIDTH) -> int:
    return ((1 << width) - 1) ^ value


def _xor_mask(value: int, mask: int) -> int:
    return value ^ mask


def _left_shift(value: int, k: int, width: int = BIT_WIDTH) -> int:
    return (value << k) & ((1 << width) - 1)


def _right_shift(value: int, k: int) -> int:
    return value >> k


_MIN_SHIFT: int = 1
_MAX_SHIFT: int = BIT_WIDTH - 1  # up to 7 for 8-bit; avoids trivial identity (k=0)
_MIN_BITS_SET_IN_OUTPUT: int = 1  # reject shift rules that produce all-zero on every demo


def _shift_param_non_degenerate(
    rng: random.Random,
    sample_values: list[int],
    shift_fn: Callable[[int, int], int],
) -> int:
    """Pick a shift amount in [_MIN_SHIFT, _MAX_SHIFT] that does not produce
    all-zero output for ALL sample_values.

    If every candidate k makes every sample_value shift to 0 (degenerate),
    falls back to k=1 (least aggressive shift).

    Args:
        rng:           Random generator.
        sample_values: Representative input values to check against.
        shift_fn:      The shift function (SHL or SHR).

    Returns:
        A shift amount k in [_MIN_SHIFT, _MAX_SHIFT].
    """
    candidates = list(range(_MIN_SHIFT, _MAX_SHIFT + 1))
    rng.shuffle(candidates)
    for k in candidates:
        # Accept k if at least one sample produces a non-zero output.
        if any(shift_fn(v, k) != 0 for v in sample_values):
            return k
    return _MIN_SHIFT  # fallback: least aggressive


def _build_rules(rng: random.Random) -> list[_BinaryRule]:
    """Construct the candidate rule pool with randomly sampled parameters.

    Shift amounts k are drawn from [1, BIT_WIDTH-1] (wider range than the
    former {1,2,3}) but validated against a sample of typical input values
    to avoid rules that produce all-zero output on every demo pair.
    """
    # Sample representative values for degenerate-output check.
    # These are not the actual puzzle values — just a quick sanity screen.
    _probe_values: list[int] = [rng.randint(1, (1 << BIT_WIDTH) - 1) for _ in range(8)]

    k_shift_shl = _shift_param_non_degenerate(rng, _probe_values, _left_shift)
    k_shift_shr = _shift_param_non_degenerate(rng, _probe_values, _right_shift)
    k_rot = rng.randint(_MIN_SHIFT, _MAX_SHIFT)
    mask = rng.randint(1, (1 << BIT_WIDTH) - 2)  # non-trivial mask

    return [
        _BinaryRule(
            name="NOT",
            fn=lambda v, _: _bit_not(v, _),
            param=0,
            description="flip every bit (bitwise NOT)",
        ),
        _BinaryRule(
            name=f"ROL-{k_rot}",
            fn=lambda v, p: _rotate_left(v, p),
            param=k_rot,
            description=f"rotate bits left by {k_rot} position(s)",
        ),
        _BinaryRule(
            name=f"ROR-{k_rot}",
            fn=lambda v, p: _rotate_right(v, p),
            param=k_rot,
            description=f"rotate bits right by {k_rot} position(s)",
        ),
        _BinaryRule(
            name=f"XOR-{mask:08b}",
            fn=_xor_mask,
            param=mask,
            description=f"XOR with fixed mask {mask:08b}",
        ),
        _BinaryRule(
            name=f"SHL-{k_shift_shl}",
            fn=lambda v, p: _left_shift(v, p),
            param=k_shift_shl,
            description=f"shift left by {k_shift_shl} bit(s), zero-fill right, truncate to 8 bits",
        ),
        _BinaryRule(
            name=f"SHR-{k_shift_shr}",
            fn=lambda v, p: _right_shift(v, p),
            param=k_shift_shr,
            description=f"shift right by {k_shift_shr} bit(s), zero-fill left",
        ),
    ]


# ── CoT builder ──────────────────────────────────────────────────────────────

def _build_cot(
    rule: _BinaryRule,
    demo_pairs: list[tuple[int, int]],
    query_val: int,
    answer_val: int,
) -> str:
    """Generate a gold chain-of-thought for a binary puzzle.

    The CoT:
      1. Restates the given examples as binary strings.
      2. States the induced rule.
      3. Verifies the rule reproduces every example.
      4. Applies the rule to the query.
      5. Emits one \\boxed{} with the zero-padded binary answer.

    Args:
        rule:       The hidden rule object.
        demo_pairs: List of (input_int, output_int) demonstration pairs.
        query_val:  The query input integer.
        answer_val: The correct output integer.

    Returns:
        Full CoT string ending in ``\\boxed{answer}``.
    """
    lines: list[str] = []
    lines.append("Let me work through this binary bit-manipulation puzzle step by step.")
    lines.append("")
    lines.append("**Step 1 — Restate the examples**")
    for i, (inp, out) in enumerate(demo_pairs, start=1):
        lines.append(f"  Example {i}: {inp:08b}  →  {out:08b}")

    lines.append("")
    lines.append("**Step 2 — Induce the rule**")
    lines.append(
        f"Looking at each transformation, the pattern is: {rule.description}."
    )

    lines.append("")
    lines.append("**Step 3 — Verify the rule on every example**")
    all_pass = True
    for i, (inp, out) in enumerate(demo_pairs, start=1):
        predicted = rule.fn(inp, rule.param)
        status = "PASS" if predicted == out else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  Example {i}: apply rule to {inp:08b} "
            f"→ {predicted:08b} (expected {out:08b}) [{status}]"
        )
    if not all_pass:
        # This should never happen; raise so generator catches it.
        raise ValueError(
            f"Rule '{rule.name}' failed to reproduce its own demo pairs — "
            "generator logic error."
        )
    lines.append("  All examples verified.")

    lines.append("")
    lines.append("**Step 4 — Apply rule to the query**")
    lines.append(f"  Query input: {query_val:08b}")
    lines.append(f"  Apply '{rule.description}' → {answer_val:08b}")

    lines.append("")
    lines.append("**Step 5 — Final answer**")
    lines.append(f"\\boxed{{{answer_val:08b}}}")

    return "\n".join(lines)


# ── public entry-point ───────────────────────────────────────────────────────

def generate(n: int, seed: int) -> list[Example]:
    """Generate n binary bit-manipulation puzzle Examples.

    Each example uses a fresh random rule and random input values so that
    puzzles are mutually independent.

    Args:
        n:    Number of examples to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of length n, each a fully-formed Example with prompt, answer,
        domain="binary_ops", and a verified gold_cot.

    Raises:
        ValueError: If any generated puzzle fails internal consistency check.
    """
    rng = random.Random(seed)
    examples: list[Example] = []

    for idx in range(n):
        rules = _build_rules(rng)
        rule = rng.choice(rules)

        # Sample distinct input values for demo + query
        all_inputs = rng.sample(range(1 << BIT_WIDTH), NUM_DEMO_PAIRS + NUM_QUERY)
        demo_inputs = all_inputs[:NUM_DEMO_PAIRS]
        query_int = all_inputs[NUM_DEMO_PAIRS]

        demo_pairs_int: list[tuple[int, int]] = [
            (v, rule.fn(v, rule.param)) for v in demo_inputs
        ]
        answer_int = rule.fn(query_int, rule.param)

        # Verify answer bit-width
        assert 0 <= answer_int < (1 << BIT_WIDTH), (
            f"Answer {answer_int} out of {BIT_WIDTH}-bit range for rule {rule.name}"
        )

        demo_pairs_str: list[tuple[str, str]] = [
            (f"{inp:08b}", f"{out:08b}") for inp, out in demo_pairs_int
        ]
        query_str = f"{query_int:08b}"
        answer_str = f"{answer_int:08b}"

        prompt = format_wonderland_prompt(
            pairs=demo_pairs_str,
            query_input=query_str,
            extra_hint=_HINT_BINARY,
        )

        gold_cot = _build_cot(rule, demo_pairs_int, query_int, answer_int)

        # Paranoia check: verify CoT ends with the correct boxed answer
        expected_box = f"\\boxed{{{answer_str}}}"
        if not gold_cot.endswith(expected_box):
            raise ValueError(
                f"[binary_ops] Example {idx}: gold_cot does not end with "
                f"'{expected_box}'. CoT tail: {gold_cot[-80:]!r}"
            )

        examples.append(
            Example(
                prompt=prompt,
                answer=answer_str,
                domain="binary_ops",
                gold_cot=gold_cot,
            )
        )
        logger.debug("binary_ops example %d: rule=%s answer=%s", idx, rule.name, answer_str)

    return examples
