"""Shared dataclass and prompt-formatting utilities for all generator families."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Example",
    "Domain",
    "format_wonderland_prompt",
]

Domain = Literal["binary_ops", "cipher", "linear_eq", "roman", "number_seq", "list_ops", "modular_arith"]

# Wonderland framing header used across all families.
# Kept as a constant so every generator produces a consistent surface form
# that matches the real evaluation distribution.
_WONDERLAND_HEADER: str = (
    "In Alice's Wonderland, a secret transformation rule governs how things change. "
    "Study the examples below to discover the hidden rule, then apply it to the query."
)


@dataclass(slots=True)
class Example:
    """One training / evaluation instance.

    Attributes:
        prompt:   The full user-visible puzzle (header + examples + query).
        answer:   The ground-truth answer string exactly as it should appear
                  inside \\boxed{}.
        domain:   Which generator family produced this example.
        gold_cot: Full chain-of-thought string ending in ``\\boxed{answer}``.
                  Used as the SFT completion target.
    """

    prompt: str
    answer: str
    domain: Domain
    gold_cot: str


def format_wonderland_prompt(
    *,
    pairs: list[tuple[str, str]],
    query_input: str,
    extra_hint: str = "",
) -> str:
    """Build the puzzle prompt in the Alice's Wonderland style.

    Args:
        pairs:       List of (input, output) demonstration pairs.
        query_input: The unseen input the model must transform.
        extra_hint:  Optional domain-flavour sentence appended after the header.

    Returns:
        Formatted prompt string (no answer; ends with "Query input: ...").
    """
    lines: list[str] = [_WONDERLAND_HEADER]
    if extra_hint:
        lines.append(extra_hint)
    lines.append("")
    lines.append("Examples:")
    for i, (inp, out) in enumerate(pairs, start=1):
        lines.append(f"  {i}. Input: {inp}  →  Output: {out}")
    lines.append("")
    lines.append(f"Query input: {query_input}")
    lines.append("What is the output?")
    return "\n".join(lines)


def _cot_header(domain_label: str) -> str:
    """Return a short header line for chain-of-thought traces."""
    return f"Let me work through this {domain_label} puzzle step by step."
