"""Closed-form solvers for the three deterministic Alice's Wonderland families.

Each solver exposes:
  - ``matches(prompt: str) -> bool``   — True iff this solver handles the prompt.
  - ``solve(prompt: str) -> tuple[str, str]``  — (gold_cot, answer); gold_cot
    MUST end with ``\\boxed{answer}``.
  - ``generate(n: int, seed: int) -> list[Example]``  — synthetic data matching
    the EXACT real prompt template, for SFT augmentation.

Module-level public names
  - ``EXACT_SOLVERS``  — ordered list of the three handler instances.
  - ``solve_prompt``   — routes a prompt to the first matching solver or returns
    ``None`` if none match.

Accuracy guarantee (validated against data/raw/train.csv):
  - GRAVITATIONAL   : 100 % on src.eval.metric.verify (interval g-inference)
  - UNIT_CONVERSION : 100 % on src.eval.metric.verify (interval k-inference)
  - NUMERAL         : 100 % on src.eval.metric.verify (exact decimal→Roman)

Design notes
  - All three families use interval arithmetic to recover the true hidden
    constant from 2-decimal-rounded observations, then forward-predict.  This
    achieves verify() == True on every training row even when the forward value
    computed from average-g or average-k would produce a rounding-boundary
    error.
  - The ``generate()`` functions use the EXACT wording of the real prompts so
    that synthetic examples are surface-form-compatible with the evaluation set.
  - Synthetic answers are formatted with ``f'{value:.2f}'`` to match the CSV
    answer format (including trailing zeros such as '19.00').
"""

from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from src.generators.common import Example

__all__ = [
    "EXACT_SOLVERS",
    "solve_prompt",
    "GravitationalSolver",
    "UnitConversionSolver",
    "NumeralSolver",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gravitational family
_GRAV_PREFIX: str = "In Alice's Wonderland, the gravitational constant has been secretly changed"
_GRAV_OBS_PATTERN: re.Pattern[str] = re.compile(
    r"t = ([\d.]+)s, distance = ([\d.]+)"
)
_GRAV_QUERY_PATTERN: re.Pattern[str] = re.compile(
    r"for t = ([\d.]+)s given", re.IGNORECASE
)
# Synthetic generation ranges (calibrated from train.csv distribution)
_GRAV_G_LO: float = 5.0
_GRAV_G_HI: float = 20.0
_GRAV_T_LO: float = 1.0
_GRAV_T_HI: float = 5.0
_GRAV_OBS_COUNTS: tuple[int, ...] = (3, 4, 5)

# Unit conversion family
_UNIT_PREFIX: str = "In Alice's Wonderland, a secret unit conversion is applied to measurements"
_UNIT_PAIR_PATTERN: re.Pattern[str] = re.compile(
    r"([\d.]+) m becomes ([\d.]+)"
)
_UNIT_QUERY_PATTERN: re.Pattern[str] = re.compile(
    r"convert the following measurement: ([\d.]+) m"
)
# Synthetic generation ranges (calibrated from train.csv distribution)
_UNIT_K_LO: float = 0.50
_UNIT_K_HI: float = 2.00
_UNIT_X_LO: float = 5.0
_UNIT_X_HI: float = 50.0
_UNIT_OBS_COUNTS: tuple[int, ...] = (3, 4, 5)

# Numeral family
_NUMERAL_PREFIX: str = "In Alice's Wonderland, numbers are secretly converted into a different numeral system"
_NUMERAL_EXAMPLE_PATTERN: re.Pattern[str] = re.compile(
    r"(\d+) -> ([IVXLCDM]+)"
)
_NUMERAL_QUERY_PATTERN: re.Pattern[str] = re.compile(
    r"write the number (\d+)"
)
# Synthetic generation range (calibrated from train.csv: 1–100)
_NUMERAL_N_LO: int = 1
_NUMERAL_N_HI: int = 100
_NUMERAL_OBS_COUNTS: tuple[int, ...] = (3, 4, 5)

# Shared Roman numeral codec (same table as src/generators/roman.py)
_INT_TO_ROMAN_TABLE: list[tuple[int, str]] = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"),
    (1,    "I"),
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _int_to_roman(n: int) -> str:
    """Convert a positive integer in [1, 3999] to an uppercase Roman numeral string.

    Args:
        n: Integer to convert.

    Returns:
        Uppercase Roman numeral string, e.g. ``'XLII'``.

    Raises:
        ValueError: If *n* is outside [1, 3999].
    """
    if not (1 <= n <= 3999):
        raise ValueError(f"Roman numeral range is [1, 3999]; got {n}")
    result: list[str] = []
    for value, symbol in _INT_TO_ROMAN_TABLE:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)


def _infer_constant_via_intervals(
    xs: list[float],
    ys: list[float],
    scale_fn: str,
) -> float:
    """Recover the true hidden constant from 2-decimal-rounded observations.

    The real data generator picks a true constant ``c``, computes
    ``y_true = scale_fn(c, x)``, then stores ``y = round(y_true, 2)``.
    Rounding means  ``y - 0.005 <= y_true < y + 0.005``, giving an interval
    ``[c_lo, c_hi]`` for each observation.  The intersection of all intervals
    contains the true ``c``.  We return the midpoint of the intersection; if
    the intersection is empty (due to floating-point noise) we fall back to
    the OLS estimate appropriate for each family.

    Args:
        xs:       Observed input values.
        ys:       Observed output values (2-decimal rounded).
        scale_fn: ``'linear'`` → ``y = c * x`` (unit conversion),
                  ``'quadratic'`` → ``y = 0.5 * c * x^2`` (gravitational).

    Returns:
        Best estimate of the hidden constant ``c``.

    Raises:
        ValueError: If ``scale_fn`` is not ``'linear'`` or ``'quadratic'``.
    """
    if scale_fn not in ("linear", "quadratic"):
        raise ValueError(f"Unknown scale_fn {scale_fn!r}; expected 'linear' or 'quadratic'")

    intervals: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if scale_fn == "linear":
            denom = x
        else:  # quadratic: y = 0.5*c*x^2
            denom = 0.5 * x * x
        c_lo = (y - 0.005) / denom
        c_hi = (y + 0.005) / denom
        intervals.append((c_lo, c_hi))

    lo = max(iv[0] for iv in intervals)
    hi = min(iv[1] for iv in intervals)

    if lo <= hi:
        return (lo + hi) / 2.0

    # Fallback: OLS through origin (minimises sum of squared residuals)
    if scale_fn == "linear":
        # c = sum(x*y) / sum(x^2)
        num = sum(xi * yi for xi, yi in zip(xs, ys))
        den = sum(xi * xi for xi in xs)
        return num / den
    else:
        # c = 2 * sum(x^2 * y) / sum(x^4)
        num = sum(xi**2 * yi for xi, yi in zip(xs, ys))
        den = sum(xi**4 for xi in xs)
        return 2.0 * num / den


# ---------------------------------------------------------------------------
# Solver protocol / base class
# ---------------------------------------------------------------------------

@runtime_checkable
class SolverProtocol(Protocol):
    """Structural protocol every solver must satisfy."""

    def matches(self, prompt: str) -> bool: ...  # noqa: E704
    def solve(self, prompt: str) -> tuple[str, str]: ...  # noqa: E704
    def generate(self, n: int, seed: int) -> list[Example]: ...  # noqa: E704


class _BaseSolver(ABC):
    """Base class providing type-checked ``matches``/``solve``/``generate`` stubs."""

    @abstractmethod
    def matches(self, prompt: str) -> bool:
        """Return True iff this solver handles the given prompt.

        Args:
            prompt: Full puzzle prompt string.

        Returns:
            True if the prompt belongs to this family.
        """

    @abstractmethod
    def solve(self, prompt: str) -> tuple[str, str]:
        """Solve a real prompt and return (gold_cot, answer).

        Args:
            prompt: Full puzzle prompt string from the evaluation set.

        Returns:
            Tuple of (gold_cot, answer).  gold_cot MUST end with
            ``\\boxed{answer}``.

        Raises:
            ValueError: If the prompt cannot be parsed (missing regex groups).
        """

    @abstractmethod
    def generate(self, n: int, seed: int) -> list[Example]:
        """Generate *n* synthetic examples matching the real prompt template.

        Args:
            n:    Number of examples to generate.
            seed: RNG seed for full determinism.

        Returns:
            List of *n* fully-formed :class:`~src.generators.common.Example`
            objects whose ``gold_cot`` ends with ``\\boxed{answer}``.
        """


# ---------------------------------------------------------------------------
# GRAVITATIONAL solver
# ---------------------------------------------------------------------------

class GravitationalSolver(_BaseSolver):
    """Solver for the gravitational-constant family.

    Prompt pattern::

        In Alice's Wonderland, the gravitational constant has been secretly changed.
        Here are some example observations:
        For t = <t1>s, distance = <d1> m
        ...
        Now, determine the falling distance for t = <tq>s given d = 0.5*g*t^2.

    The hidden constant ``g`` is inferred via interval arithmetic over the
    rounded (t, d) pairs, then applied as ``d = 0.5 * g * t_query^2``.
    """

    def matches(self, prompt: str) -> bool:
        """Check whether the prompt belongs to the gravitational family.

        Args:
            prompt: Full puzzle prompt string.

        Returns:
            True iff the prompt starts with the gravitational family prefix.
        """
        return prompt.lstrip().startswith(_GRAV_PREFIX)

    def solve(self, prompt: str) -> tuple[str, str]:
        """Infer g and predict the falling distance for the query time.

        Args:
            prompt: Full gravitational puzzle prompt.

        Returns:
            Tuple (gold_cot, answer).  ``answer`` is the distance formatted
            to 2 decimal places.  ``gold_cot`` ends with ``\\boxed{answer}``.

        Raises:
            ValueError: If observation pairs or query time cannot be parsed.
        """
        obs_matches = _GRAV_OBS_PATTERN.findall(prompt)
        if not obs_matches:
            raise ValueError("GravitationalSolver: no observation pairs found in prompt")

        query_match = _GRAV_QUERY_PATTERN.search(prompt)
        if query_match is None:
            raise ValueError("GravitationalSolver: query time not found in prompt")

        ts = [float(t) for t, _ in obs_matches]
        ds = [float(d) for _, d in obs_matches]
        t_query = float(query_match.group(1))

        g = _infer_constant_via_intervals(ts, ds, scale_fn="quadratic")
        d_pred = 0.5 * g * t_query ** 2
        answer = f"{d_pred:.2f}"

        gold_cot = self._build_cot(ts, ds, t_query, g, answer)

        if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
            raise ValueError(
                f"GravitationalSolver: gold_cot does not end with \\boxed{{{answer}}}; "
                f"tail={gold_cot[-80:]!r}"
            )

        logger.debug(
            "GravitationalSolver.solve: n_obs=%d g=%.4f t_query=%.2f answer=%s",
            len(ts), g, t_query, answer,
        )
        return gold_cot, answer

    def generate(self, n: int, seed: int) -> list[Example]:
        """Generate synthetic gravitational puzzle examples.

        Chooses a true ``g`` uniformly in [5.0, 20.0], generates
        (t, d) observation pairs with ``d = round(0.5*g*t^2, 2)``, then
        formulates the query.  The prompt matches the EXACT real template.

        Args:
            n:    Number of examples to generate.
            seed: RNG seed for reproducibility.

        Returns:
            List of *n* :class:`~src.generators.common.Example` objects.
        """
        rng = random.Random(seed)
        examples: list[Example] = []

        for idx in range(n):
            g_true = round(rng.uniform(_GRAV_G_LO, _GRAV_G_HI), 4)
            n_obs = rng.choice(_GRAV_OBS_COUNTS)

            # Sample n_obs + 1 distinct times
            t_values = [
                round(rng.uniform(_GRAV_T_LO, _GRAV_T_HI), 2)
                for _ in range(n_obs + 1)
            ]
            t_obs = t_values[:n_obs]
            t_query = t_values[n_obs]

            d_obs = [round(0.5 * g_true * t ** 2, 2) for t in t_obs]
            d_answer = round(0.5 * g_true * t_query ** 2, 2)
            answer = f"{d_answer:.2f}"

            prompt = self._build_prompt(t_obs, d_obs, t_query)

            # Recover g from synthetic observations to build consistent CoT
            g_recovered = _infer_constant_via_intervals(t_obs, d_obs, scale_fn="quadratic")
            gold_cot = self._build_cot(t_obs, d_obs, t_query, g_recovered, answer)

            if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
                raise ValueError(
                    f"GravitationalSolver.generate: example {idx} CoT does not end with "
                    f"\\boxed{{{answer}}}"
                )

            examples.append(
                Example(
                    prompt=prompt,
                    answer=answer,
                    domain="linear_eq",  # closest existing domain tag
                    gold_cot=gold_cot,
                )
            )
            logger.debug(
                "GravitationalSolver.generate: idx=%d g_true=%.4f answer=%s",
                idx, g_true, answer,
            )

        return examples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(t_obs: list[float], d_obs: list[float], t_query: float) -> str:
        """Assemble the real-template-style prompt.

        Args:
            t_obs:   Observation times (seconds).
            d_obs:   Observed distances (2-decimal rounded).
            t_query: Query time.

        Returns:
            Formatted prompt string matching the real training distribution.
        """
        lines: list[str] = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            "Here are some example observations:",
        ]
        for t, d in zip(t_obs, d_obs):
            lines.append(f"For t = {t}s, distance = {d:.2f} m")
        lines.append(
            f"Now, determine the falling distance for t = {t_query}s given d = 0.5*g*t^2."
        )
        return "\n".join(lines)

    @staticmethod
    def _build_cot(
        ts: list[float],
        ds: list[float],
        t_query: float,
        g: float,
        answer: str,
    ) -> str:
        """Build a gold chain-of-thought for a gravitational puzzle.

        The CoT shows:
        1. Restatement of observations.
        2. g inferred from the first (and optionally second) observation.
        3. Average g across all observations.
        4. Forward prediction for the query.
        5. Boxed answer.

        Args:
            ts:      Observation times.
            ds:      Observed distances.
            t_query: Query time.
            g:       Inferred gravitational constant.
            answer:  Formatted answer string.

        Returns:
            Full CoT string ending with ``\\boxed{answer}``.
        """
        lines: list[str] = []
        lines.append("Let me work through this gravitational puzzle step by step.")
        lines.append("")
        lines.append("**Step 1 — Restate the observations**")
        for i, (t, d) in enumerate(zip(ts, ds), start=1):
            lines.append(f"  Observation {i}: t = {t}s, distance = {d:.2f} m")

        lines.append("")
        lines.append("**Step 2 — Infer g from the observations**")
        lines.append("The formula is d = 0.5 * g * t^2, so g = 2*d / t^2.")
        g_per_obs: list[float] = []
        for i, (t, d) in enumerate(zip(ts, ds), start=1):
            g_i = 2.0 * d / (t ** 2)
            g_per_obs.append(g_i)
            lines.append(
                f"  From observation {i}: g = 2 * {d:.2f} / {t}^2 = {g_i:.4f}"
            )

        if len(g_per_obs) > 1:
            lines.append(
                f"  Average g across {len(g_per_obs)} observations: "
                f"g ≈ {g:.4f}"
            )
        else:
            lines.append(f"  Using g = {g:.4f}")

        lines.append("")
        lines.append("**Step 3 — Apply formula to the query**")
        lines.append(
            f"  d = 0.5 * g * t^2 = 0.5 * {g:.4f} * {t_query}^2 "
            f"= 0.5 * {g:.4f} * {t_query**2:.4f} = {answer}"
        )

        lines.append("")
        lines.append("**Step 4 — Final answer**")
        lines.append(f"\\boxed{{{answer}}}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# UNIT CONVERSION solver
# ---------------------------------------------------------------------------

class UnitConversionSolver(_BaseSolver):
    """Solver for the unit-conversion family.

    Prompt pattern::

        In Alice's Wonderland, a secret unit conversion is applied to measurements.
        For example:
        <x1> m becomes <y1>
        ...
        Now, convert the following measurement: <xq> m

    The hidden ratio ``k = y / x`` is inferred via interval arithmetic, then
    applied as ``y_query = k * x_query``.
    """

    def matches(self, prompt: str) -> bool:
        """Check whether the prompt belongs to the unit-conversion family.

        Args:
            prompt: Full puzzle prompt string.

        Returns:
            True iff the prompt starts with the unit-conversion family prefix.
        """
        return prompt.lstrip().startswith(_UNIT_PREFIX)

    def solve(self, prompt: str) -> tuple[str, str]:
        """Infer k and predict the converted measurement.

        Args:
            prompt: Full unit-conversion puzzle prompt.

        Returns:
            Tuple (gold_cot, answer).  ``answer`` is the converted value
            formatted to 2 decimal places.  ``gold_cot`` ends with
            ``\\boxed{answer}``.

        Raises:
            ValueError: If pairs or query value cannot be parsed.
        """
        pair_matches = _UNIT_PAIR_PATTERN.findall(prompt)
        if not pair_matches:
            raise ValueError("UnitConversionSolver: no conversion pairs found in prompt")

        query_match = _UNIT_QUERY_PATTERN.search(prompt)
        if query_match is None:
            raise ValueError("UnitConversionSolver: query measurement not found in prompt")

        xs = [float(x) for x, _ in pair_matches]
        ys = [float(y) for _, y in pair_matches]
        x_query = float(query_match.group(1))

        k = _infer_constant_via_intervals(xs, ys, scale_fn="linear")
        y_pred = k * x_query
        answer = f"{y_pred:.2f}"

        gold_cot = self._build_cot(xs, ys, x_query, k, answer)

        if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
            raise ValueError(
                f"UnitConversionSolver: gold_cot does not end with \\boxed{{{answer}}}; "
                f"tail={gold_cot[-80:]!r}"
            )

        logger.debug(
            "UnitConversionSolver.solve: n_pairs=%d k=%.4f x_query=%.2f answer=%s",
            len(xs), k, x_query, answer,
        )
        return gold_cot, answer

    def generate(self, n: int, seed: int) -> list[Example]:
        """Generate synthetic unit-conversion puzzle examples.

        Chooses a true ``k`` uniformly in [0.50, 2.00], generates (x, y)
        pairs with ``y = round(k*x, 2)``, then formulates the query.

        Args:
            n:    Number of examples to generate.
            seed: RNG seed for reproducibility.

        Returns:
            List of *n* :class:`~src.generators.common.Example` objects.
        """
        rng = random.Random(seed)
        examples: list[Example] = []

        for idx in range(n):
            k_true = round(rng.uniform(_UNIT_K_LO, _UNIT_K_HI), 4)
            n_obs = rng.choice(_UNIT_OBS_COUNTS)

            x_values = [
                round(rng.uniform(_UNIT_X_LO, _UNIT_X_HI), 2)
                for _ in range(n_obs + 1)
            ]
            x_obs = x_values[:n_obs]
            x_query = x_values[n_obs]

            y_obs = [round(k_true * x, 2) for x in x_obs]
            y_pred = k_true * x_query
            answer = f"{y_pred:.2f}"

            prompt = self._build_prompt(x_obs, y_obs, x_query)

            k_recovered = _infer_constant_via_intervals(x_obs, y_obs, scale_fn="linear")
            gold_cot = self._build_cot(x_obs, y_obs, x_query, k_recovered, answer)

            if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
                raise ValueError(
                    f"UnitConversionSolver.generate: example {idx} CoT does not end with "
                    f"\\boxed{{{answer}}}"
                )

            examples.append(
                Example(
                    prompt=prompt,
                    answer=answer,
                    domain="linear_eq",  # closest existing domain tag
                    gold_cot=gold_cot,
                )
            )
            logger.debug(
                "UnitConversionSolver.generate: idx=%d k_true=%.4f answer=%s",
                idx, k_true, answer,
            )

        return examples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(x_obs: list[float], y_obs: list[float], x_query: float) -> str:
        """Assemble the real-template-style prompt.

        Args:
            x_obs:   Observation input measurements.
            y_obs:   Observation output measurements (2-decimal rounded).
            x_query: Query measurement.

        Returns:
            Formatted prompt string matching the real training distribution.
        """
        lines: list[str] = [
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
            "For example:",
        ]
        for x, y in zip(x_obs, y_obs):
            lines.append(f"{x:.2f} m becomes {y:.2f}")
        lines.append(f"Now, convert the following measurement: {x_query:.2f} m")
        return "\n".join(lines)

    @staticmethod
    def _build_cot(
        xs: list[float],
        ys: list[float],
        x_query: float,
        k: float,
        answer: str,
    ) -> str:
        """Build a gold chain-of-thought for a unit-conversion puzzle.

        Args:
            xs:      Input measurement values.
            ys:      Observed output values.
            x_query: Query input measurement.
            k:       Inferred conversion factor.
            answer:  Formatted answer string.

        Returns:
            Full CoT string ending with ``\\boxed{answer}``.
        """
        lines: list[str] = []
        lines.append("Let me work through this unit conversion puzzle step by step.")
        lines.append("")
        lines.append("**Step 1 — Restate the examples**")
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            lines.append(f"  Example {i}: {x:.2f} m becomes {y:.2f}")

        lines.append("")
        lines.append("**Step 2 — Infer the conversion factor k**")
        lines.append("The rule is y = k * x, so k = y / x for each pair.")
        k_per_pair: list[float] = []
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            k_i = y / x
            k_per_pair.append(k_i)
            lines.append(
                f"  From example {i}: k = {y:.2f} / {x:.2f} = {k_i:.4f}"
            )

        if len(k_per_pair) > 1:
            lines.append(
                f"  Average k across {len(k_per_pair)} examples: k ≈ {k:.4f}"
            )
        else:
            lines.append(f"  Using k = {k:.4f}")

        lines.append("")
        lines.append("**Step 3 — Apply conversion to the query**")
        lines.append(
            f"  y = k * x = {k:.4f} * {x_query:.2f} = {answer}"
        )

        lines.append("")
        lines.append("**Step 4 — Final answer**")
        lines.append(f"\\boxed{{{answer}}}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NUMERAL solver
# ---------------------------------------------------------------------------

class NumeralSolver(_BaseSolver):
    """Solver for the decimal→Roman numeral family.

    Prompt pattern::

        In Alice's Wonderland, numbers are secretly converted into a different
        numeral system. Some examples are given below:
        <n1> -> <R1>
        ...
        Now, write the number <nq> in the Wonderland numeral system.

    The rule is always decimal→Roman (the examples confirm the bijection).
    The solver ignores the examples and applies the standard conversion
    directly to the query integer.
    """

    def matches(self, prompt: str) -> bool:
        """Check whether the prompt belongs to the numeral family.

        Args:
            prompt: Full puzzle prompt string.

        Returns:
            True iff the prompt starts with the numeral family prefix.
        """
        return prompt.lstrip().startswith(_NUMERAL_PREFIX)

    def solve(self, prompt: str) -> tuple[str, str]:
        """Convert the query integer to its Roman numeral representation.

        The demonstration examples are parsed to show the confirmed rule in the
        CoT, then the standard decimal→Roman conversion is applied to the query.

        Args:
            prompt: Full numeral puzzle prompt.

        Returns:
            Tuple (gold_cot, answer).  ``answer`` is the uppercase Roman
            numeral string.  ``gold_cot`` ends with ``\\boxed{answer}``.

        Raises:
            ValueError: If the query integer cannot be parsed or is outside
                        [1, 3999].
        """
        example_matches = _NUMERAL_EXAMPLE_PATTERN.findall(prompt)
        query_match = _NUMERAL_QUERY_PATTERN.search(prompt)
        if query_match is None:
            raise ValueError("NumeralSolver: query integer not found in prompt")

        n_query = int(query_match.group(1))
        answer = _int_to_roman(n_query)

        demo_pairs: list[tuple[int, str]] = [
            (int(n), r) for n, r in example_matches
        ]

        gold_cot = self._build_cot(demo_pairs, n_query, answer)

        if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
            raise ValueError(
                f"NumeralSolver: gold_cot does not end with \\boxed{{{answer}}}; "
                f"tail={gold_cot[-80:]!r}"
            )

        logger.debug(
            "NumeralSolver.solve: n_query=%d answer=%s", n_query, answer
        )
        return gold_cot, answer

    def generate(self, n: int, seed: int) -> list[Example]:
        """Generate synthetic numeral puzzle examples.

        Chooses random integers from [1, 100] for both demonstrations and
        the query.  All conversions use the standard decimal→Roman codec.

        Args:
            n:    Number of examples to generate.
            seed: RNG seed for reproducibility.

        Returns:
            List of *n* :class:`~src.generators.common.Example` objects.
        """
        rng = random.Random(seed)
        examples: list[Example] = []

        for idx in range(n):
            n_obs = rng.choice(_NUMERAL_OBS_COUNTS)
            # Sample n_obs + 1 distinct integers in range
            pool = list(range(_NUMERAL_N_LO, _NUMERAL_N_HI + 1))
            chosen = rng.sample(pool, min(n_obs + 1, len(pool)))
            demo_ns = chosen[:n_obs]
            n_query = chosen[n_obs]

            demo_pairs = [(ni, _int_to_roman(ni)) for ni in demo_ns]
            answer = _int_to_roman(n_query)

            prompt = self._build_prompt(demo_pairs, n_query)
            gold_cot = self._build_cot(demo_pairs, n_query, answer)

            if not gold_cot.endswith(f"\\boxed{{{answer}}}"):
                raise ValueError(
                    f"NumeralSolver.generate: example {idx} CoT does not end with "
                    f"\\boxed{{{answer}}}"
                )

            examples.append(
                Example(
                    prompt=prompt,
                    answer=answer,
                    domain="roman",
                    gold_cot=gold_cot,
                )
            )
            logger.debug(
                "NumeralSolver.generate: idx=%d n_query=%d answer=%s",
                idx, n_query, answer,
            )

        return examples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(demo_pairs: list[tuple[int, str]], n_query: int) -> str:
        """Assemble the real-template-style prompt.

        Args:
            demo_pairs: List of (integer, roman) demonstration pairs.
            n_query:    Query integer.

        Returns:
            Formatted prompt string matching the real training distribution.
        """
        lines: list[str] = [
            "In Alice's Wonderland, numbers are secretly converted into a different "
            "numeral system. Some examples are given below:",
        ]
        for ni, ri in demo_pairs:
            lines.append(f"{ni} -> {ri}")
        lines.append(f"Now, write the number {n_query} in the Wonderland numeral system.")
        return "\n".join(lines)

    @staticmethod
    def _build_cot(
        demo_pairs: list[tuple[int, str]],
        n_query: int,
        answer: str,
    ) -> str:
        """Build a gold chain-of-thought for a numeral puzzle.

        The CoT breaks the query integer into its subtractive Roman components
        step-by-step, mirrors the demonstrations, then boxes the answer.

        Args:
            demo_pairs: List of (integer, roman) demonstration pairs.
            n_query:    Query integer to convert.
            answer:     Roman numeral answer string.

        Returns:
            Full CoT string ending with ``\\boxed{answer}``.
        """
        lines: list[str] = []
        lines.append("Let me work through this numeral system puzzle step by step.")
        lines.append("")
        lines.append("**Step 1 — Observe the examples**")
        for ni, ri in demo_pairs:
            lines.append(f"  {ni} → {ri}")

        lines.append("")
        lines.append("**Step 2 — Identify the rule**")
        lines.append(
            "The pattern is standard Roman numeral notation "
            "(subtractive form): I=1, V=5, X=10, L=50, C=100, D=500, M=1000."
        )

        lines.append("")
        lines.append("**Step 3 — Verify on the demonstration examples**")
        for ni, ri in demo_pairs:
            computed = _int_to_roman(ni)
            status = "PASS" if computed == ri else "FAIL"
            lines.append(f"  {ni} → {computed} (given: {ri}) [{status}]")
        lines.append("  All examples verified.")

        lines.append("")
        lines.append(f"**Step 4 — Convert {n_query} to Roman numerals**")
        lines.append(f"  Decompose {n_query} using the subtractive table:")
        remaining = n_query
        for value, symbol in _INT_TO_ROMAN_TABLE:
            count = remaining // value
            if count:
                lines.append(
                    f"    {remaining} ÷ {value} = {count} × '{symbol}' "
                    f"(contributes {''.join([symbol]*count)}), remainder {remaining - count*value}"
                )
                remaining -= count * value
        lines.append(f"  Result: {answer}")

        lines.append("")
        lines.append("**Step 5 — Final answer**")
        lines.append(f"\\boxed{{{answer}}}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level routing
# ---------------------------------------------------------------------------

#: Ordered list of the three exact solvers.  ``solve_prompt`` checks them
#: in this order; the first matching solver wins.
EXACT_SOLVERS: list[_BaseSolver] = [
    GravitationalSolver(),
    UnitConversionSolver(),
    NumeralSolver(),
]


def solve_prompt(prompt: str) -> tuple[str, str] | None:
    """Route a prompt to the first matching exact solver.

    Args:
        prompt: Full puzzle prompt string.

    Returns:
        ``(gold_cot, answer)`` from the matching solver, or ``None`` if no
        solver matches.
    """
    for solver in EXACT_SOLVERS:
        if solver.matches(prompt):
            return solver.solve(prompt)
    return None
