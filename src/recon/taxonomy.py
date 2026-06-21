"""Family-taxonomy recon for the Nemotron reasoning challenge train set.

Answers the one question that decides medal vs. contender: *does our synthetic
data cover the hidden test's rule families?* It does this by

  1. classifying every train.csv row into a generator domain (or "uncovered"),
  2. clustering rows by a normalised prompt *template* signature (so distinct
     families inside a domain, and entirely new families, both surface),
  3. profiling the answer FORMAT per domain (binary / int / float / roman /
     word / phrase / other) and, for binary, the bit-WIDTH distribution, because
     the official ``verify()`` treats binary answers as length-sensitive strings,
  4. cross-checking those formats against what ``src/generators`` actually emits
     and printing explicit MISSING-COVERAGE warnings.

Pure standard library — runs on CPU with no installed dependencies.

Usage:
    python -m src.recon.taxonomy --train-csv data/raw/train.csv
    python -m src.recon.taxonomy --train-csv data/raw/train.csv --json-out recon.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "AnswerFormat",
    "DomainStat",
    "ReconReport",
    "classify_answer",
    "classify_domain",
    "template_signature",
    "analyze_rows",
    "render_report",
    "main",
]

logger = logging.getLogger("recon.taxonomy")

# Domains our generators in src/generators/ produce.
GENERATOR_DOMAINS: tuple[str, ...] = (
    "binary_ops", "cipher", "linear_eq", "roman",
    "number_seq", "list_ops", "modular_arith",
)

# Answer formats we expect each generator to emit. A real domain whose answers
# fall outside this set is a coverage gap (the SFT traces won't match verify()).
GENERATOR_EXPECTED_FORMATS: dict[str, set[str]] = {
    "binary_ops": {"binary"},
    "cipher": {"word", "phrase"},
    "linear_eq": {"int", "float"},
    "roman": {"roman", "int"},  # roman_to_int rules emit bare int answers
    "number_seq": {"int"},
    "list_ops": {"other"},      # comma-separated strings like "-3, 1, 7" -> "other"
    "modular_arith": {"int"},
}
# Bit-widths our binary generator emits. Real widths outside this set => gap,
# because verify() compares binary answers as exact-length strings.
GENERATOR_BINARY_WIDTHS: set[int] = {8}

AnswerFormat = str  # one of: binary int float roman word phrase hex other empty

_RE_BINARY = re.compile(r"[01]+$")
_RE_INT = re.compile(r"-?\d+$")
_RE_FLOAT = re.compile(r"-?\d+\.\d+$")
_RE_ROMAN = re.compile(r"[IVXLCDM]+$", re.IGNORECASE)
_RE_WORD = re.compile(r"[A-Za-z]+$")
_RE_HEX = re.compile(r"[0-9A-Fa-f]+$")
_RE_PHRASE = re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+)+$")


def classify_answer(answer: str) -> AnswerFormat:
    """Classify a ground-truth answer into a format bucket.

    Order matters: binary is checked before int/hex because the official
    ``verify()`` routes any all-[01] string through strict string comparison,
    never numeric tolerance.  However, a single character "0" or "1" is
    ambiguous: it satisfies the binary regex but is also a valid small integer
    that modular_arith (and other families) legitimately produce.  We resolve
    the ambiguity conservatively: a single "0" or "1" is classified as "int"
    rather than "binary", because binary_ops always emits fixed-width strings
    (BIT_WIDTH == 8) so a 1-char answer cannot originate from that family.
    Multi-character all-[01] strings (length >= 2) remain "binary" because
    they cannot be integers without a leading zero (which is not a valid bare
    int representation) and match the binary_ops output format.

    Args:
        answer: The raw ``answer`` field from train.csv.

    Returns:
        One of: binary, int, float, roman, word, phrase, hex, other, empty.
    """
    a = answer.strip()
    if not a:
        return "empty"
    # Guard: single "0" or "1" is classified as int, not binary.
    # binary_ops always emits BIT_WIDTH-char zero-padded strings (length >= 2).
    if len(a) >= 2 and _RE_BINARY.fullmatch(a):
        return "binary"
    if _RE_INT.fullmatch(a):
        return "int"
    if _RE_FLOAT.fullmatch(a):
        return "float"
    if _RE_ROMAN.fullmatch(a):
        return "roman"
    if _RE_PHRASE.fullmatch(a):
        return "phrase"
    if _RE_WORD.fullmatch(a):
        return "word"
    if _RE_HEX.fullmatch(a):
        return "hex"
    return "other"


# Keyword signals for domain classification, checked in priority order.
# Rule: more-specific / longer phrases come before single-word catches so
# that e.g. "circular clock" is caught by modular_arith before "binary" in
# binary_ops could ever fire.  Existing 4 families are unchanged.
_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ── new families (checked first where tokens might overlap) ──────────────
    # modular_arith: hint says "circular clock that resets back to zero"
    ("modular_arith", ("circular clock", "resets back to zero", "clock arithmetic",
                       "modulo", "mod ")),
    # list_ops: hint says "lists of numbers are transformed by a hidden structural rule"
    ("list_ops", ("lists of numbers", "hidden structural rule", "structural rule",
                  "list transformation", "rotate the list", "sort the list",
                  "reverse the order of the list")),
    # number_seq: hint says "numbers follow a hidden pattern in a magical sequence"
    ("number_seq", ("magical sequence", "hidden pattern in a magical",
                    "secret rule applied to its position",
                    "numbers follow a hidden pattern")),
    # ── original 4 families ───────────────────────────────────────────────────
    ("roman", ("roman",)),
    ("binary_ops", ("bit manipulation", "binary", "8-bit", "bitwise", "bits")),
    ("cipher", ("encrypt", "cipher", "secret encryption", "decode", "decrypt",
                "shift", "substitut")),
    ("linear_eq", ("equation", "solve for", "algebra", "linear", "value of x")),
)


def classify_domain(prompt: str) -> str:
    """Map a prompt to a generator domain, or 'uncovered' if none matches.

    Args:
        prompt: The puzzle prompt text.

    Returns:
        A domain in GENERATOR_DOMAINS, or 'uncovered'.
    """
    p = prompt.lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(k in p for k in keywords):
            return domain
    return "uncovered"


_RE_BIN_TOKEN = re.compile(r"\b[01]{3,}\b")
_RE_NUM_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")
_RE_ARROW = re.compile(r"\s*(?:->|=>|→|:)\s*")
_RE_WS = re.compile(r"\s+")


def template_signature(prompt: str, head_chars: int = 140) -> str:
    """Build a normalised template signature for clustering similar prompts.

    Variable content (binary tokens, numbers, arrows) is masked so prompts that
    differ only in their concrete values collapse to the same signature, while
    structurally different families stay distinct.

    Args:
        prompt: The puzzle prompt text.
        head_chars: Keep this many leading chars of the normalised string; the
            rule-describing preamble (the family fingerprint) lives at the front.

    Returns:
        A normalised signature string.
    """
    s = prompt.strip().lower()
    s = _RE_BIN_TOKEN.sub("<BIN>", s)
    s = _RE_NUM_TOKEN.sub("<NUM>", s)
    s = _RE_ARROW.sub(" : ", s)
    s = _RE_WS.sub(" ", s)
    return s[:head_chars]


@dataclass(slots=True)
class DomainStat:
    """Per-domain aggregate statistics."""

    domain: str
    count: int = 0
    answer_formats: Counter = field(default_factory=Counter)
    binary_widths: Counter = field(default_factory=Counter)
    templates: Counter = field(default_factory=Counter)
    example_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReconReport:
    """Full recon result over a train set."""

    total: int
    domains: dict[str, DomainStat]
    gaps: list[str]


def analyze_rows(rows: list[dict[str, str]]) -> ReconReport:
    """Analyze train rows into a ReconReport.

    Args:
        rows: List of dicts with at least 'prompt' and 'answer' keys
            (an 'id' key is used for examples when present).

    Returns:
        Populated ReconReport.

    Raises:
        ValueError: If rows is empty or missing required columns.
    """
    if not rows:
        raise ValueError("No rows to analyze.")
    required = {"prompt", "answer"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"train rows missing columns: {sorted(missing)}")

    domains: dict[str, DomainStat] = defaultdict(lambda: DomainStat(domain=""))
    for row in rows:
        prompt, answer = row["prompt"], row["answer"]
        dom = classify_domain(prompt)
        fmt = classify_answer(answer)
        st = domains[dom]
        st.domain = dom
        st.count += 1
        st.answer_formats[fmt] += 1
        st.templates[template_signature(prompt)] += 1
        if fmt == "binary":
            st.binary_widths[len(answer.strip())] += 1
        if len(st.example_ids) < 3:
            st.example_ids.append(row.get("id", "?"))

    gaps = _detect_gaps(domains)
    return ReconReport(total=len(rows), domains=dict(domains), gaps=gaps)


def _detect_gaps(domains: dict[str, DomainStat]) -> list[str]:
    """Derive explicit coverage-gap warnings from per-domain stats.

    Reports, per family:
    - covered count (rows mapped to a known generator domain)
    - uncovered count (rows that matched no generator)
    - format mismatches (real answer formats not emitted by our generator)
    - binary width mismatches (verify() is length-sensitive)
    - real-test answer format/width combos that NO generator covers

    Args:
        domains: Mapping of domain name -> DomainStat from analyze_rows.

    Returns:
        List of human-readable gap warning strings.
    """
    gaps: list[str] = []

    # ── coverage summary ──────────────────────────────────────────────────────
    total_covered = sum(
        st.count for d, st in domains.items() if d in GENERATOR_DOMAINS
    )
    total_uncovered = domains["uncovered"].count if "uncovered" in domains else 0
    gaps.append(
        f"COVERAGE SUMMARY: {total_covered} rows covered by known generators, "
        f"{total_uncovered} rows uncovered."
    )

    # ── uncovered rows ────────────────────────────────────────────────────────
    if "uncovered" in domains:
        st = domains["uncovered"]
        fmt_str = ", ".join(f"{k}={v}" for k, v in st.answer_formats.most_common())
        top_template = st.templates.most_common(1)[0][0] if st.templates else "<none>"
        gaps.append(
            f"UNCOVERED FAMILY: {st.count} rows match no generator domain. "
            f"Answer formats: {fmt_str}. "
            f"Sample ids: {st.example_ids}. "
            f"Top template: {top_template!r}"
        )

    # ── per-domain checks ─────────────────────────────────────────────────────
    for dom in GENERATOR_DOMAINS:
        if dom not in domains:
            gaps.append(
                f"NOT SEEN: generator domain {dom!r} has 0 rows in this "
                f"train set (may still appear in the hidden test)."
            )
            continue
        st = domains[dom]
        expected = GENERATOR_EXPECTED_FORMATS[dom]
        seen = {f for f in st.answer_formats if f != "empty"}
        unexpected = seen - expected
        if unexpected:
            # These are real answer formats our generator never emits — a hard
            # gap: any test row with such answers will score 0 on our SFT model.
            gaps.append(
                f"FORMAT GAP [{dom}]: real answers include {sorted(unexpected)} "
                f"but generator only emits {sorted(expected)}. "
                f"Counts: {dict(st.answer_formats)}. "
                f"ACTION: add generator variants that produce "
                f"{sorted(unexpected)} for {dom!r}."
            )
        if dom == "binary_ops" and st.binary_widths:
            real_widths = set(st.binary_widths)
            extra = real_widths - GENERATOR_BINARY_WIDTHS
            if extra:
                gaps.append(
                    f"BINARY WIDTH GAP: real binary answers have widths "
                    f"{sorted(real_widths)} but generator emits "
                    f"{sorted(GENERATOR_BINARY_WIDTHS)}-bit. verify() is "
                    f"length-sensitive => mismatched widths score 0. "
                    f"ACTION: add BIT_WIDTH variants {sorted(extra)}."
                )
    return gaps


def render_report(report: ReconReport) -> str:
    """Render a human-readable text report.

    Args:
        report: The ReconReport to render.

    Returns:
        Multi-line report string.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"FAMILY-TAXONOMY RECON  ·  {report.total} rows")
    lines.append("=" * 72)

    ordered = sorted(report.domains.values(), key=lambda s: -s.count)
    for st in ordered:
        pct = 100.0 * st.count / report.total
        tag = "" if st.domain in GENERATOR_DOMAINS else "  <-- NOT A GENERATOR DOMAIN"
        lines.append("")
        lines.append(f"[{st.domain}]  {st.count} rows ({pct:.1f}%){tag}")
        fmts = ", ".join(f"{k}={v}" for k, v in st.answer_formats.most_common())
        lines.append(f"  answer formats : {fmts}")
        if st.binary_widths:
            widths = ", ".join(f"{w}bit={c}" for w, c in
                               sorted(st.binary_widths.items()))
            lines.append(f"  binary widths  : {widths}")
        lines.append(f"  distinct templates: {len(st.templates)}")
        for sig, c in st.templates.most_common(3):
            lines.append(f"    ({c:>4}) {sig[:90]}")

    lines.append("")
    lines.append("-" * 72)
    if report.gaps:
        lines.append(f"COVERAGE GAPS ({len(report.gaps)}):")
        for g in report.gaps:
            lines.append(f"  ! {g}")
    else:
        lines.append("COVERAGE: no gaps detected — generators span the train families.")
    lines.append("-" * 72)
    return "\n".join(lines)


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV into a list of dict rows.

    Args:
        path: Path to train.csv.

    Returns:
        List of row dicts.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {path}. Download it first:\n"
            "  kaggle competitions download -c "
            "nvidia-nemotron-model-reasoning-challenge -p data/\n"
            "  unzip data/*.zip -d data/raw/"
        )
    # csv field size: prompts can be long.
    csv.field_size_limit(10_000_000)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 ok, 2 on bad input).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Family-taxonomy recon for train.csv")
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to dump the report as JSON.")
    args = parser.parse_args(argv)

    try:
        rows = _load_csv(args.train_csv)
        report = analyze_rows(rows)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    print(render_report(report))

    if args.json_out is not None:
        payload = {
            "total": report.total,
            "gaps": report.gaps,
            "domains": {
                d: {
                    "count": s.count,
                    "answer_formats": dict(s.answer_formats),
                    "binary_widths": dict(s.binary_widths),
                    "distinct_templates": len(s.templates),
                    "top_templates": s.templates.most_common(5),
                }
                for d, s in report.domains.items()
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote JSON report -> %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
