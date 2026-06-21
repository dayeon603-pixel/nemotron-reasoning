"""Build a synthetic training JSONL from all four generator families.

Output: data/synthetic.jsonl — same schema as data/accepted.jsonl so that
either file can be passed directly to src/sft_train.py.

Usage:
    python scripts/build_synthetic.py \
        --n_per_domain 500 \
        --seed 42 \
        --output data/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.generators import (
    generate_binary_ops,
    generate_cipher,
    generate_linear_eq,
    generate_roman,
    generate_number_seq,
    generate_list_ops,
    generate_modular_arith,
)

__all__: list[str] = []

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_N_PER_DOMAIN: int = 2000
DEFAULT_SEED: int = 42
DEFAULT_OUTPUT: Path = Path("data/synthetic.jsonl")


def build_synthetic(
    n_per_domain: int,
    seed: int,
    output: Path,
) -> int:
    """Generate synthetic training data for all seven domain families.

    Args:
        n_per_domain: Number of examples per domain family.
        seed:         Base seed; each family gets seed + domain_offset for
                      independence without overlap.
        output:       Path to write the JSONL file.

    Returns:
        Total number of examples written.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    generators = [
        ("binary_ops",     generate_binary_ops,     seed + 0),
        ("cipher",         generate_cipher,         seed + 1000),
        ("linear_eq",      generate_linear_eq,      seed + 2000),
        ("roman",          generate_roman,          seed + 3000),
        ("number_seq",     generate_number_seq,     seed + 4000),
        ("list_ops",       generate_list_ops,       seed + 5000),
        ("modular_arith",  generate_modular_arith,  seed + 6000),
    ]

    total = 0
    with output.open("w", encoding="utf-8") as fh:
        for domain_name, gen_fn, domain_seed in generators:
            examples = gen_fn(n_per_domain, domain_seed)
            for ex in examples:
                record = {
                    "id": f"synthetic_{domain_name}_{total:06d}",
                    "prompt": ex.prompt,
                    "trace": ex.gold_cot,
                    "extracted_answer": ex.answer,
                    "gold_answer": ex.answer,
                }
                fh.write(json.dumps(record) + "\n")
                total += 1
            logger.info("Domain '%s': wrote %d examples.", domain_name, len(examples))

    logger.info("Total synthetic examples written: %d -> %s", total, output)
    return total


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build synthetic training JSONL.")
    p.add_argument("--n_per_domain", type=int, default=DEFAULT_N_PER_DOMAIN)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args()
    build_synthetic(n_per_domain=args.n_per_domain, seed=args.seed, output=args.output)
