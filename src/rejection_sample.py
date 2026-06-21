"""STaR (Self-Taught Reasoner) rejection sampling loop.

Reads train.csv (cols: id, prompt, answer), samples K traces from the base
model per prompt, keeps traces whose extracted answer passes verify(), and
writes accepted (prompt, trace) pairs to data/accepted.jsonl.

The actual model.generate call is isolated behind ModelInterface so the
loop runs end-to-end once a GPU + loaded model is available.

Usage:
    python -m src.rejection_sample \
        --train_csv data/train.csv \
        --output    data/accepted.jsonl \
        --k         8 \
        --seed      42

The --dry_run flag skips the model entirely (useful for pipeline smoke tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src.eval.metric import extract_final_answer, verify

__all__ = [
    "ModelInterface",
    "StubModelInterface",
    "run_rejection_sampling",
]

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_K: int = 8                  # samples per prompt
DEFAULT_MAX_NEW_TOKENS: int = 3500  # keep reasoning within scoring budget
DEFAULT_TEMPERATURE: float = 0.7    # exploration during STaR
DEFAULT_TOP_P: float = 0.9


# ── model interface ───────────────────────────────────────────────────────────

class ModelInterface(ABC):
    """Abstract base for model generation backends.

    Subclass this to wire a real vLLM / HF model without touching the
    rejection-sampling loop.
    """

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate one completion per prompt.

        Args:
            prompts:        List of fully-formatted prompt strings.
            max_new_tokens: Maximum tokens to generate per prompt.
            temperature:    Sampling temperature.
            top_p:          Nucleus sampling probability cutoff.

        Returns:
            List of completion strings, same length as prompts.
        """


@dataclass(slots=True)
class StubModelInterface(ModelInterface):
    """Deterministic stub used for smoke tests and --dry_run mode.

    Returns a completion that always contains ``\\boxed{STUB}`` so that
    the rejection loop can run without a live model.  Passes verify() only
    when the gold answer happens to equal "STUB" (i.e., never in practice),
    so no fake data reaches accepted.jsonl.
    """

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        logger.warning(
            "StubModelInterface.generate called — returning placeholder outputs. "
            "Wire a real model before producing accepted.jsonl for training."
        )
        return [
            f"Let me think about this.\n\\boxed{{STUB}}" for _ in prompts
        ]


# ── CSV reader ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TrainRow:
    """One row from train.csv."""
    id: str
    prompt: str
    answer: str


def _read_train_csv(csv_path: Path) -> list[TrainRow]:
    """Parse train.csv with columns id, prompt, answer.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of TrainRow objects.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    import csv  # stdlib — no pandas dependency for this simple read

    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {csv_path}")

    rows: list[TrainRow] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"id", "prompt", "answer"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"train.csv must have columns {required}; "
                f"found {reader.fieldnames}"
            )
        for raw in reader:
            rows.append(TrainRow(id=raw["id"], prompt=raw["prompt"], answer=raw["answer"]))

    logger.info("Loaded %d rows from %s", len(rows), csv_path)
    return rows


# ── prompt formatter ──────────────────────────────────────────────────────────

def _format_prompt_for_model(raw_prompt: str) -> str:
    """Append the competition boxed-answer instruction to a raw prompt.

    This replicates the exact user_content construction from the scoring
    notebook so that model inputs during STaR match evaluation inputs.

    Args:
        raw_prompt: The puzzle prompt string (from train.csv).

    Returns:
        The full user_content string (NOT yet chat-template wrapped).

    NOTE: apply_chat_template is NOT called here because the ModelInterface
    is expected to handle it (or the caller passes pre-templated strings).
    If you use vLLM with a chat endpoint, call apply_chat_template before
    passing to ModelInterface.generate.
    """
    boxed_instruction = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    return raw_prompt + boxed_instruction


# ── rejection sampling loop ───────────────────────────────────────────────────

@dataclass(slots=True)
class AcceptedTrace:
    """One accepted (prompt, trace) pair written to accepted.jsonl."""
    row_id: str
    prompt: str
    trace: str
    extracted_answer: str
    gold_answer: str


def _iter_batches(items: list[TrainRow], batch_size: int) -> Iterator[list[TrainRow]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_rejection_sampling(
    train_csv: Path,
    output_jsonl: Path,
    model: ModelInterface,
    k: int = DEFAULT_K,
    seed: int = 42,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    batch_size: int = 8,
) -> int:
    """Run the STaR rejection-sampling loop.

    For each prompt in train_csv, samples k traces from ``model``, extracts
    the boxed answer from each trace, checks it against the gold via verify(),
    and appends accepted traces to output_jsonl.

    Args:
        train_csv:      Path to train.csv (cols: id, prompt, answer).
        output_jsonl:   Path to write accepted traces (appended if exists).
        model:          ModelInterface implementation.
        k:              Number of samples per prompt.
        seed:           RNG seed (affects sampling order, not model weights).
        max_new_tokens: Per-completion token budget.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling cutoff.
        batch_size:     Prompts per model.generate() call.

    Returns:
        Number of accepted traces written.

    Raises:
        FileNotFoundError: If train_csv does not exist.
        ValueError:         If train_csv is malformed.
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = _read_train_csv(train_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    accepted_total = 0

    with output_jsonl.open("a", encoding="utf-8") as out_fh:
        for row in rows:
            formatted_prompt = _format_prompt_for_model(row.prompt)
            batch_prompts = [formatted_prompt] * k

            accepted_for_row = 0

            for batch in _iter_batches(
                [TrainRow(id=row.id, prompt=formatted_prompt, answer=row.answer)] * k,
                batch_size,
            ):
                completions = model.generate(
                    prompts=[b.prompt for b in batch],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                for completion in completions:
                    extracted = extract_final_answer(completion)
                    if verify(row.answer, extracted):
                        trace_record = AcceptedTrace(
                            row_id=row.id,
                            prompt=row.prompt,
                            trace=completion,
                            extracted_answer=extracted,
                            gold_answer=row.answer,
                        )
                        out_fh.write(
                            json.dumps(
                                {
                                    "id": trace_record.row_id,
                                    "prompt": trace_record.prompt,
                                    "trace": trace_record.trace,
                                    "extracted_answer": trace_record.extracted_answer,
                                    "gold_answer": trace_record.gold_answer,
                                }
                            )
                            + "\n"
                        )
                        accepted_for_row += 1
                        accepted_total += 1

            logger.info(
                "Row %s: %d/%d traces accepted",
                row.id,
                accepted_for_row,
                k,
            )

    logger.info(
        "Rejection sampling complete. Total accepted: %d / %d",
        accepted_total,
        len(rows) * k,
    )
    return accepted_total


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STaR rejection sampling for Nemotron Reasoning Challenge."
    )
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/accepted.jsonl"))
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Use StubModelInterface — no GPU required, no real traces produced.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args()

    model_impl: ModelInterface
    if args.dry_run:
        logger.warning("--dry_run active: using StubModelInterface.")
        model_impl = StubModelInterface()
    else:
        # TODO(gpu): Import and instantiate your real ModelInterface here.
        # Example (vLLM):
        #   from src.vllm_model import VLLMModelInterface
        #   model_impl = VLLMModelInterface(model_id="metric/nemotron-3-nano-30b-a3b-bf16/...")
        raise NotImplementedError(
            "Wire a real ModelInterface before running without --dry_run. "
            "See src/rejection_sample.py TODO(gpu) comment."
        )

    n_accepted = run_rejection_sampling(
        train_csv=args.train_csv,
        output_jsonl=args.output,
        model=model_impl,
        k=args.k,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
    logger.info("Done. Accepted traces written: %d", n_accepted)
