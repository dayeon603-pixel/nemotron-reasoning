"""Proposal A experiment: solver-synthetic vs real data efficiency under QLoRA.

Runs on a single free 16GB GPU (Kaggle T4/P100 or Colab T4). It fine-tunes a
small open model with 4-bit QLoRA on two data sources (zero-noise solver-generated
data and real competition data) at several training-set sizes, then measures
held-out accuracy. The output is an accuracy-vs-budget table you can turn into the
"effective-token multiplier" and accuracy-per-dollar results in the proposal.

This is intentionally small so a first data point finishes in roughly 1 to 2 hours
on a free GPU. Scale N_TRAIN_BUDGETS and the model up later.

Run from the repo root (so `src` imports work):
    python experiments/proposal_a_solver_data_efficiency.py

Or in a Kaggle/Colab cell:
    !git clone https://github.com/dayeon603-pixel/nemotron-reasoning && cd nemotron-reasoning && pip install -q transformers peft bitsandbytes accelerate datasets && python experiments/proposal_a_solver_data_efficiency.py
"""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("proposalA")

# ── config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B"        # open, no gating, fits 4-bit on 16GB
FAMILY = "gravitational"               # has a 100%-correct solver in this repo
N_TRAIN_BUDGETS = [200, 500, 1000]     # training-set sizes to sweep
N_TEST = 200                           # held-out test size (solver-generated)
MAX_LEN = 512                          # traces are short
EPOCHS = 1
LR = 2e-4
SEED = 42
OUT = Path("experiments/proposal_a_results.json")


def _pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def ensure_deps() -> None:
    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        log.info("Installing dependencies...")
        _pip("transformers>=4.44", "peft>=0.11", "bitsandbytes>=0.43", "accelerate>=0.30", "datasets>=2.19")


def build_datasets() -> tuple[list[dict], list[dict], list[dict]]:
    """Return (solver_train_pool, real_train_pool, test_set) as {prompt, answer} dicts."""
    sys.path.insert(0, str(Path.cwd()))
    from src.solvers.exact import GravitationalSolver  # noqa: E402

    solver = GravitationalSolver()

    # Solver-synthetic pool (zero label noise) and a held-out test set.
    pool = [{"prompt": e.prompt, "answer": e.answer} for e in solver.generate(max(N_TRAIN_BUDGETS), seed=SEED)]
    test = [{"prompt": e.prompt, "answer": e.answer} for e in solver.generate(N_TEST, seed=SEED + 9999)]

    # Real pool from competition train.csv if present (else fall back to solver).
    real: list[dict] = []
    csv = next(iter(Path("data/raw").glob("train.csv")), None)
    if csv is not None:
        import csv as _csv

        with csv.open() as fh:
            for row in _csv.DictReader(fh):
                p = row.get("prompt", "")
                if p.lower().startswith("in alice's wonderland, the gravitational"):
                    real.append({"prompt": p, "answer": str(row.get("answer", "")).strip()})
        log.info("Loaded %d real gravitational rows from %s", len(real), csv)
    if not real:
        log.warning("No real train.csv found; 'real' arm will reuse a disjoint solver split.")
        real = [{"prompt": e.prompt, "answer": e.answer} for e in solver.generate(max(N_TRAIN_BUDGETS), seed=SEED + 1)]

    return pool, real, test


PROMPT_SUFFIX = "\nPut your final answer inside \\boxed{}."


def to_text(rec: dict) -> str:
    return f"{rec['prompt']}{PROMPT_SUFFIX} \\boxed{{{rec['answer']}}}"


def train_one(source_name: str, train_recs: list[dict], test: list[dict]) -> dict:
    import torch
    # NOTE: is_bf16_supported() returns True on a T4 via emulation, so check the
    # GPU compute capability directly. Native bf16 needs sm_80+ (Ampere). T4 is 7.5.
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16  # T4 has no bf16
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Dataset.from_list([{"text": to_text(r)} for r in train_recs])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), remove_columns=["text"])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    args = TrainingArguments(
        output_dir=f"experiments/_ckpt_{source_name}_{len(train_recs)}",
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS, learning_rate=LR, bf16=bf16_ok, fp16=not bf16_ok,
        logging_steps=20, save_strategy="no", report_to=[])
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    # Evaluate: greedy-generate, extract last \boxed{}, verify with the repo metric.
    from src.eval.metric import verify  # noqa: E402
    model.eval()
    correct = 0
    for r in test:
        ids = tok(r["prompt"] + PROMPT_SUFFIX, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=64, do_sample=False)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.findall(r"\\boxed\{([^}]*)\}", gen)
        pred = m[-1] if m else (re.findall(r"-?\d+\.?\d*", gen) or [""])[-1]
        if verify(r["answer"], pred):
            correct += 1
    acc = correct / len(test)
    log.info("[%s | n=%d] accuracy = %.3f", source_name, len(train_recs), acc)
    del model
    torch.cuda.empty_cache()
    return {"source": source_name, "n_train": len(train_recs), "accuracy": acc}


def main() -> None:
    random.seed(SEED)
    ensure_deps()
    solver_pool, real_pool, test = build_datasets()
    results = []
    for n in N_TRAIN_BUDGETS:
        results.append(train_one("solver_synthetic", solver_pool[:n], test))
        results.append(train_one("real", real_pool[:n], test))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
    log.info("\n=== RESULTS (accuracy vs training size) ===")
    for r in results:
        log.info("%-18s n=%-5d acc=%.3f", r["source"], r["n_train"], r["accuracy"])
    log.info("Saved -> %s", OUT)
    log.info("Next: plot accuracy vs n for each source; the horizontal gap at a "
             "fixed accuracy is the effective-data multiplier.")


if __name__ == "__main__":
    main()
