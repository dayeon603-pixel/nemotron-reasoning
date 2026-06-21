r"""LoRA SFT training script for Nemotron Reasoning Challenge.

HARDWARE NOTE:
  The base model (nemotron-3-nano-30b-a3b BF16) requires approximately 60-80 GB
  of GPU VRAM.  A single A100-80GB or two A100-40GB (tensor parallelism) is the
  minimum viable setup.  This script does NOT fit on Kaggle's free-tier T4/P100.
  You must use a cloud GPU (e.g. Lambda Labs A100, RunPod, GCP A2) or a Kaggle
  notebook with two A100s enabled via the competition's GPU quota.

LoRA config notes:
  target_modules = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

  WHY NOT in_proj/out_proj (the submission demo's default): those are the Mamba-2
  projections.  Mamba-2 passes out_proj.weight straight to custom CUDA kernels and
  never calls out_proj.forward(), so a PEFT LoRA on it is silently ignored at train
  time (HF PEFT issue #2274).  vLLM's NemotronHForCausalLM also does NOT register
  in_proj in its LoRA allowlist (unlike Jamba), so even a trained Mamba adapter is a
  no-op at SCORING time.  The demo "works" only because it ships an UNTRAINED adapter
  (B=0 == identity), which never tests whether those modules train or load.

  This hybrid has only 6 attention layers / 52, so attention capacity is thin — the
  MLP projections (gate/up/down) carry most of the adapter.  After get_peft_model,
  ALWAYS run _assert_adapter_changes_output() (called below) to prove the adapter is
  not a silent no-op before spending GPU hours.  Open vLLM bug #42008 can corrupt MoE
  base-model queries under LoRA — at inference route only through the named adapter.

Usage:
    python -m src.sft_train \
        --config configs/train.yaml

The script expects either:
  (a) data/accepted.jsonl from rejection_sample.py  (STaR traces), or
  (b) a synthetic JSONL built from the generators.
Each line: {"prompt": "...", "trace": "...", "gold_answer": "..."}
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import shutil
import zipfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

__all__ = ["TrainConfig", "main"]

logger = logging.getLogger(__name__)

# ── seed utility ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.info("Global seed set to %d", seed)


# ── config dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All training hyperparameters, loaded from YAML.

    Attributes:
        model_kaggle_id:      Kaggle dataset ID for the base model.
        data_jsonl:           Path to training JSONL (prompt + trace pairs).
        output_dir:           Directory to save LoRA adapter.
        submission_zip:       Path for final submission.zip.
        seed:                 Global RNG seed.
        lora_r:               LoRA rank (must be <= 32 per challenge rules).
        lora_alpha:           LoRA scaling factor.
        lora_dropout:         LoRA dropout probability.
        lora_target_modules:  Explicit list of LoRA target module suffixes.
                              Takes priority over lora_target_regex when present.
        lora_target_regex:    Regex for target module names; used when
                              lora_target_modules is absent/empty (legacy fallback).
        max_length:           Maximum sequence length in tokens.
        per_device_batch:     Per-device training batch size.
        grad_accum_steps:     Gradient accumulation steps.
        num_epochs:           Number of training epochs.
        lr:                   AdamW learning rate.
        weight_decay:         AdamW weight decay.
        warmup_ratio:         Fraction of steps for LR warmup.
        max_grad_norm:        Gradient clipping max norm.
        bf16:                 Use bfloat16 mixed precision.
        logging_steps:        Log every N steps.
        save_steps:           Save checkpoint every N steps.
    """

    model_kaggle_id: str = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    data_jsonl: str = "data/accepted.jsonl"
    output_dir: str = "outputs/lora_adapter"
    submission_zip: str = "submission.zip"

    seed: int = 42

    # LoRA — DO NOT change lora_r above 32 (challenge constraint)
    lora_r: int = 32
    lora_alpha: int = 64  # scale = lora_alpha / lora_r = 2.0
    lora_dropout: float = 0.05
    # Attention-only targets: guaranteed loadable by vLLM's NemotronH LoRA
    # allowlist. MoE expert layers (gate/up/down_proj) may not load in the
    # scorer's vLLM build — keep commented out until smoke-tested.
    # lora_target_modules takes priority over lora_target_regex when non-empty.
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # Legacy regex fallback — used only when lora_target_modules is empty.
    # ACTIVE (attention-only):
    lora_target_regex: str = r".*\.(q_proj|k_proj|v_proj|o_proj)$"
    # ALTERNATE (attn+MLP) — only if vLLM smoke test confirms it loads:
    #   MoE-expert LoRA may not load in scorer's vLLM.
    # lora_target_regex: str = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    max_length: int = 7680  # 7680 < scorer max_tokens=7680; leaves boxed-answer headroom
    per_device_batch: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 3
    lr: float = 8e-6        # safe for rank-32 LoRA on 30B; 1e-4 is ~10x too high
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05  # 5% warmup for small-dataset SFT
    max_grad_norm: float = 1.0
    bf16: bool = True
    # Gradient checkpointing: saves activation memory (needed on 80GB GPUs for a
    # large MLP LoRA) but ~2x slower. On a 141GB H200 it can be disabled for
    # speed. Default True for safety.
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    save_steps: int = 100


def _load_config(yaml_path: Path) -> TrainConfig:
    """Load TrainConfig from a YAML file, overriding dataclass defaults.

    Args:
        yaml_path: Path to configs/train.yaml.

    Returns:
        Populated TrainConfig.

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If lora_r > 32 or lora_target_regex has been altered.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    _field_names = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in _field_names})

    if cfg.lora_r > 32:
        raise ValueError(
            f"lora_r={cfg.lora_r} exceeds the challenge maximum of 32. "
            "Set lora_r <= 32 in configs/train.yaml."
        )

    # Guard against regressing to the silent-no-op trap: Mamba-2 in_proj/out_proj
    # are not trainable via PEFT (kernel bypass) nor loadable in vLLM's NemotronH
    # LoRA allowlist. Refuse to target them via either config path.
    bad_modules = [m for m in cfg.lora_target_modules if re.search(r"in_proj|out_proj", m)]
    if bad_modules:
        raise ValueError(
            f"lora_target_modules contains Mamba-2 in_proj/out_proj ({bad_modules}), "
            "which produce a SILENT no-op adapter (PEFT issue #2274; not in vLLM "
            "NemotronH LoRA allowlist). Use attention/MLP projections only."
        )
    if not cfg.lora_target_modules and re.search(r"in_proj|out_proj", cfg.lora_target_regex):
        raise ValueError(
            "lora_target_regex targets Mamba-2 in_proj/out_proj, which produce a "
            "SILENT no-op adapter (PEFT issue #2274; not in vLLM NemotronH LoRA "
            "allowlist). Use attention + MLP projections, e.g.\n"
            r"  r'.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'"
        )

    return cfg


# ── dataset ───────────────────────────────────────────────────────────────────

class _SFTDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    """SFT dataset from accepted.jsonl.

    Each item is a dict with keys "input_ids" and "labels".
    Expects a pre-tokenised cache at <jsonl_path>.cache/ to avoid re-tokenising
    on every run; delete the cache dir to force re-tokenisation.

    Args:
        jsonl_path: Path to accepted.jsonl.
        tokenizer:  Chat-aware HF tokenizer for nemotron-3-nano-30b-a3b.
        max_length: Max sequence length.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        import json as _json

        from src.trace_format import render_sft_pair

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {jsonl_path}. "
                "Run src/rejection_sample.py or build synthetic data first."
            )

        self._items: list[dict[str, list[int]]] = []
        skipped = 0

        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = _json.loads(line)
                try:
                    pair = render_sft_pair(
                        example_prompt=record["prompt"],
                        gold_cot=record["trace"],
                        tokenizer=tokenizer,
                        max_length=max_length,
                    )
                    self._items.append(pair)
                except Exception as exc:
                    logger.warning("Skipping malformed record: %s", exc)
                    skipped += 1

        logger.info(
            "Loaded %d SFT items from %s (%d skipped)",
            len(self._items),
            jsonl_path,
            skipped,
        )
        if not self._items:
            raise RuntimeError(
                f"All {skipped} records from {jsonl_path} were skipped — "
                "no trainable examples. Check trace length vs max_length and "
                "the JSONL schema before spending GPU time."
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._items[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def _collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
    label_ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Right-pad input_ids; fill labels with label_ignore_index.

    Args:
        batch:              List of dicts with "input_ids" and "labels".
        pad_token_id:       Token ID used to pad input_ids.
        label_ignore_index: Value used to mask padded label positions.

    Returns:
        Batched dict with padded tensors and attention_mask.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(
            torch.nn.functional.pad(
                item["input_ids"], (0, pad_len), value=pad_token_id
            )
        )
        labels_list.append(
            torch.nn.functional.pad(
                item["labels"], (0, pad_len), value=label_ignore_index
            )
        )
        attention_mask_list.append(
            torch.cat(
                [torch.ones(seq_len, dtype=torch.long),
                 torch.zeros(pad_len, dtype=torch.long)]
            )
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


# ── LoRA model builder ────────────────────────────────────────────────────────

def _build_lora_model(cfg: TrainConfig) -> tuple[Any, Any]:
    """Load the base model from kagglehub and wrap with PEFT LoRA.

    Args:
        cfg: TrainConfig with model_kaggle_id and LoRA hyperparameters.

    Returns:
        Tuple of (peft_model, tokenizer).

    Raises:
        ImportError: If kagglehub, transformers, or peft are not installed.
        RuntimeError: If no target modules match the lora_target_regex.
    """
    import os as _os
    from pathlib import Path as _Path

    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    # Prefer a local model directory (set by the Modal runner via env, or if
    # model_kaggle_id is itself a local path). kagglehub is only a last resort
    # because it is unusable in some pinned environments.
    env_path = _os.environ.get("NEMOTRON_MODEL_PATH")
    if env_path and _Path(env_path).is_dir():
        model_path = env_path
        logger.info("Loading base model from NEMOTRON_MODEL_PATH: %s", model_path)
    elif _Path(cfg.model_kaggle_id).is_dir():
        model_path = cfg.model_kaggle_id
        logger.info("Loading base model from local path: %s", model_path)
    else:
        import kagglehub  # type: ignore[import]

        logger.info("Downloading base model via kagglehub: %s", cfg.model_kaggle_id)
        model_path = kagglehub.model_download(cfg.model_kaggle_id)
        logger.info("Base model downloaded to %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token (%s)", tokenizer.eos_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Log the model's actual unique leaf-module names so you can confirm the
    # target names exist for THIS checkpoint (they vary across releases).
    leaf_names = sorted(
        {name.rsplit(".", 1)[-1] for name, m in model.named_modules()
         if len(list(m.children())) == 0 and name}
    )
    logger.info("unique leaf module names: %s", leaf_names)

    # Resolve the set of unique leaf-name suffixes for PEFT.
    # PEFT LoraConfig.target_modules must be a list of name SUFFIXES (or
    # "all-linear"), NOT a regex string and NOT full dotted paths — it matches
    # against the last component of each named module.
    if cfg.lora_target_modules:
        # Explicit list supplied in config: use directly (already leaf suffixes).
        target_module_suffixes: list[str] = sorted(set(cfg.lora_target_modules))
        logger.info(
            "LoRA target_modules from explicit list: %s",
            target_module_suffixes,
        )
    else:
        # Fallback: resolve via regex over named modules, then extract suffixes.
        target_pattern = re.compile(cfg.lora_target_regex)
        matched_paths: list[str] = [
            name
            for name, _ in model.named_modules()
            if target_pattern.fullmatch(name)
        ]
        if not matched_paths:
            raise RuntimeError(
                f"No model modules matched lora_target_regex={cfg.lora_target_regex!r}. "
                "Check that the model was loaded correctly and the regex is right."
            )
        # Convert full dotted paths to leaf suffixes for PEFT.
        target_module_suffixes = sorted(
            {name.rsplit(".", 1)[-1] for name in matched_paths}
        )
        logger.info(
            "LoRA will target %d unique suffixes (from %d matched paths) via regex %r: %s",
            len(target_module_suffixes),
            len(matched_paths),
            cfg.lora_target_regex,
            target_module_suffixes,
        )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_module_suffixes,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Gradient checkpointing: recompute activations in the backward pass instead
    # of storing them. Needed to fit a large (MLP/MoE) LoRA on an 80GB GPU, but
    # ~2x slower; can be turned off on a 141GB H200 for speed (cfg flag).
    if cfg.gradient_checkpointing:
        peft_model.config.use_cache = False
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled (use_cache=False).")
    else:
        logger.info("Gradient checkpointing DISABLED (relying on GPU memory headroom).")

    _assert_adapter_changes_output(peft_model, tokenizer)
    return peft_model, tokenizer


def _assert_adapter_changes_output(
    peft_model: Any,
    tokenizer: Any,
    probe: str = "In Alice's Wonderland, decode 01000011.",
) -> None:
    """Fail fast if the LoRA adapter is a silent no-op.

    Perturbs the adapter's B matrices (normally zero-init => identity) with a tiny
    random delta, then checks that enabling vs. disabling the adapter changes the
    next-token logits. If logits are identical, the targeted modules are not on the
    forward path (the Mamba in_proj/out_proj trap) and any training would be wasted.

    Args:
        peft_model: The get_peft_model-wrapped model.
        tokenizer: Matching tokenizer.
        probe: A short prompt used only to elicit a forward pass.

    Raises:
        RuntimeError: If the adapter does not affect the model's logits.
    """
    import torch as _torch
    from peft.tuners.lora import LoraLayer

    perturbed = 0
    with _torch.no_grad():
        for module in peft_model.modules():
            if isinstance(module, LoraLayer):
                # lora_B is a ModuleDict of nn.Linear (zero-init => identity).
                for lin in module.lora_B.values():  # type: ignore[attr-defined]
                    lin.weight.add_(_torch.randn_like(lin.weight) * 1e-2)
                    perturbed += 1
    if perturbed == 0:
        raise RuntimeError(
            "No LoRA layers were created — target_modules matched nothing on the "
            "forward path. Check the logged leaf module names and fix the regex."
        )

    # Use the embedding layer's device — safe under both single-GPU and
    # device_map="auto" (sharded) layouts. peft_model.device is unreliable
    # when parameters live on multiple GPUs.
    embed_device = peft_model.get_input_embeddings().weight.device
    ids = tokenizer(probe, return_tensors="pt").input_ids.to(embed_device)
    peft_model.eval()
    with _torch.no_grad():
        peft_model.enable_adapter_layers()
        on = peft_model(ids).logits[:, -1, :].float()
        peft_model.disable_adapter_layers()
        off = peft_model(ids).logits[:, -1, :].float()
        peft_model.enable_adapter_layers()

    max_delta = (on - off).abs().max().item()
    logger.info("adapter logit delta (perturbed): max|Δ|=%.3e over %d LoRA layers",
                max_delta, perturbed)
    if max_delta < 1e-6:
        raise RuntimeError(
            "Adapter is a SILENT NO-OP: perturbing LoRA weights did not change "
            "logits. The targeted modules are off the forward path (likely the "
            "Mamba in_proj/out_proj trap). Re-target attention/MLP and re-run."
        )
    # Restore identity init so training starts clean.
    with _torch.no_grad():
        for module in peft_model.modules():
            if isinstance(module, LoraLayer):
                for lin in module.lora_B.values():  # type: ignore[attr-defined]
                    lin.weight.zero_()


# ── training loop ─────────────────────────────────────────────────────────────

def _train(cfg: TrainConfig) -> None:
    """Execute the full SFT training loop.

    Args:
        cfg: Fully-populated TrainConfig.
    """
    set_seed(cfg.seed)

    model, tokenizer = _build_lora_model(cfg)

    dataset = _SFTDataset(
        jsonl_path=Path(cfg.data_jsonl),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    pad_id: int = tokenizer.pad_token_id  # type: ignore[assignment]
    # pin_memory requires a background thread worker to work correctly;
    # with num_workers=0 it is a no-op and triggers a UserWarning in PyTorch >= 2.1.
    _num_workers: int = 0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.per_device_batch,
        shuffle=True,
        collate_fn=lambda b: _collate_fn(b, pad_token_id=pad_id),
        num_workers=_num_workers,
        pin_memory=(_num_workers > 0),  # False when num_workers==0
    )

    # Prefer 8-bit Adam (bitsandbytes) to shrink optimizer state ~4x (fp32 m+v ->
    # int8). Critical headroom for a large MLP/MoE LoRA on a 30B in 80 GB. Falls
    # back to torch AdamW if bitsandbytes is unavailable.
    _trainable = filter(lambda p: p.requires_grad, model.parameters())
    try:
        import bitsandbytes as _bnb  # type: ignore[import]

        optimizer = _bnb.optim.AdamW8bit(
            _trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        logger.info("Using bitsandbytes AdamW8bit optimizer (low memory).")
    except Exception as _bnb_exc:  # noqa: BLE001
        logger.warning("bitsandbytes unavailable (%s); using torch AdamW.", _bnb_exc)
        optimizer = torch.optim.AdamW(
            _trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    total_steps = math.ceil(len(dataloader) / cfg.grad_accum_steps) * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    from transformers import get_cosine_schedule_with_warmup  # type: ignore[import]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the device where input embeddings live — safe under both
    # single-GPU and device_map="auto" (sharded) configurations.
    # next(model.parameters()).device is WRONG when params span multiple GPUs.
    embed_device: torch.device = model.get_input_embeddings().weight.device

    global_step = 0
    best_loss: float = float("inf")
    # Initialize best_ckpt so it is always bound, even if no epoch improves loss.
    best_ckpt: Path = output_dir / "best"

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pending_grads: bool = False  # tracks whether a partial window exists

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(embed_device)
            labels = batch["labels"].to(embed_device)
            attention_mask = batch["attention_mask"].to(embed_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * cfg.grad_accum_steps
            pending_grads = True

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pending_grads = False
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch + 1,
                        global_step,
                        epoch_loss / (batch_idx + 1),
                        scheduler.get_last_lr()[0],
                    )

                if global_step % cfg.save_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-step{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    logger.info("Checkpoint saved: %s", ckpt_path)

        # Flush any remaining gradients from the final partial accumulation window.
        if pending_grads:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            logger.info(
                "epoch=%d flushed partial accumulation window at step=%d",
                epoch + 1,
                global_step,
            )

        avg_loss = epoch_loss / len(dataloader)
        logger.info("Epoch %d complete. avg_loss=%.4f", epoch + 1, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = output_dir / "best"
            model.save_pretrained(str(best_ckpt))
            logger.info("Best checkpoint updated: loss=%.4f -> %s", best_loss, best_ckpt)

    # Always save the last-epoch adapter (WER / loss may plateau early; last
    # weights are needed to confirm vs. best checkpoint on eval).
    last_ckpt = output_dir / "last"
    model.save_pretrained(str(last_ckpt))
    logger.info("Last-epoch adapter saved: %s", last_ckpt)

    _package_submission(best_ckpt, Path(cfg.submission_zip))


# ── submission packaging ──────────────────────────────────────────────────────

def _package_submission(adapter_dir: Path, zip_path: Path) -> None:
    """Package the LoRA adapter into submission.zip.

    The zip must contain at its root:
      - adapter_config.json
      - adapter_model.safetensors  (or sharded: adapter_model.safetensors.index.json
        + adapter_model-NNNNN-of-NNNNN.safetensors, ...)

    save_pretrained() may shard weights into multiple *.safetensors files with an
    accompanying *.index.json.  This function zips everything present rather than
    assuming a single monolithic file.

    Args:
        adapter_dir: Directory produced by model.save_pretrained().
        zip_path:    Output zip file path.

    Raises:
        FileNotFoundError: If adapter_config.json or no *.safetensors files exist.
    """
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}. "
            "Ensure model.save_pretrained() completed successfully."
        )

    safetensors_files = sorted(adapter_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"No *.safetensors files found in {adapter_dir}. "
            "Ensure peft is producing safetensors output (safetensors>=0.4)."
        )

    # Collect all files to zip: config + all safetensors + any shard index files.
    files_to_zip: list[Path] = [config_file] + safetensors_files
    files_to_zip += sorted(adapter_dir.glob("*.safetensors.index.json"))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_zip:
            zf.write(fpath, arcname=fpath.name)
            logger.info("  added to zip: %s (%d bytes)", fpath.name, fpath.stat().st_size)

    logger.info(
        "submission.zip created at %s (%d bytes, %d files)",
        zip_path,
        zip_path.stat().st_size,
        len(files_to_zip),
    )


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for sft_train.

    Args:
        argv: Argument list (defaults to sys.argv if None).
    """
    parser = argparse.ArgumentParser(
        description="LoRA SFT training for Nemotron Reasoning Challenge."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to YAML training config.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    cfg = _load_config(args.config)
    logger.info("Training config: %s", cfg)
    _train(cfg)


if __name__ == "__main__":
    main()
