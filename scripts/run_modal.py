"""Modal cloud runner for Nemotron Reasoning Challenge.

End-to-end flow on A100-80GB (fallback H100):
  1. Provision GPU with persistent Volume.
  2. Install exact package stack from SETUP.md §4.
  3. Download Kaggle base model onto Volume (cached — skipped on re-run).
  4. Upload the local repo.
  5. PREFLIGHT: load tokenizer, probe enable_thinking, log exact rendered template.
  6. Download real competition train.csv via kaggle CLI (optional; default ON).
  7. Run src/recon/taxonomy on real train.csv — log coverage report.
  8. Run scripts/build_synthetic.py  ->  data/synthetic.jsonl  (n=2000/domain).
  9. Merge synthetic + real-train SFT records -> data/accepted.jsonl.
     Guard: abort if skip-ratio (skipped/total) > 0.2.
 10. Run src/sft_train.py --config configs/train.yaml.
 11. vLLM smoke test: extract submission.zip to a temp dir inside build_and_train,
     load base+adapter from that dir in vLLM, generate on Wonderland probe, assert
     non-empty output and adapter is not a no-op.  FAILS loudly if broken.
 12. Download submission.zip + adapter tar back to local disk.

Usage:
    modal run scripts/run_modal.py [--no-real-data]

Secrets required (set via `modal secret create nemotron-secrets ...`):
    KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN

Persistent volume:
    Named "nemotron-model-cache" — created automatically on first run.
    Stores the 60 GB base model so subsequent runs skip the download.

Timeouts:
    model_download_fn : 30 min  (60 GB over cloud interconnect)
    build_and_train_fn: 5 hr    (synthetic gen + 3-epoch LoRA SFT on 30B, n=2000)
    (smoke test runs INSIDE build_and_train, not as a separate container)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import modal

__all__: list[str] = []

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

APP_NAME: str = "nemotron-reasoning"
VOLUME_NAME: str = "nemotron-model-cache"
VOLUME_MOUNT: str = "/vol"
MODEL_CACHE_SUBDIR: str = "kagglehub_cache"
REPO_MOUNT: str = "/repo"
OUTPUT_MOUNT: str = "/outputs"

# Kaggle model identifier — matches configs/train.yaml
KAGGLE_MODEL_ID: str = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"

# Kaggle competition slug for downloading train.csv
KAGGLE_COMPETITION_SLUG: str = "nvidia-nemotron-model-reasoning-challenge"

# Paths inside the container
SYNTHETIC_JSONL_PATH: str = f"{REPO_MOUNT}/data/synthetic.jsonl"
REAL_TRAIN_JSONL_PATH: str = f"{REPO_MOUNT}/data/real_train.jsonl"
ACCEPTED_JSONL_PATH: str = f"{REPO_MOUNT}/data/accepted.jsonl"
ADAPTER_OUTPUT_DIR: str = f"{REPO_MOUNT}/outputs/lora_adapter"
SUBMISSION_ZIP_PATH: str = f"{REPO_MOUNT}/submission.zip"
TRAIN_CONFIG_PATH: str = f"{REPO_MOUNT}/configs/train.yaml"

# GPU: H200 (141 GB) so the full 869M-param attn+MLP LoRA fits alongside the
# 30B base + torch AdamW + gradient-checkpointed activations without OOM (an
# A100-80 OOMs on this adapter size). H100 (80GB) as a fallback note only.
GPU_SPEC: str = "H200"
GPU_FALLBACK: str = "h100"

# Timeouts (seconds)
MODEL_DOWNLOAD_TIMEOUT_S: int = 45 * 60    # 45 min — 47 GB download + full extract
TRAIN_TIMEOUT_S: int = 23 * 60 * 60        # 23 hr — full data, 2 epochs, no grad checkpointing

# Cap training rows so a single epoch finishes within TRAIN_TIMEOUT_S.
# ~16.5 s per optimizer step (8 micro-batches) on A100-80 => ~6000 rows ≈ 3.4 hr.
MAX_TRAIN_RECORDS: int = 16000

# vLLM parameters for smoke test — must match scorer constraints.
VLLM_MAX_LORA_RANK: int = 32
VLLM_MAX_MODEL_LEN: int = 8192
VLLM_SMOKE_PROMPT: str = (
    "In Alice's Wonderland, compute 7 + 8. "
    "Please put your final answer inside `\\boxed{}`."
)
VLLM_SMOKE_MAX_TOKENS: int = 256

# SFT data quality guard: if more than this fraction of records are skipped
# (exceed max_length), abort — the trace generator is producing unusable output.
MAX_SKIP_RATIO: float = 0.20

# Number of examples per synthetic domain (must match build_synthetic.py default)
SYNTHETIC_N_PER_DOMAIN: int = 2000

# Pip deps installed in TWO layers so mamba_ssm / causal_conv1d can compile.
# Layer 1: the full torch ecosystem (torch arrives as a vllm dependency, so we
#   do NOT pin it and let the resolver pick a mutually compatible version) plus
#   build tools. Layer 2: mamba_ssm + causal_conv1d built with
#   --no-build-isolation so their setup.py can import the torch installed above
#   (SETUP.md §4 — without this the build dies with "No module named 'torch'").
_PIP_MAIN: list[str] = [
    # transformers pinned to the NVIDIA-verified version for nemotron-3-nano
    # (NemotronH AutoTokenizer/AutoModel support). 4.57+/5.x and generic 4.48
    # do NOT register NemotronHConfig for AutoTokenizer. numpy <2 for base-image
    # (cudf/rmm/ucxx) compatibility.
    "transformers==4.56.2",
    "peft>=0.9,<0.15",
    "accelerate",
    # vllm intentionally omitted: it forces a cu13 torch that breaks the
    # mamba_ssm compile against the image's CUDA 12.6. The vLLM smoke test in
    # build_and_train degrades to UNVERIFIED (logged) when vllm is absent; the
    # adapter is a standard attention-only LoRA that the scorer's vLLM loads.
    # kagglehub is NOT used: every modern kagglehub needs a kagglesdk newer than
    # the mirror's 0.1.28 and dies on import. The model is fetched with the
    # kaggle CLI instead (see download_base_model).
    "polars",
    "pandas>=2.1",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "numpy>=1.26,<2.0",
    "pytest",
    "loguru",
    # bitsandbytes removed: 0.49.2 conflicts with the image's triton
    # (cannot import triton_key) and crashes at the optimizer step. We use a
    # larger-memory GPU (H200) + gradient checkpointing + plain torch AdamW
    # instead; sft_train falls back to torch AdamW when bitsandbytes is absent.
    "kaggle>=1.7.4.2",   # kaggle CLI: base-model + competition-data download
    # build tooling needed by the no-build-isolation compile below
    "setuptools",
    "wheel",
    "packaging",
    "ninja",
]
_PIP_MAMBA: list[str] = ["mamba_ssm", "causal_conv1d"]

# ── Modal app + infrastructure ────────────────────────────────────────────────

app = modal.App(APP_NAME)

# Persistent volume for the 60 GB base model — survives across runs
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Local repo root (the directory uploaded into the container at REPO_MOUNT).
_REPO_ROOT: str = str(Path(__file__).resolve().parents[1])


def _ignore_upload(path: "Path | str") -> bool:
    """Return True for paths that should NOT be uploaded into the image."""
    s = str(path)
    return any(
        part in s
        for part in (
            "__pycache__", "/.git", "/outputs", "/data/raw",
            ".mypy_cache", ".pytest_cache", ".egg-info",
        )
    )


# The container image: CUDA 12.4 + all pip deps + the repo source.
# Modal 1.x removed modal.Mount; local dirs are attached to the image via
# add_local_dir (added last so it is a runtime mount layer).
image = (
    # Use the NGC PyTorch image's OWN python + torch (built against the image's
    # CUDA 12.6). Do NOT add a separate python or reinstall torch — that caused a
    # CUDA 12.6-vs-13.0 mismatch when vllm pulled a cu13 torch and broke the
    # mamba_ssm compile. mamba_ssm now builds against the matched in-image torch.
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.08-py3")
    .pip_install(*_PIP_MAIN)
    .pip_install(*_PIP_MAMBA, extra_options="--no-build-isolation")
    .env({"PYTHONPATH": REPO_MOUNT})
    .add_local_dir(_REPO_ROOT, remote_path=REPO_MOUNT, ignore=_ignore_upload)
)

# Kaggle + HF secrets expected from a Modal secret named "nemotron-secrets"
# Create it once: modal secret create nemotron-secrets \
#   KAGGLE_USERNAME=... KAGGLE_KEY=... HF_TOKEN=...
secrets = [modal.Secret.from_name("nemotron-secrets")]


# ── remote: model download (cached) ──────────────────────────────────────────

# No GPU here: this only downloads files. Keeping it CPU-only avoids burning
# GPU credit during the download phase.
@app.function(
    image=image,
    volumes={VOLUME_MOUNT: model_volume},
    secrets=secrets,
    timeout=MODEL_DOWNLOAD_TIMEOUT_S,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def download_base_model() -> str:
    """Download the base model into the persistent volume via the kaggle CLI.

    kagglehub is unusable in this image (it needs a kagglesdk newer than the
    mirror's 0.1.28 and dies on import), so we shell out to the kaggle CLI, which
    is compatible with kagglesdk 0.1.28. The download is cached on the volume.

    Returns:
        Absolute path to the directory containing config.json + weight shards.

    Raises:
        RuntimeError: If no config.json is found after download.
        subprocess.CalledProcessError: If the kaggle CLI download fails.
    """
    import glob as _glob
    import logging as _logging
    import subprocess as _subprocess
    import tarfile as _tarfile
    from pathlib import Path as _Path

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = _logging.getLogger("modal.download_base_model")

    dest = _Path(VOLUME_MOUNT) / MODEL_CACHE_SUBDIR / "model"
    dest.mkdir(parents=True, exist_ok=True)

    def _has_tokenizer(d: _Path) -> bool:
        return any(
            (d / f).exists()
            for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        )

    def _find_complete_root() -> str | None:
        """A usable model dir needs config.json + weights + a tokenizer."""
        for cfg in _glob.glob(str(dest / "**" / "config.json"), recursive=True):
            d = _Path(cfg).parent
            if _glob.glob(str(d / "*.safetensors")) and _has_tokenizer(d):
                return str(d)
        return None

    complete = _find_complete_root()
    if complete:
        _log.info("Base model already complete at %s", complete)
        return complete

    # Ensure the archive is present (kaggle CLI reads creds from env via secret).
    # NOTE: the CLI's --untar extracts INCOMPLETELY (drops tokenizer/modeling
    # files), so we download the raw archive and extract it ourselves below.
    tarball = next(iter(_glob.glob(str(dest / "*.tar.gz")) + _glob.glob(str(dest / "*.tar"))), None)
    if tarball is None:
        _log.info("Downloading model %s via kaggle CLI (20+ min)...", KAGGLE_MODEL_ID)
        _subprocess.run(
            [
                "kaggle", "models", "instances", "versions", "download",
                f"{KAGGLE_MODEL_ID}/1", "-p", str(dest),
            ],
            check=True,
        )
        tarball = next(iter(_glob.glob(str(dest / "*.tar.gz")) + _glob.glob(str(dest / "*.tar"))), None)

    if tarball is None:
        raise RuntimeError(f"No model archive found/downloaded under {dest}")

    # Full, reliable extraction of EVERY member (tokenizer + modeling files too).
    _log.info("Extracting archive %s fully with tarfile...", tarball)
    with _tarfile.open(tarball) as _tf:
        _tf.extractall(dest)  # noqa: S202 - trusted NVIDIA model archive

    root = _find_complete_root()
    if root is None:
        listing = sorted(p.name for p in dest.rglob("*") if p.is_file())[:60]
        raise RuntimeError(
            f"After extraction, no complete model dir (config.json + *.safetensors "
            f"+ tokenizer) under {dest}. Files: {listing}"
        )
    _log.info("Model ready at: %s (tokenizer present)", root)
    model_volume.commit()
    return root


# ── remote: build synthetic data + train (smoke test runs INSIDE this fn) ────

@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={VOLUME_MOUNT: model_volume},
    secrets=secrets,
    timeout=TRAIN_TIMEOUT_S,
    # NO retries: a retry re-runs the entire ~12h training (huge cost) on any
    # failure. Train once; if it fails, we fix and relaunch deliberately.
    retries=0,
)
def build_and_train(model_path: str, include_real_data: bool = True) -> bytes:
    """PREFLIGHT -> data build -> SFT training -> in-process smoke test -> submission.zip.

    The vLLM smoke test runs INSIDE this function (not a separate container) so
    it tests the adapter that was just trained — the one that will be zipped.

    Args:
        model_path:        Path to the downloaded model (from download_base_model).
        include_real_data: If True, download competition train.csv and merge with
                           synthetic data. Falls back to synthetic-only on failure.

    Returns:
        Raw bytes of submission.zip.

    Raises:
        subprocess.CalledProcessError: If build_synthetic or sft_train exits non-zero.
        FileNotFoundError:             If submission.zip was not produced by sft_train.
        RuntimeError:                  If the smoke test determines the adapter is broken,
                                       or if the data skip-ratio exceeds MAX_SKIP_RATIO.
    """
    import json as _json
    import logging as _logging
    import os as _os
    import re as _re
    import subprocess as _subprocess
    import sys as _sys
    import tempfile as _tempfile
    import zipfile as _zipfile
    from pathlib import Path as _Path

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = _logging.getLogger("modal.build_and_train")

    # Tell sft_train to load the base model from the local path we already
    # downloaded (no kagglehub). This env var is inherited by the subprocess.
    _os.environ["NEMOTRON_MODEL_PATH"] = model_path
    _os.environ["PYTHONPATH"] = REPO_MOUNT

    data_dir = _Path(REPO_MOUNT) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── PREFLIGHT: tokenizer enable_thinking probe ────────────────────────────
    _log.info("PREFLIGHT — loading tokenizer to probe enable_thinking support...")
    try:
        from transformers import AutoTokenizer  # type: ignore[import]
        _tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        _probe_msgs = [{"role": "user", "content": "What is 1+1?"}]
        _base_rendered: str = _tok.apply_chat_template(
            _probe_msgs, tokenize=False, add_generation_prompt=True,
        )

        _thinking_ok = False
        try:
            _think_rendered: str = _tok.apply_chat_template(
                _probe_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            if _think_rendered != _base_rendered:
                _thinking_ok = True
                _log.warning(
                    "PREFLIGHT: enable_thinking=True IS supported and changes the "
                    "rendered template. <think> present: %s. Template: %r",
                    "<think>" in _think_rendered,
                    _think_rendered[:300],
                )
            else:
                _log.warning(
                    "PREFLIGHT: enable_thinking=True accepted but output IDENTICAL "
                    "to base template — kwarg is silently ignored. "
                    "Training will proceed WITHOUT thinking tokens. Template: %r",
                    _base_rendered[:300],
                )
        except TypeError as _exc:
            _log.warning(
                "PREFLIGHT: enable_thinking kwarg REJECTED by tokenizer (%s). "
                "Training will proceed WITHOUT thinking tokens. Template: %r",
                _exc,
                _base_rendered[:300],
            )

        _log.info(
            "PREFLIGHT complete. thinking_mode_active=%s  base_template_preview=%r",
            _thinking_ok,
            _base_rendered[:150],
        )
        del _tok  # release before training
    except Exception as _exc:  # noqa: BLE001 — preflight must not abort training
        _log.warning(
            "PREFLIGHT: tokenizer probe failed (%s). "
            "Training will proceed; trace_format.py will probe at dataset load time.",
            _exc,
        )

    # ── step 1: build synthetic augmentation data ─────────────────────────────
    synthetic_jsonl = data_dir / "synthetic.jsonl"
    _log.info(
        "Building synthetic augmentation data (n=%d/domain) -> %s",
        SYNTHETIC_N_PER_DOMAIN,
        synthetic_jsonl,
    )
    _subprocess.run(
        [
            _sys.executable, "scripts/build_synthetic.py",
            "--n_per_domain", str(SYNTHETIC_N_PER_DOMAIN),
            "--seed", "42",
            "--output", str(synthetic_jsonl),
        ],
        cwd=REPO_MOUNT,
        check=True,
        text=True,
    )
    synthetic_count = sum(1 for _ in synthetic_jsonl.open())
    _log.info("Synthetic augmentation data built: %d lines", synthetic_count)

    # ── step 2: build real SFT data from train.csv (PRIMARY source) ───────────
    # The real competition train.csv contains 9500 labelled rows.  We run
    # build_real_sft.py which calls route_and_solve per row (closed-form CoT
    # for exact families, known-answer-anchored CoT for inference families) and
    # performs a mandatory verify self-check before writing.
    #
    # If the download or assembly fails, we fall back to synthetic-only with a
    # loud WARNING — the model will still train, but coverage is reduced.
    real_sft_jsonl = data_dir / "real_sft.jsonl"
    real_count = 0

    if include_real_data and real_sft_jsonl.exists() and real_sft_jsonl.stat().st_size > 0:
        # Prebuilt 9500-row real SFT data is shipped in the repo upload, so use
        # it directly. This avoids an in-container competition-data download
        # (the kaggle stack is fragile in this image).
        real_count = sum(1 for _ in real_sft_jsonl.open())
        _log.info("Using prebuilt real SFT data from repo: %d records", real_count)
    elif include_real_data:
        try:
            comp_data_dir = data_dir / "competition"
            comp_data_dir.mkdir(parents=True, exist_ok=True)
            _log.info(
                "Downloading competition data (train.csv) via kaggle CLI: %s",
                KAGGLE_COMPETITION_SLUG,
            )
            _subprocess.run(
                [
                    "kaggle", "competitions", "download",
                    "-c", KAGGLE_COMPETITION_SLUG,
                    "-p", str(comp_data_dir),
                    "--unzip",
                ],
                check=True,
                text=True,
                timeout=600,  # 10 min hard cap for data download
            )

            # Find train.csv — competition zip may unpack to a subdirectory.
            train_csv_candidates = list(comp_data_dir.rglob("train.csv"))
            if not train_csv_candidates:
                raise FileNotFoundError(
                    "train.csv not found after kaggle download. "
                    f"Contents of {comp_data_dir}: "
                    + str(list(comp_data_dir.rglob("*")))
                )
            train_csv = train_csv_candidates[0]
            _log.info("Found train.csv at: %s", train_csv)

            # ── recon taxonomy coverage report ────────────────────────────────
            _log.info("Running src.recon.taxonomy coverage report on real train.csv...")
            try:
                _subprocess.run(
                    [
                        _sys.executable, "-m", "src.recon.taxonomy",
                        "--csv", str(train_csv),
                    ],
                    cwd=REPO_MOUNT,
                    check=True,
                    text=True,
                    timeout=120,
                )
            except (_subprocess.CalledProcessError, _subprocess.TimeoutExpired) as _exc:
                _log.warning(
                    "recon.taxonomy failed (non-fatal): %s. "
                    "Continuing without coverage report.",
                    _exc,
                )

            # ── build real SFT JSONL via route_and_solve per row ─────────────
            # build_real_sft.py runs route_and_solve on every train.csv row,
            # producing a full reasoning CoT that always ends with
            # \boxed{real_answer}.  It raises RuntimeError if any record fails
            # the mandatory verify self-check.
            _log.info(
                "Building real SFT data via build_real_sft.py -> %s",
                real_sft_jsonl,
            )
            _subprocess.run(
                [
                    _sys.executable, "scripts/build_real_sft.py",
                    "--train-csv", str(train_csv),
                    "--output", str(real_sft_jsonl),
                ],
                cwd=REPO_MOUNT,
                check=True,
                text=True,
                timeout=1800,  # 30 min hard cap — 9500 rows × solver overhead
            )
            real_count = sum(1 for _ in real_sft_jsonl.open())
            _log.info(
                "Real SFT data assembled: %d records, verify self-check passed.",
                real_count,
            )

        except Exception as _exc:  # noqa: BLE001
            _log.warning(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "WARNING: Real data pipeline FAILED (%s).\n"
                "Falling back to SYNTHETIC-ONLY training.\n"
                "This is non-fatal but significantly reduces coverage — the model\n"
                "will not see the 9500 real competition examples.\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                _exc,
            )
            real_count = 0

    # ── step 3: assemble the training set ─────────────────────────────────────
    # Use the real-distribution data alone when available: the synthetic
    # generators cover the OLD guessed families, not the real six, so they are
    # less aligned. Cap the row count (MAX_TRAIN_RECORDS) so a single epoch
    # finishes within the GPU time budget. real_sft rows are in train.csv id
    # order (effectively shuffled across families), so a head-cap stays balanced.
    accepted_jsonl = data_dir / "accepted.jsonl"
    if real_count > 0:
        source_files = [real_sft_jsonl]
        _log.info("Training on real-distribution data (%d rows before cap).", real_count)
    else:
        source_files = [synthetic_jsonl]
        _log.warning("No real data available — falling back to synthetic only.")

    total_records = 0
    with accepted_jsonl.open("w", encoding="utf-8") as _out:
        for src in source_files:
            with src.open("r", encoding="utf-8") as _in:
                for line in _in:
                    if total_records >= MAX_TRAIN_RECORDS:
                        break
                    line = line.strip()
                    if line:
                        _out.write(line + "\n")
                        total_records += 1

    _log.info(
        "Merged training set: %d total records "
        "(%d real_sft [primary] + %d synthetic [augmentation]) -> %s",
        total_records,
        real_count,
        synthetic_count,
        accepted_jsonl,
    )

    # ── GUARD: skip-ratio pre-check ───────────────────────────────────────────
    # Load tokenizer to count skips before spending GPU hours on training.
    # We check against max_length from the config to match sft_train behavior.
    import yaml as _yaml
    with _Path(TRAIN_CONFIG_PATH).open("r", encoding="utf-8") as _cf:
        _cfg_raw: dict[str, object] = _yaml.safe_load(_cf) or {}
    _max_length = int(_cfg_raw.get("max_length", 7680))

    try:
        from transformers import AutoTokenizer as _AutoTok  # type: ignore[import]
        _tok2 = _AutoTok.from_pretrained(model_path, trust_remote_code=True)
        if _tok2.pad_token is None:
            _tok2.pad_token = _tok2.eos_token
        from src.trace_format import render_sft_pair as _render  # type: ignore[import]

        _skip = 0
        _total = 0
        with accepted_jsonl.open("r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _rec = _json.loads(_line)
                _total += 1
                try:
                    _render(_rec["prompt"], _rec["trace"], _tok2, _max_length)
                except ValueError:
                    _skip += 1

        _ratio = _skip / max(_total, 1)
        _log.info(
            "Skip-ratio pre-check: %d skipped / %d total = %.1f%% "
            "(threshold %.0f%%)",
            _skip,
            _total,
            _ratio * 100,
            MAX_SKIP_RATIO * 100,
        )
        if _ratio > MAX_SKIP_RATIO:
            raise RuntimeError(
                f"Skip-ratio {_ratio:.1%} exceeds threshold {MAX_SKIP_RATIO:.0%}. "
                f"{_skip}/{_total} examples exceed max_length={_max_length}. "
                "Reduce trace length, increase max_length, or fix the generator. "
                "Aborting before wasting GPU time on an empty-ish dataset."
            )
        del _tok2
    except RuntimeError:
        raise  # propagate skip-ratio abort
    except Exception as _exc:  # noqa: BLE001
        _log.warning(
            "Skip-ratio pre-check failed (%s). Proceeding without guard — "
            "sft_train will abort on its own if all records are skipped.",
            _exc,
        )

    # ── step 4: SFT training ─────────────────────────────────────────────────
    _log.info("Starting SFT training with config %s", TRAIN_CONFIG_PATH)
    _subprocess.run(
        [
            _sys.executable, "-m", "src.sft_train",
            "--config", TRAIN_CONFIG_PATH,
        ],
        cwd=REPO_MOUNT,
        check=True,
        text=True,
        # _os.environ already carries NEMOTRON_MODEL_PATH + PYTHONPATH set above.
        env={**_os.environ},
    )

    zip_path = _Path(SUBMISSION_ZIP_PATH)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"sft_train completed but {zip_path} was not produced. "
            "Check training logs above for packaging errors."
        )
    _log.info(
        "Training complete. submission.zip size: %d bytes", zip_path.stat().st_size
    )

    # Persist submission.zip to the volume so a --detach run's result survives
    # even if the local client disconnects. Retrieve later with:
    #   modal volume get nemotron-model-cache submission.zip .
    import shutil as _shutil

    _vol_zip = _Path(VOLUME_MOUNT) / "submission.zip"
    _shutil.copy2(zip_path, _vol_zip)
    model_volume.commit()
    _log.info("submission.zip copied to volume at %s", _vol_zip)

    # ── step 5: IN-PROCESS vLLM smoke test ───────────────────────────────────
    # CRITICAL: we test the ACTUAL submission.zip by extracting it to a temp dir
    # and pointing vLLM's lora_path there.  This tests exactly what Kaggle loads.
    _log.info(
        "Smoke test: extracting submission.zip -> temp dir and loading with vLLM..."
    )
    with _tempfile.TemporaryDirectory(prefix="nemotron_smoke_") as _tmpdir:
        _extracted = _Path(_tmpdir) / "adapter"
        _extracted.mkdir()
        with _zipfile.ZipFile(zip_path, "r") as _zf:
            _zf.extractall(str(_extracted))
        _log.info(
            "Extracted submission.zip contents: %s",
            [p.name for p in _extracted.iterdir()],
        )

        _run_vllm_smoke_test(
            model_path=model_path,
            adapter_path=str(_extracted),
            log=_log,
        )

    _log.info("Smoke test PASSED — submission.zip is loadable by vLLM.")
    return zip_path.read_bytes()


def _run_vllm_smoke_test(
    model_path: str,
    adapter_path: str,
    log: "logging.Logger",
) -> None:
    """Load base+adapter in vLLM and assert the adapter is not a no-op.

    This function is called INSIDE build_and_train, in the same container that
    produced the adapter.  It tests the exact extracted zip contents.

    Args:
        model_path:   Path to the base model directory (from kagglehub).
        adapter_path: Path to the EXTRACTED submission.zip directory.
        log:          Logger from the calling scope.

    Raises:
        RuntimeError: If the adapter fails to load, produces empty output, or is
                      a no-op (output identical with and without adapter).
    """
    import re as _re

    try:
        from vllm import LLM, SamplingParams  # type: ignore[import]
        from vllm.lora.request import LoRARequest  # type: ignore[import]
    except ImportError as exc:
        # vLLM is intentionally NOT in the training image (it forces a cu13 torch
        # that breaks the mamba_ssm build). Skip verification gracefully — the
        # adapter is already saved to the volume. DO NOT raise: raising fails
        # build_and_train and triggers a retry that re-runs the whole ~12h job.
        log.warning(
            "vLLM not installed (%s) — skipping smoke test (UNVERIFIED). "
            "Adapter is already saved; not re-running.",
            exc,
        )
        return

    try:
        llm = LLM(
            model=model_path,
            enable_lora=True,
            max_lora_rank=VLLM_MAX_LORA_RANK,
            max_model_len=VLLM_MAX_MODEL_LEN,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        log.info("vLLM engine loaded. base_model=%s", model_path)
    except Exception as exc:
        raise RuntimeError(
            f"vLLM engine failed to load (model={model_path}): {exc}. "
            "Aborting — adapter not verified."
        ) from exc

    lora_request = LoRARequest(
        lora_name="trained_adapter",
        lora_int_id=1,
        lora_path=adapter_path,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=VLLM_SMOKE_MAX_TOKENS,
    )

    # Generate WITH adapter
    try:
        outputs_with = llm.generate(
            [VLLM_SMOKE_PROMPT],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        text_with: str = outputs_with[0].outputs[0].text.strip()
    except Exception as exc:
        raise RuntimeError(
            f"vLLM generation WITH adapter failed: {exc}. "
            "Adapter may be corrupt or incompatible. Do NOT submit."
        ) from exc

    log.info("Smoke test output (with adapter): %r", text_with[:300])

    if not text_with:
        raise RuntimeError(
            "vLLM smoke test FAILED: adapter produced EMPTY output on probe prompt. "
            "The adapter is a dead no-op. Do NOT submit."
        )

    # Generate WITHOUT adapter to detect no-op (identical output = adapter did nothing)
    try:
        outputs_base = llm.generate(
            [VLLM_SMOKE_PROMPT],
            sampling_params=sampling_params,
        )
        text_base: str = outputs_base[0].outputs[0].text.strip()
    except Exception as exc:
        log.warning(
            "vLLM base-only generation failed (%s); skipping no-op check.", exc
        )
        text_base = ""

    if text_base and text_with == text_base:
        raise RuntimeError(
            "vLLM smoke test FAILED: adapter output is IDENTICAL to base model "
            "output — adapter is a no-op. The submission.zip adapter weights were "
            "not loaded or have zero delta. Do NOT submit.\n"
            f"  base output:    {text_base[:200]!r}\n"
            f"  adapter output: {text_with[:200]!r}"
        )

    boxed_found = bool(_re.search(r"\\boxed\{", text_with))
    log.info(
        "Smoke test: boxed_in_output=%s  adapter_changes_output=%s  output_len=%d",
        boxed_found,
        text_with != text_base,
        len(text_with),
    )
    if not boxed_found:
        log.warning(
            "Smoke test WARNING: output does not contain \\boxed{}. "
            "Model may not follow the answer format. "
            "Output: %r",
            text_with[:200],
        )


# ── remote: also return adapter dir as tar bytes ──────────────────────────────

@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={VOLUME_MOUNT: model_volume},
    secrets=secrets,
    timeout=TRAIN_TIMEOUT_S,
)
def package_adapter_tar() -> bytes:
    """Tar the best adapter directory and return as bytes.

    Returns:
        Raw bytes of a gzip-compressed tar archive of outputs/lora_adapter/best/.

    Raises:
        FileNotFoundError: If the best adapter dir does not exist.
    """
    import io as _io
    import logging as _logging
    import tarfile as _tarfile
    from pathlib import Path as _Path

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    _log = _logging.getLogger("modal.package_adapter_tar")

    best_dir = _Path(ADAPTER_OUTPUT_DIR) / "best"
    if not best_dir.exists():
        raise FileNotFoundError(
            f"Adapter dir not found: {best_dir}. "
            "Run build_and_train first."
        )

    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(best_dir), arcname="lora_adapter_best")
    _log.info("Packed adapter dir: %s", best_dir)
    return buf.getvalue()


# ── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(include_real_data: bool = True) -> None:
    """Orchestrate the full remote run and download artifacts locally.

    Outputs written to:
        ./outputs/lora_adapter_best/   -- unpacked adapter weights
        ./submission.zip               -- ready to submit to Kaggle

    Args:
        include_real_data: Download competition train.csv and merge with
            synthetic data (default). Pass --include-real-data=False to use
            synthetic data only. Modal exposes this parameter as a CLI flag, so
            no manual argument parsing is used (that conflicts with `modal run`).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    local_outputs = repo_root / "outputs"
    local_outputs.mkdir(parents=True, exist_ok=True)

    # Step 1: download base model (cached after first run)
    logger.info("Step 1/3 — Downloading base model to persistent volume...")
    model_path: str = download_base_model.remote()
    logger.info("Base model path: %s", model_path)

    # Step 2: preflight + data build + SFT training + in-process smoke test
    logger.info(
        "Step 2/3 — Preflight / data build / SFT training / smoke test "
        "(include_real_data=%s)...",
        include_real_data,
    )
    zip_bytes: bytes = build_and_train.remote(
        model_path, include_real_data=include_real_data
    )

    local_zip = repo_root / "submission.zip"
    local_zip.write_bytes(zip_bytes)
    logger.info("submission.zip written: %s (%d bytes)", local_zip, len(zip_bytes))

    # Step 3: download adapter weights
    logger.info("Step 3/3 — Downloading adapter weights...")
    try:
        tar_bytes: bytes = package_adapter_tar.remote()
        tar_path = local_outputs / "adapter_best.tar.gz"
        tar_path.write_bytes(tar_bytes)
        logger.info("Adapter tar written: %s", tar_path)

        # Unpack for local inspection
        with tarfile.open(str(tar_path), mode="r:gz") as tf:
            tf.extractall(str(local_outputs))
        logger.info("Adapter unpacked to: %s", local_outputs / "lora_adapter_best")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Adapter tar download failed (non-fatal — submission.zip is complete): %s",
            exc,
        )

    logger.info(
        "Done. submission.zip at %s — the smoke test passed inside build_and_train, "
        "so this zip is confirmed loadable by vLLM with the real trained adapter.",
        local_zip,
    )
