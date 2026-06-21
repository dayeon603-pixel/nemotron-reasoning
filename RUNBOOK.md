# Nemotron Reasoning Challenge — 23-Hour Launch Runbook

Competition: `nvidia-nemotron-model-reasoning-challenge`
Deadline: 2026-06-15 (today). Submit before 23:59 UTC.

---

## Quick-reference column guide

| Step | What | Est. wall-clock | If it fails |
|------|------|-----------------|-------------|
| 0 | Accounts + secrets | 10 min | See §0 fallbacks |
| 1 | modal token new | 2 min | See §1 fallbacks |
| 2 | modal run (full pipeline) | 2.5–4 hr | See §2 fallbacks |
| 3 | Local CV check | 5 min | See §3 fallbacks |
| 4 | kaggle competitions submit | 2 min | See §4 fallbacks |

---

## Step 0 — Accounts and secrets (10 min)

You need three credentials. All are free.

**Kaggle API token**
1. Go to https://www.kaggle.com/settings → "Create New API Token"
2. Move the downloaded `kaggle.json` to `~/.kaggle/` and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Accept competition rules at https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules

**Hugging Face token**
1. Go to https://huggingface.co/settings/tokens → "New token" (read scope is enough)
2. Accept the Nemotron model license at https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

**Download competition data (needed for CV check in Step 3)**
```bash
cd /Users/chloekang/Documents/nemotron-reasoning
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/
unzip data/*.zip -d data/raw/
```

If `kaggle` command not found: `pip install kaggle`

---

## Step 1 — Modal setup (2 min)

```bash
pip install modal
modal token new
```

This opens a browser to authorize your Modal account. After that:

```bash
# Create the secret (fill in your real values):
modal secret create nemotron-secrets \
  KAGGLE_USERNAME="<your_kaggle_username>" \
  KAGGLE_KEY="<your_kaggle_key>" \
  HF_TOKEN="<your_hf_read_token>"
```

Verify the secret exists:
```bash
modal secret list
# Should show: nemotron-secrets
```

Check your free credit balance at https://modal.com/settings/billing — new accounts get $30/mo. An A100-80GB costs ~$2.50/hr. A full run (model download cached + 2-epoch SFT) takes roughly 2–3 hr = ~$5–8.

**If Modal credit is zero**: skip to §Fallback — vast.ai at the bottom.

---

## Step 2 — Launch the full pipeline (2.5–4 hr wall-clock)

One command. Run from the repo root:

```bash
cd /Users/chloekang/Documents/nemotron-reasoning
modal run scripts/run_modal.py
```

What this does (logged to your terminal in real time):
1. Provisions A100-80GB on Modal (falls back to H100 if A100 unavailable).
2. Installs all pip deps inside the container (torch, transformers, peft, vllm, mamba_ssm, etc.).
3. Downloads `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (~60 GB) via kagglehub into the persistent volume named `nemotron-model-cache`. **On re-runs this step is skipped** — kagglehub detects the cache.
4. Uploads your local repo (src/, configs/, scripts/, tests/) to the container at `/repo`.
5. Runs `scripts/build_synthetic.py --n_per_domain 500 --seed 42` → `data/synthetic.jsonl` (2000 examples across 4 domains, CPU only, ~30 sec).
6. Symlinks `data/synthetic.jsonl` to `data/accepted.jsonl` (what `sft_train.py` expects by default).
7. Runs `python -m src.sft_train --config configs/train.yaml` → trains LoRA rank-32 adapter for 2 epochs on A100-80GB (~2–3 hr). Saves `outputs/lora_adapter/best/` and `submission.zip`.
8. Downloads `submission.zip` + `outputs/lora_adapter_best.tar.gz` back to your local repo root.

**Estimated wall-clock breakdown:**
- Container boot + pip install: ~8 min (first run) / ~2 min (cached image)
- Model download: ~20–25 min (first run) / ~0 min (volume cached)
- Synthetic data build: ~30 sec
- SFT training (2 epochs, 2000 examples, rank-32, bf16): ~90–150 min on A100-80GB
- Artifact download: ~2 min

Total first run: ~2.5–3.5 hr. Subsequent runs (model cached): ~1.5–2.5 hr.

**If A100-80GB is unavailable**: Modal automatically retries on H100. If neither is available within 5 min, see §Fallback.

**If the run crashes mid-training**: The volume retains the model cache. Re-run the same command — model download is skipped, training resumes from scratch (no checkpoint resumption currently). Expect ~1.5 hr on re-run.

**If `nemotron-secrets` is not found**: You skipped Step 1. Run:
```bash
modal secret create nemotron-secrets KAGGLE_USERNAME="..." KAGGLE_KEY="..." HF_TOKEN="..."
```

---

## Step 3 — Local CV check (5 min)

After Step 2 completes, `submission.zip` and `outputs/lora_adapter_best/` are on your local disk.

**Validate the adapter and zip:**
```bash
cd /Users/chloekang/Documents/nemotron-reasoning
python scripts/package_submission.py \
  --adapter-dir outputs/lora_adapter_best \
  --zip-path submission.zip
```

Expected output includes:
```
Adapter validated: dir=outputs/lora_adapter_best  rank=32  ...
submission.zip layout OK: submission.zip
All checks passed. Ready to submit: ...
```

**Run the local test suite to confirm nothing is broken:**
```bash
python -m pytest -q
```

All existing tests should pass. The new `tests/test_package_submission.py` tests the validation logic.

**Optional — taxonomy recon (tells you coverage gaps, ~10 sec):**
```bash
python -m src.recon.taxonomy --train-csv data/raw/train.csv
```
Look for `COVERAGE GAPS` in the output. If `uncovered` rows are large (>20%), consider bumping `--n_per_domain` and re-running Step 2 (but check time budget first).

**If validate fails with rank > 32**: Re-check `configs/train.yaml` — `lora_r` must be <= 32.

**If `outputs/lora_adapter_best/` is missing**: The tar download in Step 2 may have failed (non-fatal). The `submission.zip` is still valid. Run:
```bash
python scripts/package_submission.py --zip-path submission.zip
```
(Without `--adapter-dir` pointing at a local dir — skip the adapter-dir check if you only have the zip.)

---

## Step 4 — Kaggle submission (2 min)

```bash
cd /Users/chloekang/Documents/nemotron-reasoning

kaggle competitions submit \
  -c nvidia-nemotron-model-reasoning-challenge \
  -f submission.zip \
  -m "LoRA rank-32 SFT on 2000 synthetic examples (4 domains x 500), 2 epochs, bf16 A100"
```

Then check your leaderboard position:
```bash
kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge
```

Score appears within ~10 min (NVIDIA runs inference on their side).

**If `kaggle` command not found**: `pip install kaggle` then retry.

**If submission fails with "file too large"**: Your zip is > 100 MB. This shouldn't happen for a rank-32 LoRA (expect ~50–200 MB). If it does:
```bash
python scripts/package_submission.py \
  --adapter-dir outputs/lora_adapter_best \
  --zip-path submission.zip \
  --repackage
```
This re-zips with only the two required files.

**If submission fails with "rules not accepted"**: Go to the competition Rules tab and accept. Then retry.

---

## Fallback — vast.ai (if Modal credit is zero)

Expected cost: ~$3–8 total for a full run.

1. Go to https://vast.ai and create an account.
2. Search for: GPU = A100 80GB SXM, disk >= 120 GB, CUDA >= 12.1.
3. Rent the instance (expect $0.80–1.30/hr). Note the SSH address.
4. SSH in and run:

```bash
# On the vast.ai instance:
git clone https://github.com/dayeon603-pixel/<your-repo> nemotron-reasoning
cd nemotron-reasoning
pip install torch transformers>=4.46 peft>=0.9 accelerate vllm>=0.12 \
    mamba_ssm causal_conv1d kagglehub polars pandas pyyaml safetensors numpy pytest

# Set credentials:
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"<YOUR_KAGGLE_USERNAME>","key":"<YOUR_KAGGLE_KEY>"}
EOF
chmod 600 ~/.kaggle/kaggle.json

export HF_TOKEN="<YOUR_HF_TOKEN>"
export PYTHONPATH=$(pwd)

# Build synthetic data:
python scripts/build_synthetic.py --n_per_domain 500 --seed 42 --output data/synthetic.jsonl
ln -s data/synthetic.jsonl data/accepted.jsonl

# Train:
python -m src.sft_train --config configs/train.yaml
```

5. After training, download `submission.zip` back:
```bash
scp -P <PORT> root@<HOST>:/path/to/nemotron-reasoning/submission.zip .
```

6. Continue from Step 3 locally.

---

## Common failure modes and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Adapter is a SILENT NO-OP` | Wrong `lora_target_regex` | Check `configs/train.yaml` — must target `q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj`, NOT `in_proj/out_proj` |
| `FileNotFoundError: data/accepted.jsonl` | Build step didn't run or symlink failed | Re-run `build_synthetic.py` and check symlink exists |
| OOM on A100-80GB | Batch or sequence too large | Reduce `max_length` to 2048 or `per_device_batch` to 1 in `configs/train.yaml` |
| `No module named mamba_ssm` | mamba_ssm not installed | `pip install mamba_ssm causal_conv1d` (requires CUDA; must run on GPU box) |
| `kaggle: command not found` | kaggle CLI not installed | `pip install kaggle` |
| Modal timeout during model download | Slow interconnect | Increase `MODEL_DOWNLOAD_TIMEOUT_S` in `scripts/run_modal.py` to `45 * 60` and re-run |
| submission.zip > 100 MB | Extra files in zip | Run `package_submission.py --repackage` |
| Leaderboard score = 0 | Model outputs `\boxed{STUB}` or empty | Check training loss converged; if loss stayed at init, retrain |
