# What You Need — Procurement & Setup (free-first)

Goal: train a rank-≤32 LoRA adapter for `nemotron-3-nano-30b-a3b` and submit `submission.zip`.
Scoring runs on **NVIDIA's** hardware, so you do **not** need to run the 30B model for
submission — you only need compute to **train the adapter**. Going **pure-synthetic**
(generators already emit gold CoT) removes the rejection-sampling inference step, so the
*minimum* requirement is a single GPU for a few hours.

---

## 0. Accounts & keys (all free, ~10 min)

| Item | Where | Cost | Notes |
|---|---|---|---|
| Kaggle API token | kaggle.com/settings → "Create New API Token" → `kaggle.json` → `~/.kaggle/` | free | needed to download comp data / submit via CLI |
| Hugging Face token | huggingface.co/settings/tokens (read scope) | free | for the model mirror; accept the model license once |
| (rules accepted) | competition Rules tab | free | ✅ already done |

```bash
pip install kaggle kagglehub
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## 1. Data (free, ~3 MB)

| Item | Where | Cost |
|---|---|---|
| `train.csv` + `test.csv` (full) | `kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge` | free |
| Synthetic data | generated locally by `src/generators/` (CPU, no GPU) | free |

```bash
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/
unzip data/*.zip -d data/raw/
```
First job after download: run the **family-taxonomy recon** on the *full* train.csv — it
decides how much synthetic coverage you're missing (medal vs. contender).

---

## 2. Base model (free, ~60 GB — download on the GPU box, NOT your Mac)

| Source | Path | Cost |
|---|---|---|
| Kaggle (what the scorer uses) | `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")` | free |
| Hugging Face mirror | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | free (accept license) |

Use the **Kaggle** copy so your training base byte-matches the scoring base. ~60 GB —
download it directly on whichever GPU machine you train on; do not pull it to the MacBook.

---

## 3. Compute — the only real cost (tiers, cheapest first)

The 30B base in bf16 ≈ 60 GB → needs **80 GB GPU** (or 4-bit QLoRA to fit smaller).
A full rank-32 LoRA SFT run on a few-thousand-example set, 2 epochs ≈ **2–4 GPU-hours**.

| Tier | Source | GPU | Cost | Verdict |
|---|---|---|---|---|
| **0 — sponsored (try first)** | Competition's Google Cloud G4 credits / NVIDIA brev launchable (linked in comp resources); ask in the Nemotron Discord or community@nvidia.com | RTX PRO 6000 Blackwell 96 GB | **free if granted** | Best case — purpose-built for this comp. Claim it today. |
| **1 — Modal (recommended free-ish)** | modal.com — serverless, scales to zero | A100-80 / H100 on demand | **$30/mo free credit**, then ~$2.5/hr (A100-80) | New accounts get $30/mo free → ~12 A100-hrs = several full runs at $0. **The Progress-Prize winner used Modal.** |
| **2 — Kaggle free (fully free)** | Kaggle Notebooks, GPU on | 2×T4 (32 GB total) | **free, ~30 hr/wk** | Requires **4-bit QLoRA** to fit 30B. Slower; bitsandbytes 4-bit on this hybrid Mamba+MoE is unverified — test on 1 step first. |
| **3 — cheapest paid** | vast.ai (spot) | A100-80 / H100 | ~$0.8–1.3/hr A100-80 | Most reliable pure-paid. Whole comp ≈ **$20–50**. |
| **4 — paid alt** | runpod.io, lambdalabs.com | A100/H100 | ~$1.3–3/hr | Use if vast.ai capacity is out. |

**Recommended plan:** Tier 0 if granted → otherwise **Tier 1 (Modal, $30 free credit)** as the
working default → Tier 3 (vast.ai) only if you blow through credits. Realistic out-of-pocket: **$0–30**.

> Do NOT count on free Kaggle/Colab single T4/A100-40 holding 30B bf16 — they can't (40 GB < 60 GB).
> Either go 80 GB (Tiers 0/1/3) or 4-bit QLoRA on Tier 2.

---

## 4. Software (all free, pip — already on Kaggle/Modal images)

```
torch  transformers>=4.46  peft>=0.9  accelerate  bitsandbytes   # QLoRA only
vllm>=0.12  mamba_ssm  causal_conv1d  kagglehub  polars  pandas  pytest
```
- `peft>=0.9` for regex `target_modules` (else enumerate leaf names — see TODO in `sft_train.py`).
- `vllm>=0.12` is the version NVIDIA's guides pin for Nemotron inference.
- `mamba_ssm` + `causal_conv1d` are required to load the hybrid base at all.

---

## 5. Minimum-viable FREE path (if you want $0 total)

1. Kaggle CLI → download `train.csv`.
2. Local Mac (CPU) → run generators → `data/synthetic.jsonl` (pure synthetic, gold CoT). No model needed.
3. Modal **or** Kaggle 2×T4 → `sft_train.py` (4-bit QLoRA if on T4) → adapter.
   - `_assert_adapter_changes_output()` aborts if the adapter is a no-op before you spend the run.
4. Local CPU → `src/eval/` metric-exact CV on the held-out train slice.
5. Package `submission.zip` → upload. Scoring is on NVIDIA's side (free).

Inference compute = **$0** (skipped entirely with pure-synthetic). Only training uses GPU,
and Modal's $30/mo credit covers several runs.

---

## 6. Action checklist (today)

- [ ] Create Kaggle API token + HF token.
- [ ] `kaggle competitions download` → get the **full** train.csv.
- [ ] Ask in Nemotron Discord / email community@nvidia.com about **Google Cloud / brev credits** (Tier 0).
- [ ] Create a **Modal** account → confirm the $30 free credit (Tier 1 fallback).
- [ ] Run the family-taxonomy recon on full train.csv → decide synthetic coverage.
