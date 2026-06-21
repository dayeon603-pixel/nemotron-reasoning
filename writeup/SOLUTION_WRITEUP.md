# NVIDIA Nemotron Model Reasoning Challenge — Solution Write-up
**Open Contribution Award submission: Best Data / Synthetic Data Method**

---

> **MUST FILL BEFORE SUBMISSION (deadline: June 22)**
>
> The following placeholders block reproducibility or will draw a disqualifying challenge from a reviewer. Fill every one before publishing.
>
> | Placeholder | Where to get it | Status |
> |---|---|---|
> | `<REPO_URL>` | Public Kaggle notebook or GitHub URL | BLOCKING: notebook Cell 3 will fail |
> | `<ADAPTER_PATH>` | Kaggle dataset path of uploaded trained adapter | BLOCKING: inference demo will not run |
> | `<EXACT_PACKAGE_VERSIONS>` | Run `pip freeze` in the training env and replace the `>=` bounds in §7.2 | BLOCKING: reproducibility claim is false without exact pins |
> | `<N_SYNTHETIC_TOTAL>` | Printed by `build_synthetic.py` at end of run (= 7 x n_per_domain) | Required for results table |
> | `<TRAIN_WALL_TIME>`, `<GPU_TYPE>`, `<TRAIN_LOSS_FINAL>` | Timer + instance info from training run | Required for results table |
> | `<BASELINE_LB_SCORE>` | Zero-shot Kaggle submission or competition leaderboard page | Required for delta claim |
> | `<CV_ACC>`, `<CV_CI_LOWER>`, `<CV_CI_UPPER>` | `run_cv()` output | Required for results table |
> | `<LB_SCORE>` | Kaggle public leaderboard after submission | Required for results table |
> | `<DELTA_VS_BASELINE>`, `<CV_LB_GAP>` | Computed from above | Required for results table |
> | Per-domain N and accuracy placeholders | `run_cv()` domain breakdown output | Required — recon table and figure are REQUIRED evidence |
> | `<FINAL_N_PER_DOMAIN>` | Final run config (`--n_per_domain` value used) | Required for results table |

---

## 1. Problem framing

The challenge asks models to solve inductive rule-discovery puzzles in the "Alice's Wonderland" format: given N demonstration pairs showing an input-output transformation, induce the rule and apply it to a held-out query. The official scorer (`verify`) is exact-match with three ordered branches:

1. Binary strings: strict length-sensitive string equality.
2. Numeric: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`.
3. Everything else: case-insensitive string equality.

This structure tells you exactly what the training signal needs to look like: the model must emit the right surface form of the answer, not just the right value. A binary answer with the wrong bit-width scores 0 even if the integer interpretation is correct.

**Core claim.** For tasks whose answer is computed by a closed-form rule (bit manipulation, Roman numeral codec, linear algebra, substitution cipher, number sequences, list operations, modular arithmetic), generating training data programmatically is strictly better than sampling from a teacher model:

- Zero label noise: the generator is the rule, so `gold_answer = generate(inputs)` is exact by construction.
- Zero inference cost: no teacher model is needed during data creation.
- Infinite coverage: any (n, rule variant) can be generated; distribution is under your control.
- CoT quality: chain-of-thought is templated from the rule logic itself, so every step is verifiable and consistent.

The tradeoff is coverage: generators only work for families whose rules are fully enumerable. The recon tool (`src/recon/taxonomy.py`) is what ensures that coverage assumption holds before any GPU time is spent.

---

## 2. Pipeline overview

```
train.csv
    |
    v
[recon] src/recon/taxonomy.py
    -- classify every row into a generator domain or "uncovered"
    -- flag format gaps and binary bit-width gaps
    -- emit recon.json + human-readable report
    |
    v
[gap analysis] review recon.json
    -- extend generators for any uncovered family with significant row count
    -- confirm bit-width distribution matches GENERATOR_BINARY_WIDTHS={8}
    |
    v
[generate] scripts/build_synthetic.py --n_per_domain 2000 --seed 42
    -- calls generate_binary_ops, generate_cipher, generate_linear_eq, generate_roman,
       generate_number_seq, generate_list_ops, generate_modular_arith
    -- writes data/synthetic.jsonl  (schema: id, prompt, trace, extracted_answer, gold_answer)
    |
    v
[SFT] src/sft_train.py --config configs/train.yaml
    -- rank-32 LoRA on nemotron-3-nano-30b-a3b BF16
    -- target modules: q/k/v/o_proj + gate/up/down_proj  (NOT Mamba in_proj/out_proj)
    -- 2 epochs, lr=1e-4, grad_accum=8, max_length=4096
    -- _assert_adapter_changes_output() aborts if adapter is a silent no-op
    |
    v
[local CV] src/eval/cv.py
    -- metric-exact replication of official verify()
    -- per-domain accuracy + bootstrap 95% CI (n_boot=1000, seed=42)
    |
    v
submission.zip  ->  Kaggle leaderboard
```

---

## 3. Family taxonomy and recon

### 3.1 Why recon first

The hidden test set shares the same rule families as the public train set, but the exact distribution is unknown. Running `taxonomy.py` on the full `train.csv` answers three questions before any GPU time is spent:

1. Which families are present, and in what proportion?
2. Do real answer formats match what the generators emit?
3. For binary families, do bit widths match? (`verify` compares binary answers as exact-length strings, so a width mismatch always scores 0.)

This converts "which families are covered?" from a post-hoc leaderboard discovery into a pre-compute quantitative audit. The recon output table and figure showing the domain distribution (including the "uncovered" bucket) are required evidence for this submission, not optional.

### 3.2 How it works

`classify_domain(prompt)` runs keyword heuristics in priority order:

```
roman       -> keywords: "roman"
binary_ops  -> keywords: "bit manipulation", "binary", "8-bit", "bitwise", "bits"
cipher      -> keywords: "encrypt", "cipher", "secret encryption", "decode", "decrypt", "shift", "substitut"
linear_eq   -> keywords: "equation", "solve for", "algebra", "linear", "value of x"
uncovered   -> catch-all
```

`classify_answer(answer)` routes answers into format buckets in a fixed-priority order (binary before int, to match how `verify` dispatches). The order matters: "1" and "101" are binary, not integers, per the official scorer.

`template_signature(prompt)` normalises variable content (binary tokens, numbers, arrows) so structurally distinct families produce distinct signatures even when their surface values differ.

`_detect_gaps()` then cross-checks:
- Any "uncovered" rows: unknown families needing new generators.
- Any domain whose real answer formats include formats outside `GENERATOR_EXPECTED_FORMATS[domain]`.
- Any binary domain where real bit-widths fall outside `GENERATOR_BINARY_WIDTHS = {8}`.

### 3.3 Running recon

```bash
# requires only stdlib; no GPU
python -m src.recon.taxonomy \
    --train-csv data/raw/train.csv \
    --json-out recon.json
```

The output explicitly flags every coverage gap with a human-readable `!` prefix. Fix every gap before generating data.

### 3.4 Classifier scope vs. generator scope

The keyword classifier in `taxonomy.py` (`GENERATOR_DOMAINS`) currently recognizes four families: binary_ops, cipher, linear_eq, roman. The generator suite in `build_synthetic.py` produces seven families, adding number_seq, list_ops, and modular_arith. When recon runs on the real train set, those three newer generator families will appear in the "uncovered" bucket because the classifier has no keywords for them yet. This does not mean the generators are wrong; it means the recon audit for those families requires inspecting the "uncovered" bucket directly (check template signatures and answer formats within it) until the classifier is extended with matching keywords. The recon output's uncovered-bucket breakdown is required evidence for the submission.

---

## 4. Generators

### 4.1 Design principles

Each generator in `src/generators/` is a pure function `generate(n: int, seed: int) -> list[Example]`. The `Example` dataclass carries four fields:

```python
@dataclass(slots=True)
class Example:
    prompt:   str   # full Wonderland-format puzzle
    answer:   str   # ground truth, exact match to verify()'s expectation
    domain:   Domain
    gold_cot: str   # full chain-of-thought ending in \boxed{answer}
```

Every generator:
- Uses `random.Random(seed)` with no global state mutation.
- Includes a self-consistency assertion: the generated `gold_cot` must end with `\boxed{answer}` exactly, enforced at generation time, not at training time.
- Covers multiple sub-families within one domain (e.g., `roman` covers `int_to_roman`, `roman_to_int`, `roman_add`, `roman_subtract`).

### 4.2 Roman numeral generator (representative example)

The roman generator (`src/generators/roman.py`) illustrates the pattern. Four rules are registered via `_RomanRule` dataclasses. The CoT template has five fixed steps:

1. Restate the demonstration pairs.
2. State the induced rule in natural language.
3. Verify the rule by applying it to every demo pair and checking against the expected output. Generation aborts with `ValueError` if any demo fails.
4. Apply the rule to the query input.
5. Emit `\boxed{answer}`.

The `_to_roman` and `_from_roman` functions are the source of truth; the CoT is derived from them, so the reasoning and the answer are guaranteed consistent. Domain-specific constraints are enforced at sampling time: for `roman_subtract`, `a > b` is guaranteed; for `roman_add`, `a + b <= 3999`.

### 4.3 Prompt surface form

All generators use `format_wonderland_prompt` from `src/generators/common.py`, which emits prompts in the competition's expected "Alice's Wonderland" surface form with a fixed `NUM_DEMO_PAIRS = 4` demonstrations per query. The header is a constant shared across all families, keeping the prompt distribution aligned with the real test set.

### 4.4 Data assembly

The generator suite covers seven domain families:

| Domain | Sub-families / notes | Sub-seed offset |
|---|---|---|
| `binary_ops` | AND, OR, XOR, NOT, shifts, two's complement (8-bit) | seed + 0 |
| `cipher` | substitution and shift ciphers (Caesar-style) | seed + 1000 |
| `linear_eq` | one-variable linear equations, integer and float solutions | seed + 2000 |
| `roman` | int_to_roman, roman_to_int, roman_add, roman_subtract [1, 3999] | seed + 3000 |
| `number_seq` | arithmetic, geometric, Fibonacci-style integer sequences | seed + 4000 |
| `list_ops` | sort, reverse, filter, map-style operations on integer lists | seed + 5000 |
| `modular_arith` | modular addition, multiplication, inverse, exponentiation | seed + 6000 |

```bash
python scripts/build_synthetic.py \
    --n_per_domain 2000 \
    --seed 42 \
    --output data/synthetic.jsonl
```

This writes 7 x n_per_domain examples total. Each family receives a deterministic sub-seed (offset listed above) so the families are independent but the entire dataset is reproducible from the single top-level seed.

**<!-- PLACEHOLDER --> Total examples in actual submitted run:** `<N_SYNTHETIC_TOTAL>` (= 7 x `<FINAL_N_PER_DOMAIN>`; fill after run).

The output schema matches `data/accepted.jsonl` (the rejection-sampling output format), so `sft_train.py` accepts either file without modification.

---

## 5. LoRA SFT

### 5.1 Base model

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` via Kaggle (`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`). Training uses the Kaggle copy to byte-match the scoring infrastructure.

### 5.2 LoRA configuration

| Parameter | Value | Rationale |
|---|---|---|
| `lora_r` | 32 | Challenge maximum; used in full. |
| `lora_alpha` | 16 | Standard 0.5x scaling. |
| `lora_dropout` | 0.05 | Light regularisation. |
| `target_modules` | `q/k/v/o_proj`, `gate/up/down_proj` | See below. |
| `max_length` | 4096 | Covers full CoT traces. |
| `num_epochs` | 2 | |
| `lr` | 1e-4 | AdamW. |
| `grad_accum` | 8 | Effective batch = 8. |
| `warmup_ratio` | 0.03 | |
| `max_grad_norm` | 1.0 | Gradient clipping. |
| `bf16` | true | |

**Critical: do not target Mamba `in_proj`/`out_proj`.** Nemotron-3-Nano-30B-A3B is a hybrid Mamba-2 + attention + MoE architecture. The model has only 6 attention layers out of 52 total. Mamba-2 passes `out_proj.weight` directly to custom CUDA kernels without calling `out_proj.forward()`, so a PEFT LoRA on those modules is silently ignored at train time (HF PEFT issue #2274). vLLM's `NemotronHForCausalLM` also does not register `in_proj` in its LoRA allowlist. The official competition submission demo uses these as default target modules, but only appears to work because it ships an untrained adapter (`B=0` is the identity transformation). This means every team using the official demo's default target modules silently trained a no-op adapter. The MLP projections (`gate/up/down_proj`) carry most of the effective adapter capacity given the low attention layer count.

The training script calls `_assert_adapter_changes_output()` before any training epoch. This function runs a single forward pass and asserts the adapter output differs from the frozen base output. The run aborts if the adapter is a no-op, preventing silent waste of GPU hours.

### 5.3 Training command

```bash
# Run on A100-80GB or equivalent (60 GB bf16 base + activations)
python -m src.sft_train --config configs/train.yaml
# OR with synthetic data:
python -m src.sft_train \
    --config configs/train.yaml \
    --data-jsonl data/synthetic.jsonl
```

**<!-- PLACEHOLDER --> Training wall time:** `<TRAIN_WALL_TIME>` on `<GPU_TYPE>`.
**<!-- PLACEHOLDER --> Final training loss:** `<TRAIN_LOSS_FINAL>`.

---

## 6. Evaluation

### 6.1 Metric-exact local CV

`src/eval/metric.py` is a verbatim replication of the official competition scorer, including the non-obvious behaviours:

- `extract_final_answer`: scans for `\boxed{` and returns the **last** non-empty match, not the first. The official docstring says "first" but the code does `boxed_answers[-1]`. Any local eval that takes the first `\boxed{}` will silently diverge from the leaderboard. This is a concrete, verifiable finding specific to this competition.
- `verify` on binary strings: length-sensitive string equality. `"10011000"` and `"00011000"` are different answers.
- `verify` on numerics: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`.
- `verify` on everything else: case-insensitive `.lower()` equality.

The metric implementation is exact. The distribution assumption is not: the local CV val split is drawn from the synthetic train distribution, while the leaderboard evaluates on a hidden test set whose family distribution and uncovered-family fraction are unknown. The CV score measures metric correctness and per-family accuracy on covered families. It does not predict the leaderboard score for uncovered families.

### 6.2 CV protocol

```bash
python -m src.eval.cv \
    --predictions outputs/predictions.csv \
    --solutions data/raw/train.csv \
    --n-boot 1000 \
    --seed 42
```

Output: overall accuracy + bootstrap 95% CI (percentile method, n=1000 resamples, seed=42) + per-domain breakdown including the "uncovered" bucket.

Domain inference in CV operates on the prompt text only, never on the answer or the prediction, so there is no lookahead leakage from the label into the domain bucket.

### 6.3 Results

**What we tried:** pure programmatic synthetic data (`<N_SYNTHETIC_TOTAL>` examples, 7 families, `<FINAL_N_PER_DOMAIN>` per family) + rank-32 LoRA SFT on nemotron-3-nano-30b-a3b BF16.

**What we measured:** metric-exact local CV accuracy on held-out train slice (20% split, stratified by domain; synthetic data has no temporal ordering); leaderboard public score.

**Baseline:** zero-shot nemotron-3-nano-30b-a3b with no adapter, public leaderboard score = `<BASELINE_LB_SCORE>` (fill after retrieving from competition page).

| Metric | Value |
|---|---|
| Local CV accuracy | `<CV_ACC>` |
| Bootstrap 95% CI | [`<CV_CI_LOWER>`, `<CV_CI_UPPER>`] (n=1000) |
| Leaderboard public score | `<LB_SCORE>` |
| delta vs. zero-shot baseline | `<DELTA_VS_BASELINE>` |
| CV-to-LB gap | `<CV_LB_GAP>` |

Per-domain breakdown (REQUIRED evidence; fill from `run_cv()` output; include uncovered bucket):

| Domain | N (val) | CV Accuracy |
|---|---|---|
| binary_ops | `<N_BINARY>` | `<ACC_BINARY>` |
| cipher | `<N_CIPHER>` | `<ACC_CIPHER>` |
| linear_eq | `<N_ALGEBRA>` | `<ACC_ALGEBRA>` |
| roman | `<N_ROMAN>` | `<ACC_ROMAN>` |
| number_seq | `<N_NUMBER_SEQ>` | `<ACC_NUMBER_SEQ>` |
| list_ops | `<N_LIST_OPS>` | `<ACC_LIST_OPS>` |
| modular_arith | `<N_MODULAR>` | `<ACC_MODULAR>` |
| uncovered | `<N_UNCOVERED>` | `<ACC_UNCOVERED>` |

**Why (hypothesis):** Gold CoT-trace noise is structurally zero. Every training trace is derived from the rule function directly. The model does not need to generalise from noisy demonstrations of reasoning; it learns the exact step pattern. This is a stronger claim than answer-level correctness: even when rejection-sampling selects examples with correct final answers, the intermediate reasoning steps can be inconsistent or incorrect, corrupting the CoT supervision signal.

---

## 7. Reproducibility

All results are reproducible from a single seed (`seed=42`) and the exact package versions recorded in the training environment.

### 7.1 Full reproduction steps

```bash
# 1. Clone repo and install
git clone <REPO_URL>
cd nemotron-reasoning
pip install -e ".[dev]"

# 2. Download competition data
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/
unzip data/*.zip -d data/raw/

# 3. Run family recon (CPU only, stdlib only)
python -m src.recon.taxonomy \
    --train-csv data/raw/train.csv \
    --json-out recon.json
# Review output for COVERAGE GAPS and inspect the "uncovered" bucket
# before proceeding. The recon output table is required evidence.

# 4. Generate synthetic data (CPU only)
python scripts/build_synthetic.py \
    --n_per_domain 2000 \
    --seed 42 \
    --output data/synthetic.jsonl

# 5. SFT (requires A100-80GB or equivalent)
python -m src.sft_train --config configs/train.yaml

# 6. Local CV
python -m src.eval.cv \
    --predictions outputs/predictions.csv \
    --solutions data/raw/train.csv

# 7. Package and submit
# submission.zip is written by sft_train.py to the path in configs/train.yaml
kaggle competitions submit \
    -c nvidia-nemotron-model-reasoning-challenge \
    -f submission.zip \
    -m "rank-32 LoRA, pure synthetic, seed=42"
```

### 7.2 Package versions

The `>=` bounds below are minimum floor constraints. They are not exact reproducibility pins. Before submission, run `pip freeze` in the training environment and replace these with `==` constraints for every package that affects numerics (torch, transformers, peft, accelerate, vllm, mamba_ssm, causal_conv1d). The exact frozen requirements should be committed to the repo and linked from the notebook.

**<!-- PLACEHOLDER --> Exact pinned requirements:** `<EXACT_PACKAGE_VERSIONS>` (fill from `pip freeze` on the training machine before June 22).

Floor constraints (for environment setup; not sufficient for reproducibility):

```
torch>=2.1
transformers>=4.46
peft>=0.9          # required for regex target_modules
accelerate
bitsandbytes       # only for 4-bit QLoRA fallback on <80GB GPU
vllm>=0.12
mamba_ssm
causal_conv1d
kagglehub
polars
pandas
pytest
```

`peft>=0.9` is a hard requirement: earlier versions do not support regex `target_modules`, so the `lora_target_regex` in `configs/train.yaml` will not resolve correctly.

### 7.3 Seed management

```python
# src/sft_train.py — set_seed() called before model load, before data split
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

Generator seeds (each family is seeded independently so adding or removing a family does not perturb the others):

| Family | Sub-seed |
|---|---|
| binary_ops | seed + 0 |
| cipher | seed + 1000 |
| linear_eq | seed + 2000 |
| roman | seed + 3000 |
| number_seq | seed + 4000 |
| list_ops | seed + 5000 |
| modular_arith | seed + 6000 |

---

## 8. What makes this the best data method

### 8.1 Recon-before-generate: a reusable methodology

The most reproducible contribution is the protocol itself, not any single result. Before any GPU time is spent, `taxonomy.py` runs a quantitative audit of the train set and converts "which families are covered?" from a post-hoc leaderboard discovery into a pre-compute measurement. The output is a machine-readable JSON plus a human-readable report that explicitly flags every format gap, bit-width gap, and uncovered family. Any team using this protocol knows its coverage before training; any team without it is guessing.

The three audit outputs are: (a) per-domain row counts and proportions, (b) per-domain answer-format distributions cross-checked against what the generators emit, (c) per-domain binary bit-width distributions cross-checked against `GENERATOR_BINARY_WIDTHS`. Each gap type maps to a specific failure mode at eval time.

### 8.2 Mamba `in_proj`/`out_proj` silent no-op finding

Nemotron-3-Nano-30B-A3B is a hybrid Mamba-2 + attention + MoE architecture with only 6 attention layers out of 52 total. Mamba-2 passes `out_proj.weight` directly to custom CUDA kernels without calling `out_proj.forward()`, so a PEFT LoRA on those modules is silently ignored at train time (HF PEFT issue #2274). vLLM's `NemotronHForCausalLM` does not register `in_proj` in its LoRA allowlist. The official competition submission demo ships these as default target modules, but the demo works only because it ships an untrained adapter (`B=0` is the identity transformation). Teams that copied the official demo's target module list silently trained a no-op adapter.

The `_assert_adapter_changes_output()` guard catches this before any training epoch by running one forward pass and asserting the adapter output differs from the frozen base. This guard would have caught the silent no-op at the start of the training run, not after hours of compute.

### 8.3 Metric-exact `\boxed{}` extraction: last not first

`extract_final_answer` in the official scorer returns the **last** non-empty `\boxed{}` match, not the first. The official docstring says "first"; the implementation does `boxed_answers[-1]`. Any local evaluation harness that reads the docstring and implements "first" will silently diverge from the leaderboard score. The divergence is larger for problems where the model emits multiple `\boxed{}` calls (e.g., checking intermediate steps before emitting the final answer). This finding is verifiable by inspecting the scorer source directly.

### 8.4 Zero CoT-trace noise (data quality claim, positioned as supporting evidence)

Rejection-sampling approaches (STaR, self-play) filter on answer correctness, not trace correctness. A teacher model can reason incorrectly while arriving at the right answer, or reason correctly while making an arithmetic slip. Both cases produce correct-answer examples with corrupted CoT traces. Programmatic generators have no teacher: `_build_cot()` derives each reasoning step directly from the rule function, and the self-consistency assertion at generation time aborts if any demo pair does not reproduce. CoT-trace label noise is structurally zero, not just low.

This is a stronger claim than answer-level zero-noise, which any rejection-sampling scheme can match. Zero CoT-trace noise requires that the reasoning path is itself derived from a verifiable source.

### 8.5 Rejection-sampling cost comparison (softened)

Generating `N` gold traces with rejection sampling requires at minimum `N` forward passes for accepted examples, plus additional passes for rejected ones. Generating the same `N` traces programmatically requires zero forward passes. The ratio is at least `N:0` forward passes in favor of the programmatic approach. The actual rejection rate for a capable teacher model on these tasks is unknown without measurement; the point is not the rejection rate but the total forward pass count (`N` vs `0`).

### 8.6 Format correctness is guaranteed

The generators know the exact surface form `verify()` expects. The roman generator always emits uppercase Roman numerals. The binary generator always emits 8-bit strings. These are asserted at generation time, not inferred from a model output. Rejection sampling from a teacher may produce correct answers in wrong surface forms ("The answer is XIV" instead of "XIV"), requiring post-processing that can fail on edge cases.

---

## 9. Ablation plan

The following experiments make the contribution claim undeniable. All four must be run before June 22. Results go into the notebook Cell 18 results table and the writeup §6.3.

**(i) Synthetic-only vs. synthetic+real-train per-family accuracy**

Train two adapters: one on `data/synthetic.jsonl` only, one on the concatenation of `data/synthetic.jsonl` and the real `train.csv` (excluding any rows in the val split). Report per-family CV accuracy for both. This shows whether the synthetic data alone captures the task distribution or benefits from real examples.

Result: `<RESULT_ABLATION_SYNTHETIC_VS_MIXED>`

**(ii) Per-family CV breakdown including the "uncovered" bucket (REQUIRED evidence)**

Run `run_cv()` with domain inference enabled and report the full per-domain table including the uncovered bucket. This is not optional; it is the direct evidence that the recon-before-generate protocol identified which families the adapter covers and which it does not. Include the recon output table/figure showing real-train domain distribution alongside the CV per-domain results.

Result: `<RESULT_ABLATION_PER_FAMILY_CV>` (fill from §6.3 table above)

**(iii) n_per_domain scaling curve (100 to 2000)**

Train adapters at n_per_domain = 100, 250, 500, 1000, 2000. Plot overall CV accuracy vs. n_per_domain. This tests whether more synthetic data monotonically helps and identifies any saturation point.

Results: `<RESULT_ABLATION_SCALING_CURVE>` (table: n_per_domain -> CV acc)

**(iv) Gold-CoT vs. noisy-distilled-CoT comparison**

Generate a noisy version of the training data by replacing the programmatic `gold_cot` with CoT traces sampled from a teacher model (e.g., the base Nemotron without LoRA, or another available model). Keep answers identical and correct. Train two adapters with identical hyperparameters, differing only in the CoT trace source. Report per-family CV accuracy for both. This is the direct empirical evidence for the zero-CoT-trace-noise claim in §8.4.

Result: `<RESULT_ABLATION_COT_QUALITY>`

---

## 10. Limitations

- **Coverage is bounded by enumerable rules.** If the test set includes families whose rules cannot be expressed as closed-form functions (e.g., free-text analogical reasoning, spatial transformation), this method provides no training signal for those families. The "uncovered" bucket in recon output is the honest measure of this gap.
- **Recon classifier lags behind the generator suite.** The current keyword classifier in `taxonomy.py` recognizes four families (binary_ops, cipher, linear_eq, roman). The three generator families added since (number_seq, list_ops, modular_arith) will appear in the "uncovered" bucket until the classifier is extended. See §3.4.
- **Prompt surface form assumes stability.** The Wonderland header and demo format are constants derived from inspecting the train set. If the hidden test uses a different framing, the adapter may not generalise. Mitigation: the `_WONDERLAND_HEADER` constant in `common.py` should be verified against the actual test prompts once they are available.
- **6/52 attention layers limits LoRA capacity.** Most of the adapter is in the MLP projections (`gate/up/down_proj`) because Nemotron-3-Nano-30B-A3B has only 6 attention layers. LoRA rank-32 on 6 attention layers is a thin signal; the bulk of the adaptation comes from the 52-layer MLP coverage. The practical implication is that attention-heavy families (long-range pattern induction) may be under-served by the current adapter placement.
- **CV-to-leaderboard gap is unknown until submission.** The bootstrap CI captures sampling variance in the val split; it does not account for distribution shift between the public train set and the private test set. The CV score is a measure of metric-exact accuracy on the synthetic val distribution, not a prediction of leaderboard rank.
- **Binary bit-width is hard-coded to 8.** `GENERATOR_BINARY_WIDTHS = {8}`. If recon reveals that the real train set includes 16-bit or 4-bit binary answers, the generators must be extended before training. The recon tool will flag this explicitly as a `BINARY WIDTH GAP`.

---

## 11. File map

```
nemotron-reasoning/
├── configs/train.yaml              # all hyperparameters; lora_r <= 32 enforced at load
├── scripts/build_synthetic.py      # entry point: generate data/synthetic.jsonl
├── src/
│   ├── generators/
│   │   ├── common.py               # Example dataclass, format_wonderland_prompt
│   │   ├── binary_ops.py
│   │   ├── cipher.py
│   │   ├── linear_eq.py
│   │   ├── roman.py                # 4 rule variants, 5-step CoT template
│   │   ├── number_seq.py           # arithmetic, geometric, Fibonacci-style sequences
│   │   ├── list_ops.py             # sort, reverse, filter, map on integer lists
│   │   └── modular_arith.py        # modular add, mul, inverse, exponentiation
│   ├── recon/
│   │   └── taxonomy.py             # classify, cluster, gap-detect; stdlib only
│   ├── eval/
│   │   ├── metric.py               # verbatim verify() + extract_final_answer()
│   │   └── cv.py                   # run_cv(): overall acc + CI + per-domain
│   └── sft_train.py                # LoRA SFT; adapter no-op check; submission.zip
└── writeup/
    ├── SOLUTION_WRITEUP.md         # this file
    └── NOTEBOOK_OUTLINE.md         # Kaggle notebook cell plan
```

---

## 12. Placeholder index

| Placeholder | Where to get it | Blocks submission? |
|---|---|---|
| `<REPO_URL>` | Public Kaggle notebook or GitHub URL | YES |
| `<ADAPTER_PATH>` | Kaggle dataset path of uploaded trained adapter | YES |
| `<EXACT_PACKAGE_VERSIONS>` | `pip freeze` on the training machine | YES |
| `<N_SYNTHETIC_TOTAL>` | Printed by `build_synthetic.py` (= 7 x n_per_domain) | YES |
| `<FINAL_N_PER_DOMAIN>` | Final `--n_per_domain` value used in training run | YES |
| `<TRAIN_WALL_TIME>` | Timer around `sft_train.py` run | No |
| `<GPU_TYPE>` | Cloud instance type used | No |
| `<TRAIN_LOSS_FINAL>` | Last logged training loss | No |
| `<BASELINE_LB_SCORE>` | Zero-shot submission or competition page | YES |
| `<CV_ACC>` | `run_cv()` output | YES |
| `<CV_CI_LOWER>`, `<CV_CI_UPPER>` | `run_cv()` bootstrap output | YES |
| `<LB_SCORE>` | Kaggle public leaderboard | YES |
| `<DELTA_VS_BASELINE>` | `<LB_SCORE>` minus `<BASELINE_LB_SCORE>` | YES |
| `<CV_LB_GAP>` | `<CV_ACC>` minus `<LB_SCORE>` | YES |
| Per-domain N and accuracy values | `run_cv()` domain breakdown (all 7 families + uncovered) | YES |
| `<RESULT_ABLATION_*>` | Ablation experiments (§9) | YES — required evidence |
