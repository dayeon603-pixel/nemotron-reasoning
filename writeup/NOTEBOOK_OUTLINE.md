# Kaggle Notebook Outline -- Nemotron Reasoning Challenge
**Public reproducibility notebook for Open Contribution Award**

Target: runs end-to-end on a Kaggle notebook with GPU enabled. Cells are ordered so every output feeds the next cell. The notebook does NOT need the full A100-80GB training run to demonstrate the pipeline -- it shows the data generation and CV harness in full, and loads a pre-trained adapter for the scoring cell.

All seeds are pinned. All results should be reproducible by re-running all cells in order.

**Award requirement:** Cell 17 must run with `GPU_AVAILABLE = True` and must show at least one correct prediction from the trained adapter. A notebook that skips inference and leaves `GPU_AVAILABLE = False` does not demonstrate that the pipeline produces a working adapter. Fill `<ADAPTER_PATH>` before publishing.

---

## Cell structure

### Cell 1 -- Markdown: Title and scope

```
# Pure Programmatic Synthetic Data for Nemotron Reasoning
**Method:** Closed-form generators emit (prompt, gold chain-of-thought, answer) tuples for
inductive rule-discovery task families. LoRA rank-32 SFT on nemotron-3-nano-30b-a3b BF16.
No teacher model required for data generation.

**This notebook covers:**
1. Family taxonomy recon on train.csv
2. Programmatic data generation (CPU, no model) -- seven domain families
3. Metric-exact local CV harness
4. LoRA adapter loading and sample inference (requires GPU and trained adapter)

All code runs from the repo root. Every command is copy-pasteable.
```

### Cell 2 -- Code: Install dependencies

```python
# Install pinned deps. mamba_ssm and causal_conv1d are required to load
# the base model; skip if only running data generation + CV cells.
# IMPORTANT: replace >= bounds with == constraints from the training environment's
# pip freeze before publishing. The >= bounds are floor constraints, not
# reproducibility pins.
!pip install -q \
    \"transformers>=4.46\" \
    \"peft>=0.9\" \
    \"accelerate\" \
    \"kagglehub\" \
    \"polars\" \
    \"numpy\" \
    \"pandas\" \
    \"pytest\"

# Only needed for model loading / inference cells:
# !pip install -q mamba_ssm causal_conv1d \"vllm>=0.12\"
```

### Cell 3 -- Code: Clone repo and set working directory

```python
import subprocess, os

REPO_URL = \"<REPO_URL>\"  # PLACEHOLDER: fill before publishing
subprocess.run([\"git\", \"clone\", REPO_URL, \"nemotron-reasoning\"], check=True)
os.chdir(\"nemotron-reasoning\")
print(\"Working directory:\", os.getcwd())
```

### Cell 4 -- Code: Download competition data

```python
import subprocess
subprocess.run([
    \"kaggle\", \"competitions\", \"download\",
    \"-c\", \"nvidia-nemotron-model-reasoning-challenge\",
    \"-p\", \"data/\"
], check=True)
subprocess.run([\"unzip\", \"-q\", \"data/*.zip\", \"-d\", \"data/raw/\"], check=True)

import pathlib
rows = sum(1 for _ in open(\"data/raw/train.csv\")) - 1  # subtract header
print(f\"train.csv rows: {rows}\")
```

### Cell 5 -- Markdown: Step 1 -- Family taxonomy recon

```
## Step 1: Family taxonomy recon

Before generating any data, we classify every train.csv row into a generator domain
and check whether real answer formats and binary bit-widths match what our generators
emit. This step runs on CPU with stdlib only.

Key checks:
- Any \"uncovered\" rows: families we are missing
- Format mismatches: generator output will not match verify()
- Binary bit-width mismatches: certain 0 scores on binary tasks

The recon output table (domain distribution + gap list) is required evidence for
the Open Contribution Award submission. The "uncovered" bucket breakdown is
particularly important: three of our seven generator families (number_seq, list_ops,
modular_arith) do not have keywords in the current classifier and will appear there.
```

### Cell 6 -- Code: Run taxonomy recon

```python
import subprocess, json

result = subprocess.run([
    \"python\", \"-m\", \"src.recon.taxonomy\",
    \"--train-csv\", \"data/raw/train.csv\",
    \"--json-out\", \"recon.json\"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print(\"STDERR:\", result.stderr)
```

### Cell 7 -- Code: Parse and display recon gaps

```python
import json

with open(\"recon.json\") as f:
    recon = json.load(f)

print(f\"Total rows analyzed: {recon['total']}\")
print(f\"\\nCoverage gaps ({len(recon['gaps'])}):\")
for gap in recon[\"gaps\"]:
    print(f\"  ! {gap}\")

print(\"\\nDomain distribution:\")
for domain, stats in sorted(recon[\"domains\"].items(), key=lambda x: -x[1][\"count\"]):
    pct = 100.0 * stats[\"count\"] / recon[\"total\"]
    print(
        f\"  {domain:15s}  {stats['count']:5d} rows ({pct:.1f}%)  \"
        f\"formats={stats['answer_formats']}  \"
        f\"templates={stats['distinct_templates']}\"
    )
```

### Cell 8 -- Markdown: Step 2 -- Generate synthetic data

```
## Step 2: Programmatic synthetic data generation

Seven generator families, each covering multiple rule sub-variants:
- **binary_ops**: 8-bit bitwise operations (AND, OR, XOR, NOT, shifts, two's complement)
- **cipher**: substitution and shift ciphers (Caesar-style)
- **linear_eq**: one-variable linear equations with integer/float solutions
- **roman**: int_to_roman, roman_to_int, roman_add, roman_subtract (range [1, 3999])
- **number_seq**: arithmetic, geometric, and Fibonacci-style integer sequences
- **list_ops**: sort, reverse, filter, map-style operations on integer lists
- **modular_arith**: modular addition, multiplication, inverse, exponentiation

Each example: `prompt` (Wonderland-format puzzle with 4 demo pairs) + `gold_cot`
(chain-of-thought ending in \\boxed{answer}) + `answer`.

No model is invoked. Generation takes seconds on CPU.

Note: the current taxonomy classifier recognizes 4 of the 7 families by keyword.
The three new families (number_seq, list_ops, modular_arith) appear in the
"uncovered" bucket in the recon output above. This is expected; their prompts
were not covered by train.csv keywords at classification time.
```

### Cell 9 -- Code: Generate synthetic data

```python
import subprocess

result = subprocess.run([
    \"python\", \"scripts/build_synthetic.py\",
    \"--n_per_domain\", \"2000\",
    \"--seed\", \"42\",
    \"--output\", \"data/synthetic.jsonl\"
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(\"STDERR:\", result.stderr)
```

### Cell 10 -- Code: Inspect generated examples

```python
import json

with open(\"data/synthetic.jsonl\") as f:
    examples = [json.loads(line) for line in f]

print(f\"Total examples: {len(examples)}\")
print(f\"Expected: 14000 (7 families x 2000 per family)\")

seen_domains = set()
for ex in examples:
    parts = ex[\"id\"].split(\"_\")
    # id format: synthetic_{domain_name}_{index:06d}
    # domain_name may contain underscores; index is last field
    domain_key = \"_\".join(parts[1:-1])
    if domain_key not in seen_domains:
        seen_domains.add(domain_key)
        print(f\"\\n{'='*60}\")
        print(f\"Domain: {domain_key}\")
        print(f\"Prompt:\\n{ex['prompt']}\")
        print(f\"\\nGold CoT (first 400 chars):\\n{ex['trace'][:400]}...\")
        print(f\"\\nAnswer: {ex['gold_answer']}\")
    if len(seen_domains) == 7:
        break
```

### Cell 11 -- Markdown: Step 3 -- Metric-exact local CV

```
## Step 3: Metric-exact local CV

`src/eval/metric.py` is a verbatim copy of the official competition scorer.

Non-obvious behaviours replicated exactly:
- `extract_final_answer` returns the **last** \\boxed{} match, not the first
  (official docstring says \"first\"; the code returns `boxed_answers[-1]`)
- Binary string comparison is length-sensitive: \"10011000\" != \"00011000\"
- Numeric: `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`
- Fallback: case-insensitive `.lower()` equality

The metric implementation is exact. The distribution assumption is not: the local
CV val split shares the synthetic train distribution, while the leaderboard hidden
test may have a different family distribution and uncovered-family fraction.
```

### Cell 12 -- Code: Verify the metric implementation on known cases

```python
from src.eval.metric import verify, extract_final_answer

cases = [
    # (stored, predicted, expected_result, description)
    (\"10011000\", \"10011000\", True,  \"binary exact match\"),
    (\"10011000\", \"00011000\", False, \"binary wrong length/value\"),
    (\"10011000\", \"152\",      False, \"binary vs decimal int\"),
    (\"XIV\",      \"xiv\",      True,  \"roman case-insensitive\"),
    (\"24.64\",    \"24.6401\",  True,  \"float within 1% rel tol\"),
    (\"42\",       \"42\",       True,  \"int exact\"),
]

all_pass = True
for stored, predicted, expected, desc in cases:
    result = verify(stored, predicted)
    status = \"PASS\" if result == expected else \"FAIL\"
    if status == \"FAIL\":
        all_pass = False
    print(f\"[{status}] {desc}: verify({stored!r}, {predicted!r}) = {result}\")

assert all_pass, \"Metric implementation diverges from expected -- do not trust CV scores.\"
print(\"\\nAll metric tests passed.\")
```

### Cell 13 -- Code: Demonstrate extract_final_answer edge cases

```python
from src.eval.metric import extract_final_answer

cases = [
    (r\"some text \\boxed{XIV} more text \\boxed{XVI}\", \"XVI\",       \"last boxed wins\"),
    (r\"\\boxed{10011000}\",                              \"10011000\",  \"binary in boxed\"),
    (r\"\\boxed{}52}\",                                   \"}52\",       \"nested brace handling\"),
    (\"The final answer is: XLII\",                       \"XLII\",      \"phrase fallback\"),
    (\"result is 42 or maybe 43\",                        \"43\",        \"number fallback: last\"),
]

for text, expected, desc in cases:
    result = extract_final_answer(text)
    status = \"PASS\" if result == expected else \"FAIL\"
    print(f\"[{status}] {desc}\")
    if status == \"FAIL\":
        print(f\"  expected {expected!r}, got {result!r}\")
```

### Cell 14 -- Code: Self-consistency sanity check (gold-as-prediction = 1.0)

```python
import json
import pandas as pd
from src.eval.cv import run_cv
import logging
logging.basicConfig(level=logging.INFO, format=\"%(message)s\")

with open(\"data/synthetic.jsonl\") as f:
    rows = [json.loads(line) for line in f]

predictions_df = pd.DataFrame({
    \"id\":         [r[\"id\"] for r in rows],
    \"prediction\": [r[\"gold_answer\"] for r in rows],
})
solutions_df = pd.DataFrame({
    \"id\":     [r[\"id\"] for r in rows],
    \"answer\": [r[\"gold_answer\"] for r in rows],
    \"prompt\": [r[\"prompt\"] for r in rows],
})

result = run_cv(predictions_df, solutions_df, n_boot=200, bootstrap_seed=42)
print(f\"\\nSanity check -- gold-as-prediction accuracy: {result.overall_accuracy:.4f}\")
print(\"Expected: 1.0000  (any deviation = generator or metric bug)\")
assert result.overall_accuracy == 1.0, (
    f\"Generator self-consistency FAILED: acc={result.overall_accuracy}\"
)
```

### Cell 15 -- Markdown: Step 4 -- LoRA adapter

```
## Step 4: LoRA adapter

Training requires an A100-80GB (60 GB BF16 base + activations + optimizer states).
This cell loads a pre-trained adapter for demonstration.

Key config points (from configs/train.yaml):
- lora_r=32 (challenge maximum; fully used)
- target_modules: q/k/v/o_proj + gate/up/down_proj
- NOT targeting Mamba in_proj/out_proj: these are a silent no-op adapter at both
  train time (HF PEFT issue #2274) and scoring time (vLLM NemotronH allowlist).
  The official demo ships an untrained adapter (B=0 identity) using these modules.
- peft>=0.9 required for regex target_modules

Award requirement: GPU_AVAILABLE must be True for the final published notebook.
The inference cell must show at least one correct prediction. A notebook that
leaves GPU_AVAILABLE=False does not demonstrate a working trained adapter.
```

### Cell 16 -- Code: Display training config

```python
import yaml
with open(\"configs/train.yaml\") as f:
    cfg = yaml.safe_load(f)

print(\"Training config (configs/train.yaml):\")
for k, v in cfg.items():
    print(f\"  {k}: {v}\")
```

### Cell 17 -- Code: Load adapter and run sample inference (GPU required)

```python
# Requires mamba_ssm, causal_conv1d, vllm>=0.12 and ~80GB GPU VRAM.
# Set GPU_AVAILABLE = True only on A100-80GB or equivalent.
# AWARD REQUIREMENT: this must be True in the final published notebook.
# The inference output must show at least one correct prediction.
GPU_AVAILABLE = False  # PLACEHOLDER: set True before publishing; fill ADAPTER_PATH first

if GPU_AVAILABLE:
    import kagglehub, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    MODEL_ID      = \"metric/nemotron-3-nano-30b-a3b-bf16/transformers/default\"
    ADAPTER_PATH  = \"<ADAPTER_PATH>\"  # PLACEHOLDER: Kaggle dataset path of trained adapter

    model_path = kagglehub.model_download(MODEL_ID)
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=\"auto\"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    from src.generators import generate_roman
    sample = generate_roman(1, seed=99)[0]
    print(\"Prompt:\")
    print(sample.prompt)
    print(f\"\\nGold answer: {sample.answer}\")

    inputs = tokenizer(sample.prompt, return_tensors=\"pt\").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f\"\\nModel output (first 400 chars):\\n{response[:400]}\")

    from src.eval.metric import extract_final_answer, verify
    pred    = extract_final_answer(response)
    correct = verify(sample.answer, pred)
    print(f\"\\nExtracted: {pred!r}  |  Gold: {sample.answer!r}  |  Correct: {correct}\")
    assert correct, (
        f\"Inference demo failed: adapter produced incorrect answer on sample. \"
        f\"Check adapter path and that in_proj/out_proj are NOT in target_modules.\"
    )
else:
    print(\"Skipping model inference (GPU_AVAILABLE=False).\")
    print(\"Set GPU_AVAILABLE=True and fill ADAPTER_PATH before publishing.\")
    print(\"The award requires a runnable inference demo with >=1 correct prediction.\")
```

### Cell 18 -- Markdown: Results

```
## Results

<!-- PLACEHOLDER: fill all values after training run and leaderboard submission -->

| Metric | Value |
|---|---|
| Synthetic examples (total) | `<N_SYNTHETIC_TOTAL>` (7 families x `<FINAL_N_PER_DOMAIN>`) |
| Local CV accuracy | `<CV_ACC>` |
| Bootstrap 95% CI | [`<CV_CI_LOWER>`, `<CV_CI_UPPER>`] (n=1000, seed=42) |
| Leaderboard public score | `<LB_SCORE>` |
| Zero-shot baseline (no adapter) | `<BASELINE_LB_SCORE>` |
| Delta vs. zero-shot baseline | `<DELTA_VS_BASELINE>` |

Per-domain CV breakdown (including uncovered bucket):

| Domain | N (val) | CV Accuracy |
|---|---|---|
| binary_ops  | `<N_BINARY>`      | `<ACC_BINARY>` |
| cipher      | `<N_CIPHER>`      | `<ACC_CIPHER>` |
| linear_eq   | `<N_ALGEBRA>`     | `<ACC_ALGEBRA>` |
| roman       | `<N_ROMAN>`       | `<ACC_ROMAN>` |
| number_seq  | `<N_NUMBER_SEQ>`  | `<ACC_NUMBER_SEQ>` |
| list_ops    | `<N_LIST_OPS>`    | `<ACC_LIST_OPS>` |
| modular_arith | `<N_MODULAR>`   | `<ACC_MODULAR>` |
| uncovered   | `<N_UNCOVERED>`   | `<ACC_UNCOVERED>` |
```

### Cell 19 -- Code: Check submission.zip

```python
import pathlib, zipfile

sub_path = pathlib.Path(\"submission.zip\")
if sub_path.exists():
    with zipfile.ZipFile(sub_path) as z:
        print(\"submission.zip contents:\")
        for info in z.infolist():
            print(f\"  {info.filename}  ({info.file_size:,} bytes)\")
else:
    print(\"submission.zip not found -- run src/sft_train.py on a GPU machine first.\")
    print(\"Expected: adapter_config.json + adapter_model.safetensors\")

print(\"\\nTo submit:\")
print(\"  kaggle competitions submit \\\\\\n\"
      \"    -c nvidia-nemotron-model-reasoning-challenge \\\\\\n\"
      \"    -f submission.zip \\\\\\n\"
      \"    -m 'rank-32 LoRA, pure synthetic, 7 families, seed=42'\")
```

---

## Pre-publication checklist

Before making the notebook public:

1. Fill `<REPO_URL>` in Cell 3.
2. Fill `<ADAPTER_PATH>` in Cell 17 with the Kaggle dataset path where the trained adapter is uploaded. This is BLOCKING: without a real adapter path the inference cell cannot run.
3. Set `GPU_AVAILABLE = True` in Cell 17 on a notebook with A100-80GB available. The award requires a runnable inference demo showing at least one correct prediction. Do not publish with `GPU_AVAILABLE = False`.
4. Confirm Cell 17 prints `Correct: True` for at least one sample. If not, check that `in_proj`/`out_proj` are not in target_modules.
5. Fill all `<PLACEHOLDER>` values in Cell 18 from actual run results.
6. Run all cells in order on Kaggle with GPU enabled and confirm no errors.
7. Confirm Cell 14 prints `Sanity check -- gold-as-prediction accuracy: 1.0000`.
8. Confirm Cell 12 prints `All metric tests passed.`
9. Replace `>=` version bounds in Cell 2 with exact `==` pins from `pip freeze` on the training environment. The `>=` bounds are floor constraints only.
10. Confirm Cell 10 prints `Total examples: 14000` (or 7 x whatever `--n_per_domain` was used) and shows one example from each of the 7 domains.
