# Experiments

## Proposal A: solver-synthetic vs real data efficiency (free GPU)

`proposal_a_solver_data_efficiency.py` fine-tunes a small open model (Qwen2.5-1.5B)
with 4-bit QLoRA on two data sources (zero-noise solver-generated data and real
competition data) at several training-set sizes, then measures held-out accuracy.

This runs on a single free 16GB GPU (Kaggle T4/P100 or Colab T4) and a first data
point finishes in roughly 1 to 2 hours.

### Run on Kaggle or Colab (free GPU)

1. Turn on the GPU runtime (Kaggle: Settings, Accelerator, GPU T4 x2 or P100;
   Colab: Runtime, Change runtime type, T4 GPU).
2. In a cell:

```
!git clone https://github.com/dayeon603-pixel/nemotron-reasoning
%cd nemotron-reasoning
!pip install -q "transformers>=4.44" "peft>=0.11" "bitsandbytes>=0.43" "accelerate>=0.30" "datasets>=2.19"
!python experiments/proposal_a_solver_data_efficiency.py
```

### Output

Results are saved to `experiments/proposal_a_results.json` as accuracy for each
(data source, training size). To get the proposal's headline result:

- Plot accuracy vs training size for each source.
- The horizontal gap at a fixed accuracy is the effective-data multiplier (how
  many fewer solver examples reach the same accuracy as real examples).
- Multiply the training size by your GPU cost per run to get accuracy per dollar.

### Notes

- If `data/raw/train.csv` is absent, the real arm reuses a disjoint solver split
  and the script logs a warning. To use real data, download the competition data
  into `data/raw/` first.
- This is a deliberately small first version. To strengthen it: add a second
  solver family, add an LLM-distilled data arm, sweep more sizes, and try a 3B
  model.
