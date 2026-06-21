# Cost-Effective Training of High-Quality Language Models

A cited, verified research brief scoping an EECS research direction. Sources are
primary papers and technical reports (2022 to 2026). Claims here survived a
three-vote adversarial verification pass; self-reported or weakly-supported items
are flagged in the caveats.

## Headline

Cost-effective training splits into two regimes that share one principle: spend
compute where it buys the most capability. The single most replicated and most
leverageable finding is that **data quality is now a first-class scaling-law
dimension** that trades directly against model size and compute. For a student on
a sub-$50 budget, data curation and synthetic data are the highest-return knob.

## Angle A: how frontier labs cut cost while keeping quality

1. Mixture-of-Experts sparsity. DeepSeek-V3 (671B total, 37B active per token)
   reached GPT-4o-tier quality in 2.788M H800 GPU-hours (about $5.6M marginal
   cost for the final run). Source: arXiv:2412.19437.
2. Quantified MoE saving: about 250 GFLOPS/token vs 394 for a 72B dense (1.6x)
   and 2448 for a 405B dense (about 10x). Source: arXiv:2505.09343.
3. FP8 / low precision: DeepSeek reports under 0.25% accuracy loss vs BF16; the
   COAT FP8 framework cuts training memory 1.54x and gives 1.43x speedup nearly
   losslessly (public code). Sources: arXiv:2505.09343, arXiv:2410.19313.
4. Memory-efficient attention (Multi-head Latent Attention): 4.66x to 7.3x
   KV-cache reduction. Note: this is an inference/serving benefit, not a
   training-memory benefit. Source: arXiv:2505.09343.
5. Data quality over quantity: Microsoft phi-1 (1.3B params) hit 50.6% HumanEval
   on about 7B curated plus synthetic tokens in about 32 A100-days; phi-1.5
   matched models 5x larger. Sources: arXiv:2306.11644, arXiv:2309.05463.
6. Synthetic data generation: over 98% of Nemotron-4-340B alignment data was
   synthetic (vs about 20K human samples), pipeline open-sourced. Source:
   arXiv:2406.11704.
7. Compute-optimal (Chinchilla) scaling: scale params and tokens about 1:1; a
   smaller well-trained model beats a larger undertrained one at equal compute
   (70B Chinchilla beats 280B Gopher and 530B MT-NLG). Source: arXiv:2203.15556.
8. Hyperparameter transfer (muP): tune on a small model, transfer to the large
   one, avoiding expensive at-scale tuning. Source: arXiv:2203.03466.

## Angle B: what a student can run free or near-free

9. Data quality as "effective tokens": quality can be written as a measurable
   multiplier on the token budget (Dq = D * exp(c1*diversity + c2*syntheticity)),
   predicting accuracy at +0.83 Pearson. Sources: arXiv:2510.03313,
   arXiv:2410.03083.
10. Synthetic/curated data is the highest-leverage cheap lever: BeyondWeb matches
    web-data accuracy with 7.7x fewer tokens at 8B scale. Source:
    datologyai.com/blog/beyondweb (vendor self-report, treat as directional).
11. Parameter-efficient and quantized fine-tuning (QLoRA 4-bit, LoRA, DoRA) plus
    8-bit optimizers, gradient checkpointing, and FlashAttention make small
    models trainable on a single 16GB GPU. Sources: arXiv:2305.14314 (QLoRA),
    arXiv:2110.02861 (8-bit optimizers), arXiv:2205.14135 (FlashAttention).

## The synthesized cheapest recipe

- Frontier scale: MoE sparsity + FP8 + compute-optimal data + muP transfer +
  heavy data curation + synthetic data. (The DeepSeek/Nemotron recipe.)
- Student scale: an open base model + QLoRA (4-bit) + curated or synthetic data +
  gradient checkpointing and an 8-bit optimizer, on a free Kaggle or Colab GPU.
- The deepest shared idea: invest in data quality and architectural sparsity, not
  raw compute.

## Honest caveats (important for a thesis)

- Several headline numbers are self-reported by the creators and not
  independently replicated at scale (DeepSeek FP8 loss, COAT "nearly lossless",
  phi benchmarks, BeyondWeb 7.7x).
- The DeepSeek $5.6M figure is the marginal cost of the final run only; it
  excludes research, ablations, failed runs, and data.
- The quality-aware scaling laws are validated mainly on small or synthetic
  setups, so the multipliers are directional, not frontier-validated.
- Two sub-claims were refuted in verification: that synthetic "textbook" data is
  the proven core causal driver of phi gains, and that quality gains vanish above
  about 1.5B params. So quality-over-scale is shown empirically, but the precise
  mechanism and the upper size bound are open.

## Verified gaps = thesis-grade open questions

The research surfaced four gaps with no solid answer in the literature. These are
exactly what makes a good thesis, and all are runnable on free or under-$50
compute with small models:

1. Concrete single-16GB-GPU results: what is the actual QLoRA quality loss vs full
   fine-tuning, in points, and the largest trainable model size? No verified
   number exists.
2. A dollar-per-quality-point ranking of the cost levers (data curation vs 4-bit
   vs distillation), normalized to a common metric. Nobody has published this.
3. Whether the quality-aware scaling laws hold beyond the small-model regime
   where they were fitted.
4. The degradation ceiling of recursive synthetic data (model collapse) over
   multiple generations.

## Proposed experiments (free or under $50, small models)

E1. Data quality vs quantity scaling curve. Train a small model (for example
    160M to 1B) on raw web vs curated vs solver-synthetic data at matched token
    budgets. Measure the effective-token multiplier and test the quality-aware
    scaling law at student scale. Fills gap 3.

E2. QLoRA quality loss vs full fine-tuning. On one task, compare full fine-tuning,
    LoRA, and 4-bit QLoRA for a 1B to 7B model. Report quality delta, memory, and
    cost. Directly fills gap 1, which has no published number.

E3. Synthetic-data efficiency per dollar. Reuse the closed-form solvers from this
    repo (zero-label-noise data) to measure accuracy per token and per dollar vs
    real data. Connects to BeyondWeb and to your existing work.

E4. Dollar-per-quality-point ranking. Hold a small-model task fixed and compare
    cost-reduction levers normalized to quality per dollar. Fills gap 2.

E5. Recursive synthetic-data degradation. Train across several generations of
    synthetic data and find where quality collapses. Fills gap 4.

The strongest, most ownable choices are E2 and E3: E2 fills a real gap with a
clean experiment, and E3 builds directly on the solver method you already have.

## Source list (primary unless noted)

- arXiv:2412.19437 DeepSeek-V3 technical report
- arXiv:2505.09343 DeepSeek hardware/efficiency insights
- arXiv:2410.19313 COAT FP8 training (code: github.com/NVlabs/COAT)
- arXiv:2406.11704 Nemotron-4-340B (synthetic data pipeline)
- arXiv:2306.11644 phi-1 (Textbooks Are All You Need)
- arXiv:2309.05463 phi-1.5
- arXiv:2203.15556 Chinchilla (compute-optimal scaling)
- arXiv:2203.03466 muP / muTransfer (hyperparameter transfer)
- arXiv:2510.03313, arXiv:2410.03083 quality-aware scaling laws
- arXiv:2305.14314 QLoRA
- arXiv:2402.09353 DoRA
- arXiv:2110.02861 8-bit optimizers
- arXiv:2205.14135 FlashAttention
- datologyai.com/blog/beyondweb BeyondWeb (vendor, directional)
