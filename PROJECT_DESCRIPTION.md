# Project Description — Nemotron Reasoning Solver

## One line
Reverse-engineered an NVIDIA Research reasoning benchmark from raw data and built
closed-form solvers plus a zero-label-noise fine-tuning pipeline for a 30B model.

## Short (portfolio card)
A full solution pipeline for the NVIDIA Nemotron Model Reasoning Challenge, an
inductive rule-discovery benchmark scored on a 30B hybrid Mamba and attention
model via LoRA adapters. Starting from 9,500 raw labeled examples, I identified
that the benchmark is exactly six hidden task families and wrote a programmatic
solver for each. Three families (physics-formula inference, unit-scaling, and
numeral conversion) are solved at 100% against ground truth, and the text
substitution family reaches 100% with a dictionary-aided decoder. The result is
a 9,500-example training set with zero label noise on the exact task
distribution, plus a tested data and training pipeline (509 unit tests).

## Longer (application / blog paragraph)
I entered the NVIDIA Nemotron Model Reasoning Challenge, a Kaggle competition
where submissions are LoRA adapters for a 30B hybrid Mamba and attention model,
evaluated on a held-out reasoning benchmark from NVIDIA Research. Rather than
guess at the task distribution, I ran a taxonomy analysis on the 9,500-row
training set and discovered the benchmark is a closed set of six inductive
"infer the hidden rule from examples, then apply it" families. I then treated
the problem as reverse engineering: for each family I wrote a solver that parses
the examples, recovers the rule, and produces a verifiable answer. Three
families solve perfectly (a falling-body constant recovered by interval
arithmetic, a linear unit scale, and decimal to Roman numeral conversion), the
substitution cipher family solves fully with a dictionary-constrained decoder,
and I quantified exactly why two families (8-bit boolean transforms and a
variable-length symbol transducer) are only partially determined by the few
examples shown, which explains the field-wide accuracy ceiling. From the solvers
I assembled a training set of all 9,500 real prompts paired with correct
reasoning traces and verified answers, giving zero label noise on the exact
evaluation distribution. The pipeline (synthetic data generators, the real-data
assembler, a metric-exact evaluation harness, and a cloud training launcher) is
covered by 509 unit tests.

## Honest status note
The leaderboard submission was not completed before the deadline because of
cloud-environment build failures during the final training run. The analysis,
the solvers, and the training data are complete and proven, and the pipeline is
ready to produce a trained adapter and a measured accuracy number.

## Skills demonstrated
Reverse engineering, program synthesis, data-centric ML, LoRA fine-tuning on
large hybrid-architecture models, metric-exact evaluation, test-driven
pipelines, and honest reporting of limitations.
