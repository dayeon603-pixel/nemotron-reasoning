# Project Explainer: Fine-Tuning a 30B Model on an Inductive Reasoning Benchmark

This document explains the whole project in plain language so you can understand
and defend every part of it. Read it top to bottom once, then use the glossary
at the end as a reference.

## 1. What the task is

The benchmark gives the model a puzzle. Each puzzle shows a few input/output
examples produced by a hidden rule, then asks the model to apply that rule to a
new input. The model must reason out the rule and put its final answer inside a
`\boxed{...}` tag. The grader extracts the boxed value and checks it against the
true answer (exact string for some types, 1 percent tolerance for numbers).

This is called inductive reasoning: infer the rule from examples, then apply it.

## 2. The six hidden families

By analyzing the 9,500 labeled training rows, the benchmark turned out to be
exactly six puzzle families, each about one sixth of the data:

1. gravitational: observations of (time, distance) under d = 0.5 g t squared.
   Infer g, then predict distance for a new time. Answer is a float.
2. unit conversion: each measurement is multiplied by a hidden constant k.
   Infer k, then convert a new measurement. Answer is a float.
3. numeral: decimal numbers converted to Roman numerals. Answer is a string.
4. encrypt: a random letter substitution cipher. Infer the letter map from
   examples, then decrypt a new phrase.
5. bitmanip: a hidden boolean rule transforms 8 bit binary strings.
6. symbol: strings of punctuation symbols transformed by a hidden rule.

## 3. Key idea one: reverse engineering with solvers

Instead of hoping the model figures everything out, the project wrote a small
program (a "solver") for each family that parses the examples, recovers the
hidden rule, and computes the answer. Three families are solved perfectly:

- gravitational: average 2 d / t squared across observations to recover g.
- unit conversion: average y / x across examples to recover k.
- numeral: standard decimal to Roman conversion.

These solvers were validated against all the real rows and matched the true
answers 100 percent of the time. The encrypt family is solved by building the
substitution table from the examples and filling gaps with an English word list.

Why this matters: a correct solver can generate unlimited training data with
perfectly correct reasoning traces, with zero label noise. That is the
"data centric" idea: improve the data, not just the model.

## 4. Key idea two: the information ceiling

Two families cannot be fully solved, and this was proven, not guessed:

- bitmanip: only about 40 percent of puzzles are determined by the 8 examples
  shown. For the rest, many different bit rules fit the same 8 examples, so the
  correct one cannot be known from the data given.
- symbol: about half the puzzles give the same input two different outputs in
  different examples, which means no single consistent rule exists in the data.

This is an information ceiling. When the answer is not determined by the
information provided, no model and no algorithm can recover it reliably. This is
why every team on the public leaderboard tops out near 0.87 to 0.90, and why
0.93 is not reachable by anyone. Being able to state and prove this limit is one
of the most valuable parts of the project.

## 5. Key idea three: LoRA fine-tuning

The base model has about 30 billion parameters. Fully retraining it is far too
expensive. LoRA (Low Rank Adaptation) freezes the base model and trains small
add on matrices instead. A LoRA "rank" of 32 means each adapted layer gets two
thin matrices whose product is added to the original weight. Only those small
matrices train, which is a tiny fraction of the parameters.

Two choices mattered a lot:

- Which layers to adapt. Adapting only the attention layers changed almost
  nothing (score stayed at the base level, 0.54), because this model has only 6
  attention layers out of 52. Adding the MLP and expert layers gave the adapter
  real capacity (869 million trainable parameters) and the score moved.
- Rank is capped at 32 by the competition rules.

## 6. Key idea four: why the model needs a large GPU

A 30 billion parameter model in bf16 (2 bytes per parameter) needs about 60 to
65 GB just to hold the weights. Training also needs memory for gradients, the
optimizer state, and activations. That total does not fit on a typical 24 GB or
40 GB GPU, and a large adapter does not even fit on an 80 GB A100. The working
runs used an H200 (141 GB). Two tricks reduce memory: gradient checkpointing
(recompute activations during the backward pass instead of storing them) and an
8 bit optimizer (store optimizer state in 8 bits instead of 32).

## 7. The training pipeline, end to end

1. Download the base model.
2. Build the training data: real rows plus solver generated rows, each row is a
   prompt, a reasoning trace ending in `\boxed{answer}`, and the answer.
3. Run supervised fine tuning (SFT): show the model prompt and trace, train it to
   produce the trace.
4. Save the LoRA adapter (two files: adapter_config.json and the weights).
5. Zip the adapter into submission.zip and submit. The grader loads the base
   model plus the adapter and scores it on the hidden test set.

## 8. Results and what each lever did

- Base model, no adapter: about 0.53.
- Attention only LoRA: 0.54 (no real change; too little capacity).
- Attention plus MLP LoRA, 4,000 rows, 1 epoch: 0.60 (capacity helped).
- Attention plus MLP LoRA, full 9,500 rows, 2 epochs: 0.72 (more data helped).

The climb 0.54 to 0.60 to 0.72 shows each lever (capacity, then data) paying off.
The remaining gap to 0.87 is the bitmanip and symbol information ceiling.

## 9. Glossary

- Inductive reasoning: inferring a rule from examples and applying it.
- LoRA: training small add on matrices while the base model stays frozen.
- Rank: the size of the LoRA matrices. Higher rank means more capacity.
- bf16: a 16 bit number format, 2 bytes per value.
- MoE (mixture of experts): layers split into many "expert" sub networks, only
  some active per token. This model is a hybrid of Mamba, attention, and MoE.
- Mamba: a state space layer, an alternative to attention.
- Gradient checkpointing: trade compute for memory by recomputing activations.
- Optimizer state: extra memory Adam keeps per trainable parameter.
- Epoch: one full pass over the training data.
- SFT: supervised fine tuning on prompt and target pairs.
- Information ceiling: the limit set by how much the data actually determines.

## 10. Questions you should be able to answer (to own this project)

- Why did attention only LoRA fail to move the score?
- Why does a 30B model need an 80 GB or larger GPU to train?
- What is the difference between the training loss going down and the test score
  going up, and why did loss reach 0.05 while the score was 0.72?
- Why is bitmanip only about 40 percent solvable from the examples?
- Why can a correct solver generate better training data than a teacher model?
