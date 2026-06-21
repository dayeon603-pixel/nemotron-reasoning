# Cheap or Free Compute, and How to Turn This Into a Legit EECS Project

This document answers two questions honestly: how to train models with little or
no money, and how to turn this work into something genuinely useful for an EECS
application or a thesis. The short version: do not chase the 30B model on free
hardware. Reproduce the same method on a smaller model you can fully run and own.

## 1. Why free hardware cannot train the 30B model

- The 30B model in bf16 needs about 60 to 65 GB just to hold the weights, and
  training needs more on top for gradients, optimizer state, and activations.
- Kaggle free GPUs: two T4 (16 GB each) or one P100 (16 GB). Far too small.
- Colab free: one T4 (16 GB). Too small. Colab Pro: an A100 40 GB, still too
  small for a 30B without extreme tricks.
- So there is no free path to training this specific 30B model. Free tiers only
  run the scoring, which the competition already does for you.

## 2. The key move: use a smaller open model

The method in this project (reverse engineer the families, write solvers, build
clean training data, fine tune with LoRA, measure honestly) does not depend on
the model being 30B. You can run the exact same pipeline on a small open model
that fits free hardware, for example a 1B to 8B model such as Qwen, Llama, or
Gemma in the small sizes. With 4 bit loading (QLoRA), even a 7B or 8B model fits
on a free Kaggle or Colab GPU.

Why this is better for you, not just cheaper:

- You can run every step yourself, end to end, with no cloud account and no bill.
- You fully own and understand it, which is what matters for an application or a
  thesis defense.
- The interesting parts (the solvers, the data centric method, the information
  ceiling proof) are identical. The model size is a footnote.

## 3. Free and cheap options, ranked

1. Kaggle Notebooks (free): about 30 GPU hours per week on a T4 or P100. Good for
   a 1B to 8B model with 4 bit QLoRA. The data already lives on Kaggle.
2. Google Colab free: one T4, good for 1B to 3B models, or 7B with QLoRA and
   short sequences. Sessions time out, so checkpoint often.
3. Colab Pro (about 10 dollars per month): A100 40 GB, comfortable for 7B to 13B
   with QLoRA.
4. vast.ai or RunPod spot (cheapest real 80 GB): an A100 80 GB is roughly 0.8 to
   1.3 dollars per hour. A full small run is a few dollars. More setup (SSH).
5. Modal: convenient Python workflow, free monthly credit, then pay per second.
   This is what this project used. Reliable but pricier per hour than spot.

## 4. Memory tricks that let bigger models fit on smaller GPUs

- QLoRA (4 bit base): loads the frozen model in 4 bit, cutting weight memory by
  about four times. A 7B model drops from about 14 GB to about 4 GB.
- Gradient checkpointing: recompute activations in the backward pass.
- 8 bit optimizer: store optimizer state in 8 bits.
- Short sequences and batch size 1 with gradient accumulation.

Together these let a 7B or 8B model train on a single 16 GB free GPU.

## 5. A concrete free plan you can actually run and own

1. Pick a small open model that fits free hardware (start around 1B to 3B, then
   try 7B with QLoRA).
2. Reuse the solvers and the data builder in this repo to make clean training
   data for the solvable families.
3. Fine tune with LoRA on Kaggle or Colab. Checkpoint to your Google Drive or a
   Kaggle dataset so a timeout does not lose progress.
4. Evaluate with the metric exact harness already in this repo.
5. Write down per family accuracy and the information ceiling analysis.

This costs nothing, runs in hours, and produces a result you ran yourself.

## 6. Turning this into a legit EECS artifact

Pick one of these. All are honest and do not require a high score.

- A clean public repository plus a short technical writeup. Title it around the
  method and the analysis, for example "Data centric fine tuning and solvability
  limits of an inductive reasoning benchmark." Lead with the reverse engineering
  and the information ceiling proof, not the leaderboard number.
- A short research note or thesis style report. A strong, honest claim is: "On
  this benchmark, two of six task families are under determined by the examples
  provided, which sets a hard accuracy ceiling; the remaining families are fully
  solvable and a small fine tuned model recovers them." That is a real finding.
- A reproducible notebook that trains a small model free and shows the method.
  Reproducibility is itself a strong signal.

What not to do: do not present a late, unranked submission as a competition
placement or award, and do not claim a score you cannot reproduce and explain.

## 7. Similar competitions and venues to reuse the method

- Other reasoning or fine tuning challenges on Kaggle or Hugging Face, where the
  same solver plus data centric LoRA method transfers directly.
- High school and undergraduate research venues for a short paper or poster about
  the solvability analysis.
- A personal blog or arXiv style preprint for the writeup, which gives you a
  link you can cite in applications.

## 8. Honest bottom line

The cheapest and most legitimate path is to reproduce this on a small model you
run yourself, for free, and to write up the method and the information ceiling
finding. The 30B run proved the pipeline works at scale; a free small model
reproduction is what makes it truly yours and defensible.
