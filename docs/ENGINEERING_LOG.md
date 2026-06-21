# Engineering Log: Every Approach Attempted

This is an honest record of the full debugging journey: each problem, its root
cause, and the fix. Real machine learning systems work is mostly this. The value
of this log is that it shows how each failure was diagnosed and resolved.

## Compute and environment setup

1. Problem: no local GPU big enough. A 30B model needs 80 GB or more.
   Fix: rent a cloud GPU. Modal was chosen for the convenience of a Python only
   workflow. The free credit covered early runs.

2. Problem: `modal.Mount` no longer exists in Modal 1.x.
   Root cause: the API changed; mounts moved onto the image.
   Fix: attach the repo with `image.add_local_dir(...)`.

3. Problem: mamba_ssm failed to build with "No module named torch".
   Root cause: pip builds wheels in an isolated environment that cannot see the
   already installed torch.
   Fix: install mamba_ssm and causal_conv1d with `--no-build-isolation`.

4. Problem: mamba_ssm build failed with a CUDA version mismatch (12.6 vs 13.0).
   Root cause: installing vLLM pulled a torch built for CUDA 13, while the base
   image compiler is CUDA 12.6.
   Fix: stop reinstalling torch. Use the NVIDIA base image's own torch (matched
   to its CUDA 12.6) and drop vLLM from the training image. vLLM was only used
   for an optional smoke test.

5. Problem: `modal run` failed with "unrecognized arguments".
   Root cause: the entry point used its own argparse, which conflicted with
   Modal's command line.
   Fix: make the entry point take a normal function parameter; Modal exposes it
   as a flag automatically.

## Data and model loading

6. Problem: kagglehub crashed on import (missing get_access_token_from_env).
   Root cause: every modern kagglehub needs a newer kagglesdk than the package
   mirror offers (0.1.28).
   Fix: stop using kagglehub. Download the model with the kaggle command line
   tool instead, which works with the available kagglesdk.

7. Problem: the trainer still imported kagglehub and crashed again.
   Root cause: the training script independently called kagglehub.
   Fix: load the model from a local path passed by an environment variable, with
   kagglehub only as a last resort.

8. Problem: the tokenizer would not load (Unrecognized configuration class
   NemotronHConfig for AutoTokenizer).
   Root cause one: the transformers version. Pinned to 4.56.2, the version
   verified by NVIDIA for this model.
   Root cause two (the real one): the model directory had no tokenizer files at
   all. The kaggle tool's `--untar` flag had only partially extracted the
   archive, dropping the tokenizer and modeling files.
   Fix: extract the downloaded archive fully with Python's tarfile instead of
   trusting the partial `--untar`.

## Training memory and time

9. Problem: out of memory on an A100 80 GB once the adapter included the MLP and
   expert layers (869 million trainable parameters).
   Fix attempt: gradient checkpointing plus an 8 bit optimizer.

10. Problem: the 8 bit optimizer (bitsandbytes) crashed at the first step with a
    triton import error.
    Root cause: bitsandbytes 0.49.2 conflicts with the image's triton version.
    Fix: drop bitsandbytes, move to an H200 (141 GB) which has the memory headroom
    for plain torch AdamW plus gradient checkpointing.

11. Problem: a leftover `cache_dir` variable caused a NameError after refactoring
    the model download.
    Fix: remove the stale reference.

12. Problem: the first full run was interrupted at training step 40.
    Root cause: a local network drop disconnected the client; a non detached run
    stops when the client disconnects.
    Fix: launch with `--detach` and save the adapter to the persistent volume so
    the result survives a disconnect.

13. Problem: a two epoch run on the full data would not finish inside the 23 hour
    timeout (about 27 seconds per step, 3,875 steps needed about 29 hours).
    Fix: size the run correctly. Either fewer rows, or one epoch, or accept the
    measured step rate when planning.

## The billing leak (important)

14. Problem: Modal kept billing after runs appeared finished.
    Root cause: a three part chain. The post training smoke test tried to import
    vLLM, which is not in the image, and it raised a fatal error. The training
    function had a retry setting, so the fatal error caused Modal to re run the
    entire roughly 12 hour training. Because the run was detached, this retry
    loop kept running on the server independent of the local session.
    Fix: make the smoke test skip gracefully when vLLM is absent (log and return,
    never raise), and set retries to 0 so a failure can never auto re run an
    expensive job. New operating rule: after every run, explicitly stop the app
    and confirm the active container list is empty.

## Results recorded

- Base model: about 0.53.
- Attention only LoRA: 0.54.
- Attention plus MLP, 4,000 rows, 1 epoch: 0.60.
- Attention plus MLP, full 9,500 rows, 2 epochs: 0.72.

## Lessons

- Match library versions to the base image; do not let one package silently
  upgrade torch, numpy, or transformers.
- Verify what is actually on disk (the missing tokenizer files were invisible
  until the directory was listed).
- Never combine a fatal post step with automatic retries on an expensive job.
- Always confirm a cloud job has stopped, not just that it looks finished.
- The score ceiling was set by the data, not by effort or money.
