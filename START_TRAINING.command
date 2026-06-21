#!/bin/bash
# ============================================================
#  Nemotron LoRA — one-click training launcher
#  Double-click this file in Finder. It opens Terminal,
#  sets everything up, and starts the GPU training run.
#  It pauses ONCE for a browser login (Modal). That's the
#  only thing you have to do by hand.
# ============================================================

PY=/opt/miniconda3/bin/python3
REPO="$HOME/Documents/nemotron-reasoning"

cd "$REPO" || { echo "Repo not found at $REPO"; read -p "Press Enter to close."; exit 1; }

echo "============================================================"
echo "  Nemotron LoRA training launcher"
echo "  Repo: $REPO"
echo "============================================================"
echo

# --- 1. Modal installed? -------------------------------------------------
echo "[1/5] Checking Modal CLI..."
if ! "$PY" -m modal --version >/dev/null 2>&1; then
  echo "      Installing Modal..."
  "$PY" -m pip install --quiet modal || { echo "pip install modal FAILED"; read -p "Press Enter to close."; exit 1; }
fi
echo "      OK: $("$PY" -m modal --version)"
echo

# --- 2. Kaggle credentials present? -------------------------------------
echo "[2/5] Checking Kaggle credentials..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "      ERROR: ~/.kaggle/kaggle.json is missing."
  echo "      Get it from kaggle.com/settings -> Create New API Token,"
  echo "      then move it to ~/.kaggle/ and run this again."
  read -p "Press Enter to close."; exit 1
fi
echo "      OK"
echo

# --- 3. Modal login (browser) -------------------------------------------
echo "[3/5] Checking Modal login..."
if [ ! -f "$HOME/.modal.toml" ]; then
  echo "      >>> A BROWSER WINDOW WILL OPEN."
  echo "      >>> Sign in / sign up (free), click Authorize, then return here."
  echo
  "$PY" -m modal token new || { echo "Modal login failed."; read -p "Press Enter to close."; exit 1; }
else
  echo "      Already logged in."
fi
echo

# --- 4. Create the Kaggle secret on Modal (idempotent) ------------------
echo "[4/5] Configuring secret 'nemotron-secrets'..."
KU=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
KK=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['key'])")
"$PY" -m modal secret create nemotron-secrets \
    KAGGLE_USERNAME="$KU" KAGGLE_KEY="$KK" HF_TOKEN="dummy-ok" 2>/dev/null \
  && echo "      Secret created." \
  || echo "      Secret already exists (continuing)."
echo

# --- 5. Launch the run ---------------------------------------------------
echo "[5/5] Launching training on a Modal A100-80GB."
echo "      This takes about 3-4 hours. Leave this window OPEN."
echo "      Watch for the 'enable_thinking' and 'recon coverage' log lines."
echo "------------------------------------------------------------"
"$PY" -m modal run scripts/run_modal.py
STATUS=$?
echo "------------------------------------------------------------"
if [ $STATUS -eq 0 ]; then
  echo "  DONE. submission.zip should now be in:"
  echo "  $REPO/submission.zip"
  echo
  echo "  Next: submit it with"
  echo "  kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission.zip -m \"v1\""
else
  echo "  The run exited with an error (code $STATUS)."
  echo "  Scroll up, copy the last ~20 lines, and paste them to Claude."
fi
echo "============================================================"
read -p "Press Enter to close this window."
