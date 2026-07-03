#!/usr/bin/env bash
# Reproducible Keras-2 environment for SHIFT-SURV.
#
# DANN/CDAN come from `adapt==0.4.5`, which targets tf.keras (Keras 2).
# TensorFlow >=2.16 ships Keras 3 and silently degrades those models (layers get
# marked "built" without a real build()). requirements.txt therefore pins
# tensorflow<2.16 + numpy<2, which needs Python 3.10/3.11 (no TF 2.15 wheels for 3.12+).
#
# Usage:
#   bash setup_env.sh [ENV_NAME]        # default env name: shift310
#   conda activate <ENV_NAME>
set -euo pipefail

ENV_NAME="${1:-shift310}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

# An active virtualenv (e.g. a Python 3.13 venv) hijacks pip and targets the wrong
# interpreter — neutralise it so installs land in the conda env we create below.
unset VIRTUAL_ENV || true

CONDA_BASE="$(conda info --base)"

# Create the env only if missing (idempotent re-runs).
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=== Conda env '$ENV_NAME' already exists — reusing ==="
else
  echo "=== Creating conda env '$ENV_NAME' (Python 3.10) ==="
  conda create -y -n "$ENV_NAME" python=3.10
fi

# Resolve the env's REAL prefix (user envs often live in ~/.conda/envs, not
# $CONDA_BASE/envs when the base install is not user-writable).
ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1==n{print $NF}')"
if [[ -z "${ENV_PREFIX:-}" || ! -x "$ENV_PREFIX/bin/python" ]]; then
  for cand in "$CONDA_BASE/envs/$ENV_NAME" "$HOME/.conda/envs/$ENV_NAME"; do
    [[ -x "$cand/bin/python" ]] && ENV_PREFIX="$cand" && break
  done
fi

# Install with the env's explicit interpreter (bypasses PATH / venv interference).
PY="$ENV_PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: python not found for env '$ENV_NAME' (looked in '$ENV_PREFIX')"; exit 1; }
echo "=== Using interpreter: $PY ==="
"$PY" --version
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"

echo "=== Verifying Keras 2 + adapt DA layers ==="
"$PY" - <<'PYCODE'
import keras, tensorflow as tf
from adapt.feature_based import CORAL, DANN, CDAN
assert keras.__version__.startswith("2."), f"Expected Keras 2, got {keras.__version__} — DANN/CDAN will be unreliable."
print(f"OK  keras={keras.__version__}  tensorflow={tf.__version__}")
PYCODE

echo ""
echo "=== Done. Now run:  deactivate 2>/dev/null; conda activate $ENV_NAME ==="
