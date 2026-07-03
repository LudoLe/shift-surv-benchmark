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

echo "=== Creating conda env '$ENV_NAME' (Python 3.10) ==="
conda create -y -n "$ENV_NAME" python=3.10

echo "=== Installing pinned requirements ==="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"

echo "=== Verifying Keras 2 + adapt DA layers ==="
conda run -n "$ENV_NAME" python - <<'PY'
import keras, tensorflow as tf
from adapt.feature_based import CORAL, DANN, CDAN
major = int(keras.__version__.split(".")[0])
assert major == 2, f"Expected Keras 2, got {keras.__version__} — DANN/CDAN will be unreliable."
print(f"OK  keras={keras.__version__}  tensorflow={tf.__version__}")
PY

echo ""
echo "=== Done. Activate with:  conda activate $ENV_NAME ==="
