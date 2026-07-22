#!/bin/bash
set -euo pipefail

# conda env setup for the sketch-agent data-augmentation

I9_USER="${I9_USER:-s0031}"
PRAKT_DIR="/usr/prakt/$I9_USER"

ENV_NAME="sketch-agent"
CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"

echo "Setting up $ENV_NAME..."

conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || \
  conda create -y -n "$ENV_NAME" python=3.11

conda run -n "$ENV_NAME" pip install --upgrade pip setuptools wheel

conda run -n "$ENV_NAME" pip install \
  "numpy==1.26.4"

conda run -n "$ENV_NAME" pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

conda run -n "$ENV_NAME" pip install \
  "diffusers==0.39.0" \
  "transformers==5.13.0" \
  "datasets==5.0.0" \
  huggingface_hub \
  pillow \
  scipy \
  pytest

echo "Checking setup..."

conda run -n "$ENV_NAME" python - <<'PY'
import torch
import diffusers
import transformers
import datasets

print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
PY

echo "Done."
