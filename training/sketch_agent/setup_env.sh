#!/bin/bash
set -euo pipefail

# conda env setup for the sketch-agent SDXL+ControlNet+LoRA training job.

I9_USER="${I9_USER:-s0031}"
PRAKT_DIR="/usr/prakt/$I9_USER"

ENV_NAME="sketch-agent"
CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"

echo "Setting up $ENV_NAME..."

conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || \
  conda create -y -n "$ENV_NAME" python=3.11

# similar to evaluation/start_script/setup_envs.sh's promptfoo env
conda run -n "$ENV_NAME" pip install --upgrade pip setuptools wheel

conda run -n "$ENV_NAME" pip install \
  "numpy==1.26.4"

conda run -n "$ENV_NAME" pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

conda run -n "$ENV_NAME" pip install \
  diffusers \
  transformers \
  accelerate \
  peft \
  datasets \
  huggingface_hub \
  opencv-python-headless \
  torchmetrics \
  dreamsim \
  tensorboard \
  pillow \
  scipy \
  sentencepiece \
  protobuf \
  pymupdf \
  requests

echo "Checking setup..."

conda run -n "$ENV_NAME" python - <<'PY'
import torch
import diffusers
import transformers
import accelerate
import peft
import cv2

print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("opencv:", cv2.__version__)
PY

echo "Done."
