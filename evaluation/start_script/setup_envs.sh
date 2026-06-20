#!/bin/bash
set -euo pipefail

I9_USER="${I9_USER:-s0030}"

PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"

CLIENT_DIR="$PROJECT/evaluation/models/ollama/ollama_client"
CLIENT_ENV="ollama-client"
PROMPTFOO_ENV="promptfoo"

CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"
PROMPTFOO_ENV_DIR="$PRAKT_DIR/miniconda3/envs/$PROMPTFOO_ENV"

source "$CONDA_SH"

echo "Setting up $CLIENT_ENV..."

conda env list | awk '{print $1}' | grep -qx "$CLIENT_ENV" || \
  conda create -y -n "$CLIENT_ENV" python=3.11

conda run -n "$CLIENT_ENV" pip install -r "$CLIENT_DIR/requirements.txt"

echo "Setting up $PROMPTFOO_ENV..."

conda env list | awk '{print $1}' | grep -qx "$PROMPTFOO_ENV" || \
  conda create -y -n "$PROMPTFOO_ENV" -c conda-forge \
    python=3.11 \
    nodejs=22 \
    pip

conda install -y -n "$PROMPTFOO_ENV" -c conda-forge \
  ghostscript \
  poppler

conda run -n "$PROMPTFOO_ENV" npm install -g promptfoo --no-fund

conda run -n "$PROMPTFOO_ENV" pip uninstall -y \
  numpy \
  torch \
  torchvision \
  torchaudio \
  torchmetrics \
  dreamsim || true

conda run -n "$PROMPTFOO_ENV" pip install \
  "numpy==1.26.4"

conda run -n "$PROMPTFOO_ENV" pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

conda run -n "$PROMPTFOO_ENV" pip install \
  pandas \
  pyarrow \
  pillow \
  scikit-learn \
  scikit-image \
  nltk \
  crystalbleu \
  latexcodec \
  torchmetrics \
  pygments \
  sacremoses \
  transformers \
  sentencepiece \
  peft \
  dreamsim

if [[ -x "$PROMPTFOO_ENV_DIR/bin/gs" ]]; then
  ln -sf "$PROMPTFOO_ENV_DIR/bin/gs" "$PROMPTFOO_ENV_DIR/bin/ghostscript"
fi

PROMPTFOO_TMP_HOME="/tmp/promptfoo-check-$I9_USER"
rm -rf "$PROMPTFOO_TMP_HOME"
mkdir -p "$PROMPTFOO_TMP_HOME"

echo "Checking setup..."

conda run -n "$CLIENT_ENV" python --version

conda run -n "$PROMPTFOO_ENV" python - <<'PY'
import numpy
import torch
import torchvision
import torchmetrics

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("torchmetrics:", torchmetrics.__version__)
print("cuda available:", torch.cuda.is_available())
PY

conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdftoppm && pdftoppm -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which gs && gs --version'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which ghostscript && ghostscript --version'

PROMPTFOO_DISABLE_WAL_MODE=true \
PROMPTFOO_CONFIG_DIR="$PROMPTFOO_TMP_HOME" \
conda run -n "$PROMPTFOO_ENV" promptfoo --version

echo "Done."