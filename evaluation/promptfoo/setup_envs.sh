#!/bin/bash
set -euo pipefail

PROJECT=/usr/prakt/s0030/projects/tikzcodegenerator

CLIENT_DIR=$PROJECT/evaluation/models/ollama/ollama_client
CLIENT_ENV=ollama-client
PROMPTFOO_ENV=promptfoo

source /usr/prakt/s0030/miniconda3/etc/profile.d/conda.sh

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

echo "Installing Poppler and Ghostscript into $PROMPTFOO_ENV..."

conda install -y -n "$PROMPTFOO_ENV" -c conda-forge \
  ghostscript \
  poppler

echo "Installing promptfoo..."

conda run -n "$PROMPTFOO_ENV" npm install -g promptfoo --no-fund

echo "Cleaning old ML packages..."

conda run -n "$PROMPTFOO_ENV" pip uninstall -y \
  numpy \
  torch \
  torchvision \
  torchaudio \
  torchmetrics \
  dreamsim || true

echo "Installing NumPy 1.x..."

conda run -n "$PROMPTFOO_ENV" pip install \
  "numpy==1.26.4"

echo "Installing PyTorch stack for RTX 6000 / Turing / GPU_CC 7.5..."

conda run -n "$PROMPTFOO_ENV" pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "Installing evaluation dependencies..."

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

echo "Checking environments..."

conda run -n "$CLIENT_ENV" python --version

conda run -n "$PROMPTFOO_ENV" python --version
conda run -n "$PROMPTFOO_ENV" node --version
conda run -n "$PROMPTFOO_ENV" npm --version

echo "Checking NumPy / Torch / Torchvision..."

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

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.ones(1, device="cuda")
    print("cuda tensor test:", x + 1)
PY

echo "Checking Poppler/Ghostscript tools..."

conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdftoppm && pdftoppm -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdfinfo && pdfinfo -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdftocairo && pdftocairo -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which gs && gs --version'

# Optional: make "ghostscript" command available if your code expects that name.
PROMPTFOO_ENV_DIR=/usr/prakt/s0030/miniconda3/envs/promptfoo
if [[ -x "$PROMPTFOO_ENV_DIR/bin/gs" ]]; then
  ln -sf "$PROMPTFOO_ENV_DIR/bin/gs" "$PROMPTFOO_ENV_DIR/bin/ghostscript"
fi

conda run -n "$PROMPTFOO_ENV" bash -lc 'which ghostscript && ghostscript --version'

PROMPTFOO_TMP_HOME=/tmp/promptfoo-check-$USER
rm -rf "$PROMPTFOO_TMP_HOME"
mkdir -p "$PROMPTFOO_TMP_HOME"

PROMPTFOO_DISABLE_WAL_MODE=true \
PROMPTFOO_CONFIG_DIR="$PROMPTFOO_TMP_HOME" \
conda run -n "$PROMPTFOO_ENV" promptfoo --version

echo "Done."