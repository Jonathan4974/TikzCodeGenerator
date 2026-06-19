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

echo "Installing system-like tools into $PROMPTFOO_ENV..."

conda install -y -n "$PROMPTFOO_ENV" -c conda-forge \
  ghostscript \
  poppler

echo "Installing promptfoo..."

conda run -n "$PROMPTFOO_ENV" npm install -g promptfoo --no-fund

echo "Installing Python dependencies..."

conda run -n "$PROMPTFOO_ENV" pip install \
  pandas \
  pyarrow \
  pillow \
  numpy \
  scikit-learn \
  scikit-image \
  nltk \
  crystalbleu \
  latexcodec \
  torchmetrics \
  pygments \
  sacremoses \
  torch \
  torchvision \
  transformers \
  sentencepiece \
  dreamsim

echo "Checking environments..."

conda run -n "$CLIENT_ENV" python --version
conda run -n "$PROMPTFOO_ENV" node --version
conda run -n "$PROMPTFOO_ENV" npm --version

echo "Checking Poppler/Ghostscript tools..."

conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdftoppm && pdftoppm -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdfinfo && pdfinfo -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which pdftocairo && pdftocairo -v'
conda run -n "$PROMPTFOO_ENV" bash -lc 'which gs && gs --version'

PROMPTFOO_TMP_HOME=/tmp/promptfoo-check-$USER
rm -rf "$PROMPTFOO_TMP_HOME"
mkdir -p "$PROMPTFOO_TMP_HOME"

PROMPTFOO_DISABLE_WAL_MODE=true \
PROMPTFOO_CONFIG_DIR="$PROMPTFOO_TMP_HOME" \
conda run -n "$PROMPTFOO_ENV" promptfoo --version

echo "Done."