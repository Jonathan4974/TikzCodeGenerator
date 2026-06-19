#!/bin/bash
#SBATCH --job-name=ollama-promptfoo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=0-02:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-%j.out
#SBATCH --error=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-%j.err

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

echo "Job started on $(hostname)"
date
nvidia-smi

PROJECT=/usr/prakt/s0030/projects/tikzcodegenerator
LOG_DIR=$PROJECT/evaluation/logs

MODEL=gemma4:e2b-it-qat
CONTAINER_DIR=$PROJECT/evaluation/models/ollama
MODEL_STORE=/usr/prakt/s0030/projects/models/ollama_model_store
IMAGE=$CONTAINER_DIR/ollama+ollama+latest.sqsh

CLIENT_DIR=$PROJECT/evaluation/models/ollama/ollama_client
CLIENT_ENV=ollama-client

PROMPTFOO_ENV=promptfoo
PROMPTFOO_ENV_DIR=/usr/prakt/s0030/miniconda3/envs/promptfoo
PROMPTFOO_CONFIG=$PROJECT/evaluation/promptfoo/configs/image_to_tikz_promptfooconfig_gemma_latest.yaml
PROMPTFOO_PORT=15500

PROMPTFOO_BASE_DIR=$PROJECT/evaluation/promptfoo
PROMPTFOO_PERSIST_DIR=$PROMPTFOO_BASE_DIR/promptfoo-db
PROMPTFOO_RUN_DIR=/tmp/promptfoo-${USER}-${SLURM_JOB_ID}

TEXLIVE_DIR=/usr/prakt/s0030/projects/tools/texlive/2026

TEST_IMAGE=/usr/prakt/s0030/projects/data/benchmark_data/images/row_00000000.png
TEST_PROMPT="Erkläre kurz, was in diesem Bild zu sehen ist."

mkdir -p "$MODEL_STORE" "$LOG_DIR"
mkdir -p "$PROJECT/evaluation/cache/huggingface"
mkdir -p "$PROJECT/evaluation/cache/torch"
mkdir -p "$PROJECT/evaluation/cache/dreamsim"
mkdir -p "$PROMPTFOO_BASE_DIR" "$PROMPTFOO_PERSIST_DIR" "$PROMPTFOO_RUN_DIR"

# Continue from previous Promptfoo DB if it exists.
cp -a "$PROMPTFOO_PERSIST_DIR"/. "$PROMPTFOO_RUN_DIR"/ 2>/dev/null || true

# Make user-local TeX Live and promptfoo env tools available on the compute node.
export PATH="$TEXLIVE_DIR/bin/x86_64-linux:$PROMPTFOO_ENV_DIR/bin:$PATH"

echo "Checking LaTeX/TikZ tools on $(hostname)..."
for cmd in pdflatex xelatex lualatex latexmk kpsewhich dvisvgm ghostscript gs pdfinfo pdftoppm pdftocairo pdftotext; do
  echo -n "$cmd: "
  command -v "$cmd" || true
done

if command -v kpsewhich >/dev/null 2>&1; then
  for f in tikz.sty pgf.sty standalone.cls preview.sty; do
    echo -n "$f: "
    kpsewhich "$f" || true
  done
fi

enroot start \
  --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --mount "$MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$LOG_DIR/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

cleanup() {
  echo "Stopping services..."

  kill "${PROMPTFOO_PID:-}" 2>/dev/null || true
  kill "${CLIENT_PID:-}" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null || true

  echo "Copying Promptfoo DB to $PROMPTFOO_PERSIST_DIR"
  mkdir -p "$PROMPTFOO_PERSIST_DIR"
  cp -a "$PROMPTFOO_RUN_DIR"/. "$PROMPTFOO_PERSIST_DIR"/ 2>/dev/null || true
  rm -rf "$PROMPTFOO_RUN_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama server..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Ollama is ready."
    break
  fi
  sleep 2
done

curl -s http://127.0.0.1:11434/api/tags >/dev/null

if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"$MODEL\""; then
  curl http://127.0.0.1:11434/api/pull \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\"}"
fi

source /usr/prakt/s0030/miniconda3/etc/profile.d/conda.sh

cd "$CLIENT_DIR"

OLLAMA_URL=http://127.0.0.1:11434/api/chat \
OLLAMA_MODEL="$MODEL" \
conda run -n "$CLIENT_ENV" \
  uvicorn app:app --host 127.0.0.1 --port 8004 \
  > "$LOG_DIR/ollama-client-${SLURM_JOB_ID}.log" 2>&1 &

CLIENT_PID=$!

echo "Waiting for Ollama client..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:8004/docs >/dev/null; then
    echo "Client is ready."
    break
  fi
  sleep 2
done

curl http://127.0.0.1:8004/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$TEST_PROMPT\",
    \"image\": \"$TEST_IMAGE\"
  }"

echo

cd "$PROJECT"

export PROMPTFOO_DISABLE_TELEMETRY=1
export PROMPTFOO_DISABLE_UPDATE=1
export PROMPTFOO_DISABLE_WAL_MODE=true
export PROMPTFOO_CONFIG_DIR="$PROMPTFOO_RUN_DIR"

export PYTHONPATH="$PROJECT:$PROJECT/evaluation/promptfoo:${PYTHONPATH:-}"

export HF_HOME=$PROJECT/evaluation/cache/huggingface
export TORCH_HOME=$PROJECT/evaluation/cache/torch
export DREAMSIM_CACHE_DIR=$PROJECT/evaluation/cache/dreamsim

export OLLAMA_CLIENT_URL=http://127.0.0.1:8004/generate

export IMAGE_MANIFEST=/usr/prakt/s0030/projects/data/benchmark_data/small_manifest.csv
export IMAGES_DIR=/usr/prakt/s0030/projects/data/benchmark_data/images
export REFERENCES_DIR=/usr/prakt/s0030/projects/data/benchmark_data/references
export CAPTIONS_DIR=/usr/prakt/s0030/projects/data/benchmark_data/captions

echo "Checking Promptfoo..."
conda run -n "$PROMPTFOO_ENV" node --version
conda run -n "$PROMPTFOO_ENV" npm --version
conda run -n "$PROMPTFOO_ENV" promptfoo --version

conda run -n "$PROMPTFOO_ENV" \
  promptfoo eval \
    -c "$PROMPTFOO_CONFIG" \
    -j 1 \
    --watch \
  > "$LOG_DIR/promptfoo-eval-${SLURM_JOB_ID}.log" 2>&1 &

PROMPTFOO_PID=$!

sleep 5

conda run -n "$PROMPTFOO_ENV" \
  promptfoo view \
    --port "$PROMPTFOO_PORT" \
    --no \
  > "$LOG_DIR/promptfoo-view-${SLURM_JOB_ID}.log" 2>&1

echo
echo "Job finished."
date