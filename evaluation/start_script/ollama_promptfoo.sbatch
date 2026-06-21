#!/bin/bash
#SBATCH --job-name=ollama-promptfoo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint=GPU_MODEL:rtx_6000
#SBATCH --time=0-08:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=NONE
#SBATCH --output=evaluation/logs/ollama-%j.out
#SBATCH --error=evaluation/logs/ollama-%j.err

set -euo pipefail

I9_USER="${I9_USER:-s0030}"

PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"

LOG_DIR="$PROJECT/evaluation/logs"
MODEL="${MODEL:-gemma4:e2b-it-qat}"
TYPE="${TYPE:-img-tikz}"
MODEL_SAFE="${MODEL//:/_}"

CONTAINER_DIR="$PROJECT/evaluation/models/ollama"
OLLAMA_MODEL_STORE="$PRAKT_DIR/projects/models/ollama_model_store"
MODEL_STORE="$PRAKT_DIR/projects/models"
IMAGE="$CONTAINER_DIR/ollama+ollama+latest.sqsh"

CLIENT_DIR="$PROJECT/evaluation/models/ollama/ollama_client"
CLIENT_ENV="ollama-client"

PROMPTFOO_ENV="promptfoo"
PROMPTFOO_ENV_DIR="$PRAKT_DIR/miniconda3/envs/$PROMPTFOO_ENV"
PROMPTFOO_CONFIG="${PROMPTFOO_CONFIG:?PROMPTFOO_CONFIG must be set}"
MANIFEST_SPLIT="${MANIFEST_SPLIT:-0}"
PROMPTFOO_PORT="15500"

PROMPTFOO_BASE_DIR="$PROJECT/evaluation/promptfoo"
RESULT_DIR="$PROJECT/evaluation/results/$MODEL_SAFE/$TYPE/split_${MANIFEST_SPLIT}"
PROMPTFOO_PERSIST_DIR="$RESULT_DIR/promptfoo-db"
PROMPTFOO_RUN_DIR="/tmp/promptfoo-${I9_USER}-${SLURM_JOB_ID}"

TEXLIVE_DIR="$PRAKT_DIR/projects/full_latex/texlive/2026"
CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

DATA_DIR="$PRAKT_DIR/projects/data/benchmark_data"
IMAGE_MANIFEST="$DATA_DIR/small_manifest.csv"
IMAGES_DIR="$DATA_DIR/images"
REFERENCES_DIR="$DATA_DIR/references"
CAPTIONS_DIR="$DATA_DIR/captions"
GENERATED_IMAGE_DIR="$RESULT_DIR/generated_images"

mkdir -p \
  "$LOG_DIR" \
  "$OLLAMA_MODEL_STORE" \
  "$MODEL_STORE/huggingface" \
  "$MODEL_STORE/torch" \
  "$MODEL_STORE/dreamsim" \
  "$MODEL_STORE/crystalbleu_scitikzer" \
  "$PROMPTFOO_PERSIST_DIR" \
  "$PROMPTFOO_RUN_DIR" \
  "$GENERATED_IMAGE_DIR"

cp -a "$PROMPTFOO_PERSIST_DIR"/. "$PROMPTFOO_RUN_DIR"/ 2>/dev/null || true

export PATH="$TEXLIVE_DIR/bin/x86_64-linux:$PROMPTFOO_ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

export PROMPTFOO_DISABLE_TELEMETRY=1
export PROMPTFOO_DISABLE_UPDATE=1
export PROMPTFOO_DISABLE_WAL_MODE=true
export PROMPTFOO_CONFIG_DIR="$PROMPTFOO_RUN_DIR"

export PYTHONPATH="$PROJECT:$PROJECT/evaluation/promptfoo:${PYTHONPATH:-}"

export HF_HOME="$MODEL_STORE/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$MODEL_STORE/cache"

export TORCH_HOME="$MODEL_STORE/torch"
export DREAMSIM_CACHE_DIR="$MODEL_STORE/dreamsim"
export CRYSTALBLEU_SCITIKZER_CACHE_DIR="$MODEL_STORE/crystalbleu_scitikzer"

export OLLAMA_CLIENT_URL="http://127.0.0.1:8004/generate"
export IMAGE_MANIFEST
export IMAGES_DIR
export REFERENCES_DIR
export CAPTIONS_DIR
export GENERATED_IMAGE_DIR

cleanup() {
  echo "Stopping services..."

  kill "${PROMPTFOO_VIEW_PID:-}" 2>/dev/null || true
  kill "${CLIENT_PID:-}" 2>/dev/null || true
  kill "${SERVER_PID:-}" 2>/dev/null || true

  echo "Copying Promptfoo DB to $PROMPTFOO_PERSIST_DIR"
  mkdir -p "$PROMPTFOO_PERSIST_DIR"
  cp -a "$PROMPTFOO_RUN_DIR"/. "$PROMPTFOO_PERSIST_DIR"/ 2>/dev/null || true
  rm -rf "$PROMPTFOO_RUN_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "Job started on $(hostname)"
date
nvidia-smi

enroot start \
  --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --mount "$OLLAMA_MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$LOG_DIR/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

echo "Waiting for Ollama server..."
for _ in {1..60}; do
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

source "$CONDA_SH"

cd "$CLIENT_DIR"

OLLAMA_URL="http://127.0.0.1:11434/api/chat" \
OLLAMA_MODEL="$MODEL" \
conda run -n "$CLIENT_ENV" \
  uvicorn app:app --host 127.0.0.1 --port 8004 \
  > "$LOG_DIR/ollama-client-${SLURM_JOB_ID}.log" 2>&1 &

CLIENT_PID=$!

echo "Waiting for Ollama client..."
for _ in {1..60}; do
  if curl -s http://127.0.0.1:8004/docs >/dev/null; then
    echo "Client is ready."
    break
  fi
  sleep 2
done

cd "$PROJECT"

echo "Checking Promptfoo..."
conda run -n "$PROMPTFOO_ENV" promptfoo --version

conda run -n "$PROMPTFOO_ENV" \
  promptfoo view \
    --port "$PROMPTFOO_PORT" \
    --no \
  > "$LOG_DIR/promptfoo-view-${SLURM_JOB_ID}.log" 2>&1 &

PROMPTFOO_VIEW_PID=$!

conda run -n "$PROMPTFOO_ENV" \
  promptfoo eval \
    -c "$PROMPTFOO_CONFIG" \
    -j 1 \
    --watch \
  > "$LOG_DIR/promptfoo-eval-${SLURM_JOB_ID}.log" 2>&1

EVAL_EXIT=$?

echo "Promptfoo eval finished with exit code $EVAL_EXIT"
echo "Job finished."
date

exit "$EVAL_EXIT"
