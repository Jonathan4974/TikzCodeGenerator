#!/bin/bash
#SBATCH --job-name=ollama-client-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=0-02:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-%j.out

set -euo pipefail

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
CONDA_ENV=ollama-client

TEST_IMAGE=/usr/prakt/s0030/projects/data/benchmark_data/images/row_00000000.png
TEST_PROMPT="Erkläre kurz, was in diesem Bild zu sehen ist."

mkdir -p "$MODEL_STORE" "$LOG_DIR"

# Important: keep the Ollama/Enroot startup before Conda setup.
# This matches the sbatch version that already worked.
enroot start \
  --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --mount "$MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$LOG_DIR/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

cleanup() {
  echo "Stopping services..."
  kill "${CLIENT_PID:-}" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null || true
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

# Conda only after Ollama is running
source /usr/prakt/s0030/miniconda3/etc/profile.d/conda.sh

conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV" || \
  conda create -y -n "$CONDA_ENV" python=3.11

conda run -n "$CONDA_ENV" pip install -q -r "$CLIENT_DIR/requirements.txt"

cd "$CLIENT_DIR"

OLLAMA_URL=http://127.0.0.1:11434/api/chat \
OLLAMA_MODEL="$MODEL" \
conda run -n "$CONDA_ENV" \
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
echo "Job finished."
date