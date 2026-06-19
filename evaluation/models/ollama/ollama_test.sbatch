#!/bin/bash
#SBATCH --job-name=ollama-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=0-02:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=NONE
#SBATCH --output=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/ollama-%j.out

set -euo pipefail

echo "Job started on $(hostname)"
date
nvidia-smi

# Paths
PROJECT=/usr/prakt/s0030/projects/tikzcodegenerator
CONTAINER_DIR=$PROJECT/evaluation/models/ollama
MODEL_STORE=/usr/prakt/s0030/projects/models/ollama_model_store
IMAGE=$CONTAINER_DIR/ollama+ollama+latest.sqsh

mkdir -p "$MODEL_STORE"
mkdir -p "$PROJECT/evaluation/logs"

# Start Ollama server in the background
enroot start \
  --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --mount "$MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$PROJECT/evaluation/logs/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

echo "Started Ollama server with PID $SERVER_PID"

# Stop server when the job exits
cleanup() {
  echo "Stopping Ollama server..."
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until Ollama is reachable
echo "Waiting for Ollama server..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Ollama is ready."
    break
  fi
  sleep 2
done

# Pull model if necessary
curl http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e2b-it-qat"
  }'

# Test generation
curl http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e2b-it-qat",
    "prompt": "Erkläre kurz, was ein AIG ist.",
    "stream": false
  }'

echo
echo "Job finished."
date