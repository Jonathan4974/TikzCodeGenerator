#!/bin/bash
#SBATCH --job-name=geotikz_augmentation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint=GPU_MODEL:rtx_6000
#SBATCH --time=0-00:15:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=NONE
#SBATCH --output=logs/ollama-%j.out
#SBATCH --error=logs/ollama-%j.err          


I9_USER="${I9_USER:-s0042}"

PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"

LOG_DIR="$PROJECT/data/description_augment/logs"
MODEL="${MODEL:-qwen3.5:9b}"

CONTAINER_DIR="$PROJECT/evaluation/models/ollama"
OLLAMA_MODEL_STORE="$PRAKT_DIR/projects/models/ollama_model_store"
MODEL_STORE="$PRAKT_DIR/projects/models"
IMAGE="$CONTAINER_DIR/ollama+ollama+latest.sqsh"

GEOAUG_DIR="$PROJECT/data/description_augment"
CLIENT_ENV="ollama-client"

CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

mkdir -p \
  "$LOG_DIR" \
  "$OLLAMA_MODEL_STORE" \
  "$MODEL_STORE/huggingface" \
  "$MODEL_STORE/torch" \

export PYTHONUNBUFFERED=1


echo "Job started on $(hostname)"
date
nvidia-smi

enroot start --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --env OLLAMA_NUM_PARALLEL=4 \
  --mount "$OLLAMA_MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$LOG_DIR/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

OLLAMA_PID=$!

echo "Waiting for Ollama server..."
for _ in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Ollama is ready."
    OLLAMA_READY=1
    break
  fi
  sleep 2
done

if [ "$OLLAMA_READY" -ne 1 ]; then
  echo "❌ Error: Ollama failed to respond within 2 minutes. Aborting."
  exit 1
fi

curl -s http://127.0.0.1:11434/api/tags >/dev/null

if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"$MODEL\""; then
  echo "Model $MODEL not found locally. Initiating background pulling..."
  curl -s http://127.0.0.1:11434/api/pull \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\", \"stream\": false}" > /dev/null
  echo "Model $MODEL pulled successfully."
fi

echo "Activating Conda and starting python data pipeline..."
source "$CONDA_SH"
cd "$GEOAUG_DIR"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run --no-capture-output -n "$CLIENT_ENV" \
    exec python generate_description_test.py \
    > "$LOG_DIR/gen-description-${SLURM_JOB_ID}.log" 2>&1

GEOAUG_EXIT=$?

echo "Geotikz data augmentation finished with exit code $GEOAUG_EXIT"

echo "Pipeline finished. Shutting down background Ollama engine..."
kill -9 $OLLAMA_PID

date
echo "Slurm job sequence finalized"