#!/bin/bash
#SBATCH --job-name=tikz_demo_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint=GPU_MODEL:rtx_6000
#SBATCH --time=0-02:00:00          # 2 hours for interactive demo
#SBATCH --partition=PRACT
#SBATCH --mail-type=NONE
#SBATCH --output=logs/demo-%j.out
#SBATCH --error=logs/demo-%j.err

# =============================================
# Basic environment settings
# =============================================
I9_USER="${I9_USER:-s0042}"
PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"   # Your project root

CONTAINER_DIR="$PROJECT/demo"
DEMO="${DEMO:-demo_wo_descrip_gen}"                              # demo folder name
DEMO_DIR="$CONTAINER_DIR/$DEMO"
LOG_DIR="$DEMO_DIR/logs"
OLLAMA_MODEL_STORE="$PRAKT_DIR/projects/models/ollama_model_store"
MODEL_STORE="$PRAKT_DIR/projects/models"
IMAGE="$CONTAINER_DIR/ollama-server.sqsh"  # Ensure this exists

CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"
CLIENT_ENV="${CLIENT_ENV:-demo}"                         # Matches environment.yml
MODEL="${MODEL:-gemma4-tikz-rl-checkpoint-150:latest}"

# Ensure LaTeX binary path is included; config.py will append it to PATH
export PATH="/usr/prakt/$I9_USER/projects/full_latex/texlive/2026/bin/x86_64-linux:$PATH"

# Find pdftoppm location and add to PATH
PDFTOPPM_PATH=$(which pdftoppm 2>/dev/null || true)
if [ -n "$PDFTOPPM_PATH" ]; then
    PDFTOPPM_DIR=$(dirname "$PDFTOPPM_PATH")
    export PATH="$PDFTOPPM_DIR:$PATH"
else
    echo "WARNING: pdftoppm not found in PATH. Please install poppler-utils."
fi

export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$OLLAMA_MODEL_STORE"

echo "Running demo $DEMO"
echo "Job started on $(hostname)"
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

# =============================================
# 1. Start Ollama server (background)
# =============================================
echo "Starting Ollama server via enroot..."
enroot start --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --env OLLAMA_NUM_PARALLEL=4 \
  --env OLLAMA_DEBUG=1 \
  --env OLLAMA_CONTEXT_LENGTH=4096 \
  --mount "$OLLAMA_MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$LOG_DIR/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

OLLAMA_PID=$!

# Wait for Ollama to become ready
echo "Waiting for Ollama server..."
OLLAMA_READY=0
for _ in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Ollama is ready."
    OLLAMA_READY=1
    break
  fi
  sleep 2
done

if [ "$OLLAMA_READY" -ne 1 ]; then
  echo "❌ Error: Ollama failed to respond. Aborting."
  exit 1
fi

# model storage check
echo "======================================================"
echo "Debug: Models currently visible to Ollama (via /api/tags):"
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:11434/api/tags
echo ""
echo "======================================================"

# Check if the model exists; pull if missing (model name from config.py)
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"$MODEL\""; then
  echo "Model $MODEL not found. Pulling..."
  response=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\", \"stream\": false}" \
    http://127.0.0.1:11434/api/pull)
  if [ "$response" -ne 200 ]; then
      echo "❌ Pull failed with HTTP $response"
      # help the user debug by listing the model store content
      echo "Checking host storage directory content:"
      ls -la "$OLLAMA_MODEL_STORE"
      ls -la "$OLLAMA_MODEL_STORE/manifests/" 2>/dev/null || echo "manifests dir missing!"
      exit 1
  fi
  echo "Model pulled successfully."
fi

echo "=== GPU memory before loading model ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

# =============================================
# 2. Start FastAPI service (foreground)
# =============================================
echo "Activating Conda and starting Uvicorn API server..."
source "$CONDA_SH"
cd "$PROJECT" || exit 1

# Get node IP for tunnel instructions
NODE_IP=$(hostname -I | awk '{print $1}')
PORT=8444
PORT_LOCAL=8445
# Get the login node from which the job was submitted
LOGIN_NODE="${SLURM_SUBMIT_HOST:-$(hostname -s)}"

echo "======================================================"
echo "✅ Demo API is starting!"
echo "🔗 Access URL inside cluster: http://${NODE_IP}:${PORT}"
echo "======================================================"
echo "To access from your local browser, run this SSH tunnel:"
echo "   ssh -N -L ${PORT_LOCAL}:${NODE_IP}:${PORT} ${USER}@${LOGIN_NODE}"
echo "Then open http://localhost:${PORT_LOCAL} in your browser."
echo "======================================================"

cd "$DEMO_DIR"
export PYTHONPATH="$DEMO_DIR/backend:$PYTHONPATH"

# echo "Warming up model (waiting until loaded)..."
# curl -X POST http://127.0.0.1:11434/api/generate \
#   -H "Content-Type: application/json" \
#   -d "{\"model\":\"$MODEL\", \"prompt\":\"Hello\", \"stream\":false}" \
#   > "$LOG_DIR/warmup_response.json" 2>&1
# echo "Model is ready for inference."

# Run uvicorn via conda
echo "Starting Uvicorn server..."
conda run --no-capture-output -n "$CLIENT_ENV" \
  exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT

# =============================================
# 3. Cleanup (when uvicorn exits)
# =============================================
echo "Uvicorn server stopped. Shutting down Ollama..."
kill -9 $OLLAMA_PID 2>/dev/null || true

date
echo "Job finished."