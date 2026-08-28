#!/bin/bash
#SBATCH --job-name=tikz_demo_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint=GPU_MODEL:rtx_6000
#SBATCH --time=0-00:15:00          # 2 hours for interactive demo
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
DEMO="demo_wo_descrip_gen"                              # demo folder name
DEMO_DIR="$CONTAINER_DIR/$DEMO"
LOG_DIR="$DEMO_DIR/logs"
OLLAMA_MODEL_STORE="$PRAKT_DIR/projects/models/ollama_model_store"
MODEL_STORE="$PRAKT_DIR/projects/models"
IMAGE="$CONTAINER_DIR/ollama-server.sqsh"  # Ensure this exists

CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"
CLIENT_ENV="demo"                         # Matches environment.yml
MODEL="${MODEL:-gemma4:31b-it-q4_K_M}"

# Ensure LaTeX binary path is included; config.py will append it to PATH
export PATH="/usr/prakt/$I9_USER/projects/full_latex/texlive/2026/bin/x86_64-linux:$PATH"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$OLLAMA_MODEL_STORE"

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

# Check if the model exists; pull if missing (model name from config.py)
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"$MODEL\""; then
  echo "Model $MODEL not found. Pulling..."
  curl -s http://127.0.0.1:11434/api/pull \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\", \"stream\": false}" > /dev/null
  echo "Model pulled."
fi

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
# Run uvicorn via conda
conda run --no-capture-output -n "$CLIENT_ENV" \
  exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT

# =============================================
# 3. Cleanup (when uvicorn exits)
# =============================================
echo "Uvicorn server stopped. Shutting down Ollama..."
kill -9 $OLLAMA_PID 2>/dev/null || true

date
echo "Job finished."