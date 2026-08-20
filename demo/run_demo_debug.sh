#!/bin/bash
# =============================================
# Debug script for TikZ demo – run on workstation
# (no Slurm, runs in current shell)
# =============================================

# ------------------------------
# User settings (adjust as needed)
# ------------------------------
I9_USER="${I9_USER:-s0042}"
PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"   # your project root

# Change to a smaller model for 12GB GPU
MODEL="${MODEL:-gemma4:e2b-it-q4_K_M}"          # or phi4:latest, gemma4:9b, etc.

# Paths
CONTAINER_DIR="$PROJECT/demo"
OLLAMA_MODEL_STORE="$PRAKT_DIR/projects/models/ollama_model_store"
IMAGE="$CONTAINER_DIR/ollama-server.sqsh"          # ensure this exists

CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"
CLIENT_ENV="demo"                                  # your conda env name

# LaTeX
export PATH="/usr/prakt/s0042/projects/full_latex/texlive/2026/bin/x86_64-linux:$PATH"
export PYTHONUNBUFFERED=1

# Create a unique identifier for this run (use PID + timestamp)
RUN_ID="$$_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$CONTAINER_DIR/logs_debug"
mkdir -p "$LOG_DIR" "$OLLAMA_MODEL_STORE"

echo "======================================================"
echo "Debug run started at $(date)"
echo "Run ID: $RUN_ID"
echo "Working directory: $PROJECT"
echo "Using model: $MODEL"
echo "======================================================"
nvidia-smi 2>/dev/null || echo "No nvidia-smi (maybe no GPU?)"

# =============================================
# 1. Start Ollama server (background)
# =============================================
echo "Starting Ollama server via enroot..."
OLLAMA_LOG="$LOG_DIR/ollama-server-debug-${RUN_ID}.log"
enroot start --rw \
  --env OLLAMA_MODELS=/ollama-model-store \
  --env OLLAMA_NUM_PARALLEL=4 \
  --mount "$OLLAMA_MODEL_STORE:/ollama-model-store" \
  "$IMAGE" \
  serve > "$OLLAMA_LOG" 2>&1 &

OLLAMA_PID=$!
echo "Ollama server PID: $OLLAMA_PID"
echo "Ollama log: $OLLAMA_LOG"

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
  kill $OLLAMA_PID 2>/dev/null
  exit 1
fi

# Check if the model exists; pull if missing
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"$MODEL\""; then
  echo "Model $MODEL not found. Pulling (this may take a while)..."
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
conda activate "$CLIENT_ENV" || {
  echo "❌ Failed to activate conda environment '$CLIENT_ENV'"
  kill $OLLAMA_PID 2>/dev/null
  exit 1
}

cd "$PROJECT" || exit 1

# Get node IP (should be workstation's own IP)
NODE_IP=$(hostname -I | awk '{print $1}')
PORT=8444
PORT_LOCAL=8445

echo "======================================================"
echo "✅ Demo API is starting!"
echo "🔗 Access URL inside workstation: http://${NODE_IP}:${PORT}"
echo "======================================================"
echo "To access from your local browser, run this SSH tunnel:"
echo "   ssh -N -L ${PORT_LOCAL}:${NODE_IP}:${PORT} ${USER}@$(hostname -s)"
echo "Then open http://localhost:${PORT_LOCAL} in your browser."
echo "======================================================"
echo "Uvicorn log will appear below (Ctrl+C to stop)."

# Run uvicorn (foreground)
cd "$CONTAINER_DIR"
export PYTHONPATH="$CONTAINER_DIR/backend:$PYTHONPATH"
python -m uvicorn app:app --host 0.0.0.0 --port $PORT --log-level debug

# =============================================
# 3. Cleanup (when uvicorn exits)
# =============================================
echo "Uvicorn server stopped. Shutting down Ollama..."
kill -9 $OLLAMA_PID 2>/dev/null || true

date
echo "Debug run finished."