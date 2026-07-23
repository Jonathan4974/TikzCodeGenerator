#!/bin/bash
# UltraSketch-only split: a real ~50-step SDXL-scale diffusion pass per row, needs a GPU.
# Independent job from the displacement split (its own --output-dir/checkpoint), and will
# very likely span multiple 8h windows + self-resubmits at any real row count.
#
#SBATCH --job-name="sketch_dataset_ultrasketch"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --partition=NORMAL
#SBATCH --account=stud
#SBATCH --qos=stud
#SBATCH --nodelist=node4,node5,node14,node15,node16,node17,node18,node19,node20,node21
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0031/projects/tikzcodegenerator/training/sketch_agent/logs/build_sketch_dataset_ultrasketch-%j.out

set -euo pipefail

echo "Job started on $(hostname)"
pwd
date
nvidia-smi

I9_USER="${I9_USER:-s0031}"
PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"
ENV_FILE="$PROJECT/training/sketch_agent/.env"

ENV_NAME="sketch-agent"
CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

MODEL_STORE="$PRAKT_DIR/projects/models"
export HF_HOME="$MODEL_STORE/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -f "$ENV_FILE" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
  export HF_TOKEN
fi

source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$PROJECT"

python -m training.sketch_agent.build_sketch_dataset \
  --method ultrasketch \
  --output-dir training/sketch_agent/output_final/sketch_dataset_ultrasketch \
  --max-rows 25000 \
  --sbatch-script "$PROJECT/training/sketch_agent/build_sketch_dataset_ultrasketch.sbatch" \
  "$@"

echo "Job finished."
date
