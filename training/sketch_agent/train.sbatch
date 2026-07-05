#!/bin/bash
#SBATCH --job-name="sketch_agent_train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --partition=PRACT
#SBATCH --qos=practical_course
#SBATCH --nodelist=node9
#SBATCH --exclude=node1,node7
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0031/projects/tikzcodegenerator/training/sketch_agent/logs/slurm-%j.out

set -euo pipefail

echo "Job started on $(hostname)"
pwd
date
nvidia-smi

I9_USER="${I9_USER:-s0031}"
PRAKT_DIR="/usr/prakt/$I9_USER"
PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"

ENV_NAME="sketch-agent"
CONDA_SH="$PRAKT_DIR/miniconda3/etc/profile.d/conda.sh"

MODEL_STORE="$PRAKT_DIR/projects/models"
export HF_HOME="$MODEL_STORE/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$MODEL_STORE/torch"
export DREAMSIM_CACHE_DIR="$MODEL_STORE/dreamsim"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

source "$CONDA_SH"

cd "$PROJECT"

conda run -n "$ENV_NAME" python -m training.sketch_agent.train

echo "Job finished."
date
