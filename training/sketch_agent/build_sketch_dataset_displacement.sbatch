#!/bin/bash
# Displacement-only split - CPU-only, no GPU requested on purpose (this method never
# touches the GPU), so it can schedule independently of, and much faster than, the
# ultrasketch job when the GPU nodes/devcube are busy.
#
# First-pass (100 rows): sbatch build_sketch_dataset_displacement.sbatch
# Full dataset: edit/remove the `--max-rows 100` line below once the first pass looks good
# (extra args after `sbatch <script>` are appended after the hardcoded ones and win, since
# argparse keeps the last value of a repeated flag - e.g. `sbatch ... --max-rows 500000`
# also works without editing the script, if you'd rather not touch it).
#SBATCH --job-name="sketch_dataset_displacement"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --partition=NORMAL
#SBATCH --account=stud
#SBATCH --qos=stud
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0031/projects/tikzcodegenerator/training/sketch_agent/logs/build_sketch_dataset_displacement-%j.out

set -euo pipefail

echo "Job started on $(hostname)"
pwd
date

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
  --method displacement \
  --output-dir training/sketch_agent/output_final/sketch_dataset_displacement \
  --max-rows 100 \
  --sbatch-script "$0" \
  "$@"

echo "Job finished."
date
