#!/bin/bash
#SBATCH --job-name=promptfoo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint=GPU_MODEL:rtx_6000
#SBATCH --time=0-08:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonas.schulte@tum.de
#SBATCH --output=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/promptfoo-%j.out
#SBATCH --error=/usr/prakt/s0030/projects/tikzcodegenerator/evaluation/logs/promptfoo-%j.err


set -euo pipefail

APPTAINER="/usr/bin/apptainer"
echo "Using Apptainer: $APPTAINER"
"$APPTAINER" --version

EVAL_DIR="/usr/prakt/s0030/projects/tikzcodegenerator/evaluation"
DETIKZIFY_DIR="$EVAL_DIR/models/detikzify_2_5_8b"
PROMPTFOO_DIR="$EVAL_DIR/promptfoo"

mkdir -p "$EVAL_DIR/logs"
mkdir -p "$PROMPTFOO_DIR/result/promptfoo-db"
mkdir -p "$PROMPTFOO_DIR/result/generated_images"

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

echo "Starting DeTikZify from:"
echo "$DETIKZIFY_DIR"

cd "$DETIKZIFY_DIR"

"$APPTAINER" run --nv \
  --bind "$PWD:/app" \
  --bind "/usr/prakt/s0030/projects/models:/models" \
  --env QUANTIZATION=none \
  --pwd /app \
  detikzify-2-5-8b.sif &

DETIKZIFY_PID=$!

cleanup() {
  echo "Cleaning up DeTikZify process..."
  kill "$DETIKZIFY_PID" 2>/dev/null || true
  wait "$DETIKZIFY_PID" 2>/dev/null || true
}

trap cleanup EXIT

echo "Waiting for DeTikZify to start..."
sleep 20

echo "Starting promptfoo from:"
echo "$PROMPTFOO_DIR"

cd "$PROMPTFOO_DIR"

"$APPTAINER" exec \
  --env-file .env \
  -B "$PWD:/app" \
  -B "$PWD/result/promptfoo-db:/root/.promptfoo" \
  -B "$PWD/result/generated_images:/generated_images" \
  -B "/usr/prakt/s0030/projects/data/benchmark_data/manifest_splits/image_manifest_1.csv:/image_manifest.csv:ro" \
  -B "/usr/prakt/s0030/projects/data/benchmark_data/benchmark_corpus:/crystalbleu_corpus:ro" \
  -B "/usr/prakt/s0030/projects/data/benchmark_data/images:/images:ro" \
  -B "/usr/prakt/s0030/projects/data/benchmark_data/benchmark_data/references:/references:ro" \
  -B "/usr/prakt/s0030/projects/models:/models" \
  --pwd /app \
  promptfoo-tex.sif \
  sh -lc 'promptfoo eval -c configs/image_to_tikz_promptfooconfig_detikzify_2_5_8b.yaml -j 1 --watch & sleep 5 && promptfoo view --port 15500 --no'

echo "Promptfoo finished."
