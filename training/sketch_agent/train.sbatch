#!/bin/bash
#SBATCH --job-name="ultrasketch_test"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --partition=PRACT
#SBATCH --qos=practical_course
#SBATCH --exclude=node1,node7
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0031/projects/tikzcodegenerator/training/sketch_agent/logs/slurm-%j.out

export HF_HOME=/usr/prakt/s0031/projects/models/huggingface

cd /usr/prakt/s0031/projects/tikzcodegenerator

python3 -m training.sketch_agent.train \
  --output-dir training/sketch_agent/output \
  --checkpoint-dir training/sketch_agent/output/checkpoints \
  --synthetic-dir training/sketch_agent/output/synthetic_pairs \
  --max-steps 8 \
  --checkpoint-interval-steps 2 \
  --synthetic-samples 8 \
  --image-size 256 \
  --seed 3407 \
  --self-resubmit
