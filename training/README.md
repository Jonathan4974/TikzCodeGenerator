# Training

Fine-tuning scripts, job configs, and experiment setup for 
improving on baseline models toward our three project goals.

## What goes here
- Fine-tuning scripts (SFT, RL)
- sbatch job scripts for GPU server
- LoRA/PEFT configuration files
- wandb experiment configs

## What does NOT go here
- Model weights — stored on GPU server slurm storage
- Benchmarking of baseline models — that lives in `/benchmarking`

## Project Goals
| Goal | Approach | Status |
|---|---|---|
| (b) Concise, readable code | Fine-tune with length/conciseness objective | ⏳ Not started |
| (c) Compiler losses + synthetic data | RL with fine-grained compiler feedback | ⏳ Not started |

## GPU Server
- Checkpoints: `/storage/slurm/<user>/checkpoints/` — delete after done
- Logs: `/usr/prakt/<user>/logs/`
- Experiment tracking: wandb project `sketch-to-tikz`