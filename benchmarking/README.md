# Benchmarking

This folder contains everything needed to evaluate and compare 
baseline models before any fine-tuning — our M2 milestone.

## What goes here
- promptfoo config files
- Model inference scripts
- Benchmark results and comparison tables
- sbatch scripts for running inference on GPU server

## What does NOT go here
- Metric implementations — those live in `/evaluation`
- Fine-tuning scripts — those live in `/training`

## Models being benchmarked

### Image/Sketch to TikZ
| Model | Owner |
|---|---|
| DeTikZify-v2-8b | Jonas |
| DeTikZify-v2.5-8b | Jonas |
| GeoTikZBridge | Francisco |
| AutomaTikZ | Francisco |
| TikZilla | Yangjie |
| TikZero | Jonas |

### General Purpose VLMs
| Model | Owner |
|---|---|
| ChatGPT-latest | via API |
| DeepSeek | Yangjie |
| Qwen3.6-27B | Yangjie |
| Gemma-4 | Yangjie |

## Metrics used
See `/evaluation` for implementations.
| Metric |
|---|
| DreamSim |
| CrystalBLEU |
| TED |
| FID |
| KID |
| Image Structural Similarity (SSIM) |
| LPIPS distance |
| DISTS distance |
| CLIP similarity |
| SigLIP similarity |
| Renderability |
| Output length vs reference |
| Time to generate |