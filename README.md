# TikzCodeGenerator

TikzCodeGenerator is a sketch-to-TikZ research repository for model benchmarking, evaluation, and fine-tuning.

## Main folders
- `benchmarking` — baseline model inference, promptfoo configs, and benchmark results.
- `evaluation` — shared evaluation metric implementations and promptfoo assertions.
- `training` — fine-tuning, RL experiment setup, and model training utilities.
- `demo` — demo scripts and examples.
- `data` — datasets, references, and related assets.

## Evaluation metrics
This project uses shared metrics from `/evaluation`
Key metrics include:
- CrystalBLEU (cBLEU)
- Token Edit Distance (TED)
- DreamSim (DSim)
- FID and KID
- Image Structural Similarity (SSIM)
- LPIPS distance
- DISTS distance
- CLIP similarity
- SigLIP similarity
- Renderability checks

TODO Complete

## Notes
For full details, see each folder's `README.md` (eg. `evaluation/README.md`).
