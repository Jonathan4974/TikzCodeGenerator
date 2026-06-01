# Evaluation

Shared metric implementations used by both benchmarking and training.
Think of this as the project's metrics library.

## What goes here
- Metric implementation scripts
- Evaluation utilities shared across milestones

## Metrics implemented
| Metric | Type | Script / location |
|---|---|---|
| CrystalBLEU (cBLEU) | Code similarity | `code_metrics.py` |
| Token Edit Distance (TED) | Code structure | `code_metrics.py` |
| DreamSim (DSim) | Perceptual image similarity | `image_metrics.py` |
| KID | Visual quality distribution | `image_metrics.py` |
| FID | Visual quality distribution | `image_metrics.py` |
| Image Structural Similarity (SSIM) | Perceptual image similarity | `promptfoo/assertions/image_structural_similarity.py` |
| LPIPS distance | Perceptual image similarity | `promptfoo/assertions/image_lpips.py` |
| DISTS distance | Perceptual image similarity | `promptfoo/assertions/image_dists.py` |
| CLIP similarity | Semantic similarity | `promptfoo/assertions/image_clip_similarity.py` |
| SigLIP similarity | Semantic similarity | `promptfoo/assertions/image_siglip_similarity.py` |
| Renderability | Code executability / render check | `promptfoo/assertions/tikz_is_renderable.py` |
| Compilation Rate | Code executability | `compiler_metrics.py` |
| Congruence Coefficient (CC) | Sketch structural similarity | `sketch_metrics.py` |

## Usage
All metrics are designed to be imported and called from benchmarking and training scripts:

```python
from evaluation.image_metrics import dreamsim, kid
from evaluation.code_metrics import cbleu, ted
from evaluation.compiler_metrics import compilation_rate
```

The promptfoo benchmark configuration at `evaluation/promptfoo/promptfooconfig.yaml` also uses additional evaluation assertions for SSIM, renderability, CLIP similarity, SigLIP similarity, LPIPS, DISTS, CrystalBLEU, and TED.
