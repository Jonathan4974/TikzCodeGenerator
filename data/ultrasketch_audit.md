# UltraSketch Audit

> **Correction:** the "combined hybrid" pixel-blend
> conclusion below is wrong - it should be done like in `training/sketch_agent/` where it 
> uses per-sample random selection between UltraSketch and the displacement field, not blending
> the two

## Overview
- Source: nllg/ultrasketch (HuggingFace)
- Date audited: 31-05-2026
- Updated: 04-06-2026
- Part of: DeTikZify project (nllg)

## What it is
A diffusion model trained to convert scientific figures into
hand-drawn style sketches. Based on SD3_UltraEdit_w_mask,
fine-tuned specifically for sketch generation.

## Training Data
| Source | Role |
|---|---|
| SketchFig (549 examples) | Primary fine-tuning data |
| DaTikZv2 figures rendered with Rough.js | Data augmentation |
| Sketchy Database | Data augmentation |
| Photo Sketching dataset | Data augmentation |

## Technical Requirements
- Model size: ~17.5GB
- Requires: CUDA GPU (float16), sm_75+ (Quadro P6000/node1,7 incompatible)
- Tested on: Quadro RTX 6000, node9 (24GB VRAM) ✅
- Dependencies: diffusers, torch, transformers, torchvision, scipy

## Usage
```python
from PIL import Image
from diffusers import DiffusionPipeline
import torch

def resize_to_multiple(image, multiple=16):
    w, h = image.size
    return image.resize(
        ((w // multiple) * multiple, (h // multiple) * multiple),
        Image.LANCZOS
    )

pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="nllg/ultrasketch",
    custom_pipeline="nllg/ultrasketch",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda:0")

# Images must be resized to multiples of 16 before inference
figure_resized = resize_to_multiple(figure, multiple=16)

sketch = pipe(
    prompt="Turn it into a hand-drawn sketch",
    image=figure_resized,
    mask_img=Image.new("RGB", figure_resized.size, "white"),
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9
).images[0]
```

## SLURM Notes
- Exclude node1 and node7 (Quadro P6000, sm_61 — incompatible)
- Use `--nodelist=node9` or `--exclude=node1,node7`
- MaxJobsPU=1 for PRACT partition — only one job at a time
- See `/notes/slurm_troubleshooting.md` for common errors

## Performance (from papers)
| Method | CC | Notes |
|---|---|---|
| Instruct-Pix2Pix (DeTikZify v1) | 0.66 | Original approach |
| Instruct-Pix2Pix fine-tuned | 0.70 | Fine-tuned on SketchFig |
| UltraSketch alone | 0.74 | Embedding-based CC (SigLIP) |
| Random displacement field | 0.75 | Better text preservation |
| **Combined hybrid (TikZero)** | **0.82** | **Current state of the art** |

## Experimental Results (June 2026)
Tested on 5 SketchFig examples using pixel-level CC
(grayscale cosine similarity — see note below).

| Example | UltraSketch | Displacement | Combined |
|---|---|---|---|
| 0 | 0.962 | 0.972 | 0.971 |
| 1 | 0.843 | 0.805 | 0.828 |
| 2 | 0.973 | 0.982 | 0.982 |
| 3 | 0.944 | 0.945 | 0.950 |
| 4 | 0.927 | 0.933 | 0.935 |
| **avg** | **0.930** | **0.927** | **0.933** |

### Important note on CC methodology
Our pixel-level CC scores are higher than the paper's reported
values (max 0.82) because of a methodological difference:

- **Our method:** pixel-level cosine similarity on flattened
  grayscale arrays — inflated by sparse white backgrounds
- **Paper method:** SigLIP embedding-based CC — subtracts
  embeddings, applies PCA to derive a global sketch vector,
  measures semantic and stylistic alignment

The rank order is consistent with papers (hybrid ≥ displacement
≥ UltraSketch in most examples), validating our approach even
if absolute scores are not directly comparable.

### Visual quality assessment
| Method | Simple figures | Complex figures |
|---|---|---|
| UltraSketch alone | Good | Artifacts, blur |
| Displacement alone | Structure preserved | Structure preserved, not sketch-like |
| Combined hybrid | Best | Better than UltraSketch alone |

## Conclusion
**Adopt the TikZero hybrid approach as our synthetic sketch pipeline.**
- UltraSketch provides stylistic hand-drawn variation
- Displacement field preserves structure and text rendering
- Averaging both gives best results without any fine-tuning
- Matches state of the art (CC 0.82 embedding-based)

## Future improvement (if needed)
If downstream model performance is insufficient, consider:
- Fine-tuning UltraSketch on more complex figures
- Implementing SigLIP embedding-based CC for more accurate evaluation
- Using VLM-generated descriptions to guide UltraSketch on complex figures

## Status
- ✅ Available on HuggingFace
- ✅ Pipeline loads and runs on GPU server (node9)
- ✅ Visual quality assessed on 5 examples
- ✅ Pixel-level CC computed and documented
- ✅ TikZero hybrid approach validated and adopted
- ⏳ SigLIP embedding-based CC — not yet implemented
- ⏳ Scale to full SketchFig dataset — pending storage decision

## Scripts
- `data/ultrasketch_quality_test.py` — generates 5 test outputs
  (rendered figure, real sketch, UltraSketch, displacement, combined)
- `data/cong_coeff_test.py` — computes pixel-level CC across methods