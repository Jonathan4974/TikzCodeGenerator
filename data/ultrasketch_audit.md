# UltraSketch Audit

## Overview
- Source: nllg/ultrasketch (HuggingFace)
- Date audited: 31-05-2026
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
- Requires: CUDA GPU (float16)
- Cannot run on CPU locally — must run on GPU server (TODO)
- Dependencies: diffusers, torch, transformers, torchvision

## Usage
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="nllg/ultrasketch",
    custom_pipeline="nllg/ultrasketch",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="balanced"
)

sketch = pipe(
    prompt="Turn it into a hand-drawn sketch",
    image=figure,
    mask_img=Image.new("RGB", figure.size, "white"),
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9
).images[0]
```

## Performance (from DeTikZifyv2 paper)
- Congruence Coefficient (CC): 0.74
- When combined with image transformation method (CC 0.75): 
  averaged CC increases to 0.82

## Status
- ✅ Confirmed available on HuggingFace
- ✅ Pipeline loads successfully locally
- ❌ Full inference blocked — requires GPU server
- ⏳ Output quality testing pending GPU server

## Next Steps
- Run ultrasketch_test.py on GPU server 
- Visually compare synthetic vs real SketchFig sketches
- Compute CC and CLIP similarity scores on outputs
- Decide if quality is sufficient or if fine-tuning needed