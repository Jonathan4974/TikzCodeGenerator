# Sketch Similarity Metrics — Research Note

## Context
A key open question flagged in our initial presentation (May 6, 2026)
and by our supervisor: existing metrics don't capture sketch quality well.
Specifically — how do you measure how "sketch-like" a synthetic image is,
and how similar it is to a real hand-drawn sketch?

This note proposes two candidate metrics.

## The Problem
Standard image similarity metrics are not well suited for sketch evaluation:

| Metric | Problem for sketches |
|---|---|
| SSIM | Pixel-level — penalizes style differences even if structure is preserved |
| FID/KID | Distribution-level — needs many samples, not per-image |
| cBLEU | Text-based — not applicable to images |
| PSNR | Pixel-level — same issue as SSIM |

Sketches and rendered figures are visually very different (line drawings vs
filled vector graphics) but structurally similar. We need a metric that
captures **structural/semantic similarity** rather than pixel similarity.

## Candidate 1 — Congruence Coefficient (CC)
**What it measures:** Structural similarity between two images, 
focusing on layout and shape correspondence rather than pixel values.

**Why relevant:**
- Already used by DeTikZifyv2 to evaluate UltraSketch quality
- They report CC = 0.74 for UltraSketch alone, 0.82 when combined
  with image transformation — gives us a benchmark to compare against
- Directly comparable to prior work

**How to compute:**
```python
import numpy as np
from scipy.signal import correlate2d
from PIL import Image

def congruence_coefficient(img1, img2):
    # Convert to grayscale numpy arrays
    a = np.array(img1.convert("L")).astype(float)
    b = np.array(img2.convert("L")).astype(float)
    
    # Flatten and compute CC
    a = a.flatten()
    b = b.flatten()
    
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    
    return numerator / denominator if denominator != 0 else 0.0
```

**Limitations:**
- Sensitive to resolution differences — images must be resized to same dimensions
- Does not capture semantic content, only structural overlap

---

## Candidate 2 — CLIP Similarity
**What it measures:** Semantic similarity between two images using
CLIP's joint vision-language embedding space.

**Why relevant:**
- Captures whether sketch and figure convey the same concept,
  even if they look visually different
- More robust to style differences than pixel metrics
- Widely used in generative model evaluation

**How to compute:**
```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_similarity(img1, img2):
    inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    # Normalize and compute cosine similarity
    features = features / features.norm(dim=-1, keepdim=True)
    similarity = (features[0] * features[1]).sum().item()
    
    return similarity
```

**Limitations:**
- CLIP was not trained on sketches specifically — may underestimate
  similarity between sketch and rendered figure
- Semantic similarity doesn't capture structural/spatial accuracy

---

## Proposed Approach
Use **both metrics together** — they capture complementary aspects:

| Metric | Captures | Misses |
|---|---|---|
| CC | Structural layout similarity | Semantic content |
| CLIP similarity | Semantic content match | Fine-grained structure |

A good synthetic sketch should score high on both:
- High CC → structure is preserved
- High CLIP similarity → semantic content is preserved

## Baseline to Beat
From DeTikZifyv2: UltraSketch achieves CC = 0.74 on SketchFig.
Our synthetic pipeline should target at least this threshold.

## Status
- ⏳ Pending supervisor/teammates opinions at next meeting
- ⏳ Implementation ready but not yet run (blocked on GPU for UltraSketch)

## Questions for Supervisor (IGNORE THIS)
- Are CC and CLIP similarity the right metrics, or is there a 
  better established metric for sketch evaluation we're missing?
- Should we also include a human evaluation component for a 
  small sample?
- Is 0.74 CC a reasonable threshold to target for our 
  synthetic sketches?
