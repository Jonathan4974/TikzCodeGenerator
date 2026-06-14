# Testing different instruction prompts to find the best one for diagram→sketch conversion
#   ultrasketch_outputs/
#     baseline/          <- "Turn it into a hand-drawn sketch" (original)
#     v1_structure/      <- emphasize structure preservation
#     v2_scientific/     <- scientific sketch framing
#     v3_minimal/        <- minimal intervention
#     v4_anti_artifact/  <- explicit anti-artifact
#     v5_student/        <- student sketch framing
#     ground_truth/      <- real sketches + rendered figures (saved once, shared reference)

from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
import torch
import os
import sys

NUM_EXAMPLES = 5
BASE_OUTPUT_DIR = "/usr/prakt/s0031/ultrasketch_outputs" #change $USER

PROMPT_VARIANTS = {
    "baseline": "Turn it into a hand-drawn sketch",
    "v1_structure": "Convert this diagram into a hand-drawn pencil sketch, preserving all geometric shapes, lines, and text labels exactly as they appear",
    "v2_scientific": "Redraw this as a rough hand-drawn scientific sketch with pencil strokes, keeping all structural elements, annotations, and labels legible",
    "v3_minimal": "Lightly sketch this image by hand, maintaining the exact layout, geometry, and any text or labels",
    "v4_anti_artifact": "Convert to a hand-drawn pencil sketch. Preserve all text labels, geometric structure, and fine details. Avoid blurring, smearing, or distorting any elements",
    "v5_student": "Transform this TikZ figure into a hand-drawn sketch as if drawn by a student on paper, keeping all shapes, arrows, nodes, and labels intact and legible",
}

INFERENCE_PARAMS = dict(
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9,
)

def resize_to_multiple(image, multiple=16):
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    return image.resize((new_w, new_h), Image.LANCZOS)


def run_ultrasketch(pipe, image, prompt):
    return pipe(
        prompt=prompt,
        image=image,
        mask_img=Image.new("RGB", image.size, "white"),
        **INFERENCE_PARAMS,
    ).images[0]


def make_dirs(base, variants):
    os.makedirs(os.path.join(base, "ground_truth"), exist_ok=True)
    for name in variants:
        os.makedirs(os.path.join(base, name), exist_ok=True)

print("Loading SketchFig dataset...")
ds = load_dataset("nllg/sketchfig", split="train")

make_dirs(BASE_OUTPUT_DIR, PROMPT_VARIANTS.keys())

print("Loading UltraSketch pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="nllg/ultrasketch",
    custom_pipeline="nllg/ultrasketch",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

pipe.to("cuda:0")
print("Pipeline loaded. Starting inference...")

for i in range(NUM_EXAMPLES):
    figure = ds[i]['image']
    real_sketch = ds[i]['sketch']

    figure_resized = resize_to_multiple(figure, multiple=16)

    gt_dir = os.path.join(BASE_OUTPUT_DIR, "ground_truth")
    figure_resized.save(os.path.join(gt_dir, f"{i}_rendered.png"))
    real_sketch.save(os.path.join(gt_dir, f"{i}_real_sketch.png"))

    print(f"\nExample {i} (size {figure_resized.size}):")

    for variant_name, prompt in PROMPT_VARIANTS.items():
        print(f"  Running {variant_name}...")

        output = run_ultrasketch(pipe, figure_resized, prompt)
        out_path = os.path.join(BASE_OUTPUT_DIR, variant_name, f"{i}.png")
        output.save(out_path)

        print(f"  Saved -> {out_path}")

print("\nAll done.")