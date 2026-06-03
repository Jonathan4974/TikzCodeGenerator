# Is UltraSketch good enough to use as our synthetic sketch pipeline,
# or do we need to find/fine-tune something better?
# Tests three approaches: UltraSketch, random displacement field, and combined (TikZero approach)

from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
import torch
import os
import sys


def resize_to_multiple(image, multiple=16):
    """Resize image so both dimensions are divisible by multiple."""
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    return image.resize((new_w, new_h), Image.LANCZOS)


def random_displacement_field(image, alpha=20, sigma=5, seed=None):
    """
    Simulates hand-drawn style via elastic deformation.
    alpha: intensity of displacement
    sigma: smoothness of displacement field
    """
    if seed is not None:
        np.random.seed(seed)

    img_array = np.array(image).astype(np.float32)
    shape = img_array.shape[:2]

    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Create meshgrid and apply displacement
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (
        np.clip(y + dy, 0, shape[0] - 1).ravel(),
        np.clip(x + dx, 0, shape[1] - 1).ravel()
    )

    # Apply to each channel
    if len(img_array.shape) == 3:
        distorted = np.stack([
            map_coordinates(img_array[:, :, c], indices, order=1).reshape(shape)
            for c in range(img_array.shape[2])
        ], axis=2).astype(np.uint8)
    else:
        distorted = map_coordinates(
            img_array, indices, order=1
        ).reshape(shape).astype(np.uint8)

    return Image.fromarray(distorted)


def generate_synthetic_sketch(figure, pipe, alpha=20, sigma=5):
    """
    Replicates TikZero hybrid approach:
    - UltraSketch output (CC 0.74)
    - Random displacement field (CC 0.75)
    - Combined average of both -> CC 0.82
    """
    # Path 1: UltraSketch
    figure_resized = resize_to_multiple(figure, multiple=16)
    ultrasketch_output = pipe(
        prompt="Turn it into a hand-drawn sketch",
        image=figure_resized,
        mask_img=Image.new("RGB", figure_resized.size, "white"),
        num_inference_steps=50,
        image_guidance_scale=1.7,
        guidance_scale=1.5,
        strength=0.9
    ).images[0]

    # Path 2: Random displacement field
    displacement_output = random_displacement_field(
        figure_resized, alpha=alpha, sigma=sigma
    )
    displacement_output = displacement_output.resize(ultrasketch_output.size)

    # Combined: average both (TikZero approach)
    combined = Image.fromarray(
        (
            np.array(ultrasketch_output).astype(np.float32) * 0.5 +
            np.array(displacement_output).astype(np.float32) * 0.5
        ).astype(np.uint8)
    )

    return figure_resized, ultrasketch_output, displacement_output, combined


# ── Main ──────────────────────────────────────────────────────────────────────

# Load SketchFig
print("Loading SketchFig dataset...")
sys.stdout.flush()
ds = load_dataset("nllg/sketchfig", split="train")

# Output folder
output_dir = "/usr/prakt/s0031/ultrasketch_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load UltraSketch pipeline
print("Loading UltraSketch pipeline...")
sys.stdout.flush()
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="nllg/ultrasketch",
    custom_pipeline="nllg/ultrasketch",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda:0")
print("Pipeline loaded. Starting inference...")
sys.stdout.flush()

# Test on first 5 examples
for i in range(5):
    figure = ds[i]['image']        # rendered TikZ figure
    real_sketch = ds[i]['sketch']  # real hand-drawn sketch (ground truth)

    print(f"Example {i}: original size {figure.size}")
    sys.stdout.flush()

    figure_resized, ultrasketch_out, displacement_out, combined_out = \
        generate_synthetic_sketch(figure, pipe)

    # Save all outputs for visual comparison
    figure_resized.save(f"{output_dir}/{i}_1_rendered_figure.png")
    real_sketch.save(f"{output_dir}/{i}_2_real_sketch.png")
    ultrasketch_out.save(f"{output_dir}/{i}_3_ultrasketch.png")
    displacement_out.save(f"{output_dir}/{i}_4_displacement.png")
    combined_out.save(f"{output_dir}/{i}_5_combined.png")

    print(f"Example {i} done — saved 5 images")
    sys.stdout.flush()

print("All done — check ultrasketch_outputs/")
sys.stdout.flush()