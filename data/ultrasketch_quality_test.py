# Is UltraSketch good enough to use as our synthetic sketch pipeline,
# or do we need to find/fine-tune something better?

from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
import torch
import os
import sys

def resize_to_multiple(image, multiple=16):
    """Resize image so both dimensions are divisible by multiple."""
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    return image.resize((new_w, new_h), Image.LANCZOS)

# Load SketchFig — we want the rendered figures as input
ds = load_dataset("nllg/sketchfig", split="train")

# Output folder
output_dir = "/usr/prakt/s0031/ultrasketch_outputs"
os.makedirs(output_dir, exist_ok=True)

print("Loading UltraSketch pipeline...")
sys.stdout.flush()

# Load UltraSketch pipeline — using device_map as per HuggingFace usage
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
    figure = ds[i]['image']       # rendered TikZ figure
    real_sketch = ds[i]['sketch'] # real hand-drawn sketch (for comparison)

    # Resize to dimensions divisible by 16
    figure_resized = resize_to_multiple(figure, multiple=16)

    print(f"Example {i}: original size {figure.size} -> resized to {figure_resized.size}")
    sys.stdout.flush()

    # Generate synthetic sketch from rendered figure
    synthetic_sketch = pipe(
        prompt="Turn it into a hand-drawn sketch",
        image=figure_resized,
        mask_img=Image.new("RGB", figure_resized.size, "white"),
        num_inference_steps=50,
        image_guidance_scale=1.7,
        guidance_scale=1.5,
        strength=0.9
    ).images[0]

    # Save all three for visual comparison
    figure_resized.save(f"{output_dir}/{i}_1_rendered_figure.png")
    real_sketch.save(f"{output_dir}/{i}_2_real_sketch.png")
    synthetic_sketch.save(f"{output_dir}/{i}_3_synthetic_sketch.png")

    print(f"Example {i} done")
    sys.stdout.flush()

print("All done — check ultrasketch_outputs/")
sys.stdout.flush()# Is UltraSketch good enough to use as our synthetic sketch pipeline,
