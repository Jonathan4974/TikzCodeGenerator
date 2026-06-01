# Is UltraSketch good enough to use as our synthetic sketch pipeline,
# or do we need to find/fine-tune something better?

from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
import torch
import os

# Load SketchFig — we want the rendered figures as input
ds = load_dataset("nllg/sketchfig", split="train")

# Output folder
os.makedirs("ultrasketch_outputs", exist_ok=True)

# Load UltraSketch pipeline
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="nllg/ultrasketch",
    custom_pipeline="nllg/ultrasketch",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="balanced"
)

# Test on first 5 examples
for i in range(5):
    figure = ds[i]['image']      # rendered TikZ figure
    real_sketch = ds[i]['sketch'] # real hand-drawn sketch (for comparison)

    # Generate synthetic sketch from rendered figure
    synthetic_sketch = pipe(
        prompt="Turn it into a hand-drawn sketch",
        image=figure,
        mask_img=Image.new("RGB", figure.size, "white"),
        num_inference_steps=50,
        image_guidance_scale=1.7,
        guidance_scale=1.5,
        strength=0.9
    ).images[0]

    # Save all three side by side for visual comparison
    figure.save(f"ultrasketch_outputs/{i}_1_rendered_figure.png")
    real_sketch.save(f"ultrasketch_outputs/{i}_2_real_sketch.png")
    synthetic_sketch.save(f"ultrasketch_outputs/{i}_3_synthetic_sketch.png")

    print(f"Example {i} done")

print("All done — check ultrasketch_outputs/")