"""Clean-image -> synthetic-sketch generation.

`generate_synthetic_sketch` implements a three-way per-sample draw: keep the original
clean image, or substitute a sketch from one of two methods (50/50):
  - UltraSketch (`nllg/ultrasketch`): a fine-tuned img2img diffusion model.
  - A classical Gaussian-filtered random displacement warp.

`sketch_probability` sets P(substitute at all); `ultrasketch_probability` splits that
substituted mass between the two methods. `build_sketch_dataset.py` calls this with
`sketch_probability=1.0` (always substitute, never "original") since the clean/sketch
choice for training happens separately, later, in sketch_choice_dataset.py. Basically
we always are going to populate the sketch column of the datset we building with this
synthetic sketch. Then later when reading the choice is made 50/50 with original or this
sketch.  
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

MULTIPLE_OF = 16

ULTRASKETCH_MODEL = "nllg/ultrasketch"
ULTRASKETCH_PROMPT = "Turn it into a hand-drawn sketch"
ULTRASKETCH_INFERENCE_PARAMS = dict(
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9,
)


def assert_multiple_of(image: Image.Image, label: str, multiple: int = MULTIPLE_OF) -> None:
    width, height = image.size
    if width % multiple != 0 or height % multiple != 0:
        raise ValueError(
            f"{label} has size {image.size}, expected both dimensions to be "
            f"multiples of {multiple}"
        )


def resize_to_multiple(image: Image.Image, multiple: int = MULTIPLE_OF) -> Image.Image:
    """Rounds each dimension down to the nearest multiple; no-op if already aligned."""
    width, height = image.size
    new_width = max(multiple, (width // multiple) * multiple)
    new_height = max(multiple, (height // multiple) * multiple)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_like(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Resizes `image` back to `reference`'s size if they differ."""
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.LANCZOS)


def random_displacement_field(
    image: Image.Image,
    alpha: float = 6.0,
    sigma: float = 12.0,
    seed: int = 42,
) -> Image.Image:
    """Classical (non-learned) sketch-like warp: Gaussian-filtered random displacement."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    assert_multiple_of(image, "displacement input")
    rng = np.random.default_rng(seed)
    img_array = np.array(image.convert("RGB")).astype(np.float32)
    shape = img_array.shape[:2]

    dx = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (
        np.clip(y + dy, 0, shape[0] - 1).ravel(),
        np.clip(x + dx, 0, shape[1] - 1).ravel(),
    )

    distorted = np.stack(
        [
            map_coordinates(img_array[:, :, channel], indices, order=1).reshape(shape)
            for channel in range(img_array.shape[2])
        ],
        axis=2,
    )
    return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))


def load_ultrasketch_pipeline() -> Any:
    """Loads the UltraSketch img2img pipeline (~17.5GB)"""
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        ULTRASKETCH_MODEL,
        custom_pipeline=ULTRASKETCH_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    return pipe


def release_ultrasketch_pipeline(pipe: Any) -> None:
    """Explicit cleanup between batches. UltraSketch is a full SDXL-scale pipeline."""
    import torch

    del pipe
    torch.cuda.empty_cache()


def run_ultrasketch(pipe: Any, image: Image.Image, generator: Optional[Any] = None) -> Image.Image:
    original = image
    resized = resize_to_multiple(image).convert("RGB")
    assert_multiple_of(resized, "UltraSketch input")
    output = pipe(
        prompt=ULTRASKETCH_PROMPT,
        image=resized,
        mask_img=Image.new("RGB", resized.size, "white"),
        generator=generator,
        **ULTRASKETCH_INFERENCE_PARAMS,
    ).images[0]
    output = resize_like(output.convert("RGB"), resized)
    assert_multiple_of(output, "UltraSketch output")
    return resize_like(output, original)


def generate_synthetic_sketch(
    image: Image.Image,
    seed: Optional[int] = None,
    sketch_probability: float = 0.5,
    ultrasketch_probability: float = 0.5,
    displacement_alpha: float = 6.0,
    displacement_sigma: float = 12.0,
    pipe: Optional[Any] = None,
) -> Tuple[Image.Image, str]:
    """
    Three-way per-sample draw: `sketch_probability` is P(substitute at all, vs. keep
    `image`); `ultrasketch_probability` is the conditional split of that substituted mass
    between UltraSketch and displacement.
    """
    if seed is None:
        import os

        seed = int.from_bytes(os.urandom(8), "big")

    draw = np.random.default_rng(seed).random()
    original_threshold = 1.0 - sketch_probability
    ultrasketch_threshold = original_threshold + sketch_probability * ultrasketch_probability

    if draw < original_threshold:
        return image, "original"

    if draw < ultrasketch_threshold:
        method = "ultrasketch"
        active_pipe = pipe if pipe is not None else load_ultrasketch_pipeline()
        import torch

        generator = torch.Generator(device=getattr(active_pipe, "device", "cpu")).manual_seed(seed)
        sketch = run_ultrasketch(active_pipe, image, generator=generator)
    else:
        method = "displacement"
        sketch = random_displacement_field(
            image, alpha=displacement_alpha, sigma=displacement_sigma, seed=seed
        )
    return sketch, method
