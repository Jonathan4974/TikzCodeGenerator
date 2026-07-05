"""UltraSketch/displacement-field helpers for real synthetic sketch generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

MULTIPLE_OF = 16

INFERENCE_PARAMS = dict(
    num_inference_steps=50,
    image_guidance_scale=1.7,
    guidance_scale=1.5,
    strength=0.9,
)

ULTRASKETCH_PROMPT = "Turn it into a hand-drawn sketch"


def assert_multiple_of(image: Image.Image, label: str, multiple: int = MULTIPLE_OF) -> None:
    width, height = image.size
    if width % multiple != 0 or height % multiple != 0:
        raise ValueError(
            f"{label} has size {image.size}, expected both dimensions to be "
            f"multiples of {multiple}"
        )


def resize_to_multiple(image: Image.Image, multiple: int = MULTIPLE_OF) -> Image.Image:
    width, height = image.size
    new_width = max(multiple, (width // multiple) * multiple)
    new_height = max(multiple, (height // multiple) * multiple)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_like(image: Image.Image, reference: Image.Image) -> Image.Image:
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.LANCZOS)


def random_displacement_field(image: Image.Image, alpha: float = 6, sigma: float = 12, seed: int = 42) -> Image.Image:
    assert_multiple_of(image, "displacement input", MULTIPLE_OF)
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


def run_ultrasketch(pipe: Any, image: Image.Image, prompt: str) -> Image.Image:
    image = resize_to_multiple(image).convert("RGB")
    assert_multiple_of(image, "UltraSketch input", MULTIPLE_OF)
    output = pipe(
        prompt=prompt,
        image=image,
        mask_img=Image.new("RGB", image.size, "white"),
        **INFERENCE_PARAMS,
    ).images[0]
    output = resize_like(output.convert("RGB"), image)
    assert_multiple_of(output, "UltraSketch output", MULTIPLE_OF)
    return output


def load_ultrasketch_pipeline() -> Any:
    """Load the nllg/ultrasketch pipeline onto the GPU.

    Imports torch/diffusers locally so this module (and everything that imports it)
    stays importable in environments without those heavy dependencies installed.
    """
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="nllg/ultrasketch",
        custom_pipeline="nllg/ultrasketch",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    return pipe
