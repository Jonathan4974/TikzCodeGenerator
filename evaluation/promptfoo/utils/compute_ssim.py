from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_image_for_similarity(
    image_path: str | Path,
    size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")

    white_background = Image.new("RGBA", img.size, "WHITE")
    white_background.alpha_composite(img)

    img = white_background.convert("RGB")
    img = img.resize(size)

    return np.array(img)


def compute_image_ssim(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
) -> float:
    arr_a = load_image_for_similarity(image_a, size=size)
    arr_b = load_image_for_similarity(image_b, size=size)

    return float(
        ssim(
            arr_a,
            arr_b,
            channel_axis=2,
            data_range=255,
        )
    )