from __future__ import annotations

import numpy as np
from PIL import Image


def pixel_congruence_coefficient(img1: Image.Image, img2: Image.Image) -> float:
    """
    Simple pixel-cc implementation.
    """
    a = np.array(img1.convert("L")).astype(float).flatten()
    b = np.array(img2.convert("L")).astype(float).flatten()
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(numerator / denominator) if denominator != 0 else 0.0
