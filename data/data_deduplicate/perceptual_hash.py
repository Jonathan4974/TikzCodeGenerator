"""
Perceptual image hashing using DCT (64-bit).
Ref: Zauner, 2010 (Implementation and Benchmarking of Perceptual Image Hash Functions)
"""

from typing import Optional, Union
from PIL import Image
import numpy as np
from scipy.fftpack import dct


class PerceptualHash:
    """Compute 64-bit perceptual hash from an image (PIL Image, path, or bytes)."""

    RESIZE_DIM = 32
    DCT_SIZE = 8   # yields 64 bits

    def __init__(self, image: Union[Image.Image, str, bytes]):
        """
        Args:
            image: PIL Image object, file path, or raw image bytes.
        """
        self._image = image
        self._hash: Optional[int] = None

    def compute(self) -> int:
        """Compute and return the 64-bit hash (cached)."""
        if self._hash is not None:
            return self._hash

        # 1. Load image
        if isinstance(self._image, str):
            img = Image.open(self._image).convert('L')
        elif isinstance(self._image, bytes):
            import io
            img = Image.open(io.BytesIO(self._image)).convert('L')
        else:  # assume PIL Image
            img = self._image.convert('L')

        # 2. Resize to 32x32
        img = img.resize((self.RESIZE_DIM, self.RESIZE_DIM), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32)

        # 3. 2D DCT (orthonormal)
        dct_rows = dct(pixels, axis=0, norm='ortho')
        dct_2d = dct(dct_rows, axis=1, norm='ortho')

        # 4. Keep top-left 8x8 coefficients
        coeffs = dct_2d[:self.DCT_SIZE, :self.DCT_SIZE].flatten()

        # 5. Threshold by median
        median = np.median(coeffs)
        bits = (coeffs >= median).astype(np.uint8)

        # 6. Pack into 64-bit int
        hash_int = 0
        for b in bits:
            hash_int = (hash_int << 1) | int(b)

        self._hash = hash_int
        return hash_int

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

    @staticmethod
    def is_duplicate(h1: int, h2: int, threshold: int = 2) -> bool:
        return PerceptualHash.hamming_distance(h1, h2) <= threshold