from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


_lpips_metric = None


def get_lpips_metric(
    net_type: str = "alex",
    normalize: bool = True,
):
    global _lpips_metric

    if _lpips_metric is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=normalize,
        ).to(device)

        _lpips_metric.eval()

    return _lpips_metric


def load_image_for_lpips(
    image_path: str | Path,
    size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")

    white_background = Image.new("RGBA", img.size, "WHITE")
    white_background.alpha_composite(img)

    img = white_background.convert("RGB")
    img = img.resize(size)

    arr = np.asarray(img).astype(np.float32) / 255.0

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    # Shape: [1, 3, H, W]
    return torch.from_numpy(arr).unsqueeze(0)


def compute_lpips_distance(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
    net_type: str = "alex",
) -> float:
    """
    LPIPS distance.

    Niedriger ist besser.
    0.0 bedeutet sehr ähnlich.

    normalize=True bedeutet:
    Input-Bilder dürfen im Bereich [0, 1] liegen.
    """

    metric = get_lpips_metric(
        net_type=net_type,
        normalize=True,
    )

    device = next(metric.parameters()).device

    img_a = load_image_for_lpips(image_a, size=size).to(device)
    img_b = load_image_for_lpips(image_b, size=size).to(device)

    with torch.no_grad():
        distance = metric(img_a, img_b)

    return float(distance.item())