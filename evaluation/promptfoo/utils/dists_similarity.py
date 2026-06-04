from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity


_dists_metric = None
_dists_device = None


def get_dists_metric():
    global _dists_metric, _dists_device

    if _dists_metric is None:
        _dists_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _dists_metric = DeepImageStructureAndTextureSimilarity(
            reduction="mean",
        ).to(_dists_device)

        _dists_metric.eval()

    return _dists_metric, _dists_device


def load_image_for_dists(
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

    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def compute_dists_distance(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
) -> float:
    metric, device = get_dists_metric()

    img_a = load_image_for_dists(image_a, size=size).to(device)
    img_b = load_image_for_dists(image_b, size=size).to(device)

    with torch.no_grad():
        distance = metric(img_a, img_b)

    return float(distance.item())


def dists_distance_to_similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def compute_dists_similarity(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
) -> float:
    distance = compute_dists_distance(
        image_a=image_a,
        image_b=image_b,
        size=size,
    )

    return dists_distance_to_similarity(distance)