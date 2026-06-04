from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


_lpips_metric = None
_lpips_device = None
_lpips_net_type = None


def get_lpips_metric(net_type: str = "alex", normalize: bool = True):
    global _lpips_metric, _lpips_device, _lpips_net_type

    if _lpips_metric is None or _lpips_net_type != net_type:
        _lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_net_type = net_type

        _lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=normalize,
        ).to(_lpips_device)

        _lpips_metric.eval()

    return _lpips_metric, _lpips_device


def load_image_for_lpips(image_path: str | Path, size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")

    white_background = Image.new("RGBA", img.size, "WHITE")
    white_background.alpha_composite(img)

    img = white_background.convert("RGB")
    img = img.resize(size)

    arr = np.asarray(img).astype(np.float32) / 255.0

    # HWC -> CHW, plus Batch-Dimension
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def compute_lpips_distance(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
    net_type: str = "alex"
) -> float:
    
    metric, device = get_lpips_metric(net_type=net_type, normalize=True)

    img_a = load_image_for_lpips(image_a, size=size).to(device)
    img_b = load_image_for_lpips(image_b, size=size).to(device)

    with torch.no_grad():
        distance = metric(img_a, img_b)

    return float(distance.item())


def lpips_distance_to_similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def compute_lpips_similarity(
    image_a: str | Path,
    image_b: str | Path,
    size: tuple[int, int] = (512, 512),
    net_type: str = "alex",
) -> float:
    distance = compute_lpips_distance(
        image_a=image_a,
        image_b=image_b,
        size=size,
        net_type=net_type,
    )

    return lpips_distance_to_similarity(distance)