from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, resize
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import os


def load_image_for_ms_ssim(
    image_path: str | Path,
    size: tuple[int, int],
) -> torch.Tensor:
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGBA")

    white_background = Image.new("RGBA", img.size, "WHITE")
    white_background.alpha_composite(img)

    img = white_background.convert("RGB")

    tensor = to_tensor(img)  # [3, H, W], float in [0, 1]
    tensor = resize(tensor, list(size))  # [3, H, W]
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

    return tensor


def compute_image_ms_ssim(
    image_a: str | Path,
    image_b: str | Path,
) -> float:
    
    ref_image_size = int(os.getenv("REF_IMAGE_SIZE", "384"))
    size = (ref_image_size, ref_image_size)

    tensor_a = load_image_for_ms_ssim(image_a, size=size)
    tensor_b = load_image_for_ms_ssim(image_b, size=size)

    metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    with torch.no_grad():
        score = metric(tensor_a, tensor_b)

    return float(score.item())