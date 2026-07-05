from dataclasses import dataclass
from pathlib import Path
import tempfile

from PIL import Image

from pf_utils.clip_siglip_metric import image_cosine_similarity
from pf_utils.lpips_metric import compute_lpips_similarity
from pf_utils.dreamsim_metric import compute_dreamsim_score


@dataclass
class VisualResult:
    score: float
    siglip: float
    lpips: float
    dreamsim: float
    reason: str


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def save_tmp_image(image: Image.Image, path: Path):
    image.convert("RGB").save(path)


def visual_reward_func(
    cfg,
    input_image: Image.Image,
    rendered_image: Image.Image,
) -> VisualResult:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        input_path = tmp_dir / "input.png"
        rendered_path = tmp_dir / "rendered.png"

        save_tmp_image(input_image, input_path)
        save_tmp_image(rendered_image, rendered_path)

        # semantic
        siglip = image_cosine_similarity(
            input_path,
            rendered_path,
            model_key="siglip",
        )

        # structural
        lpips = compute_lpips_similarity(
            input_path,
            rendered_path,
        )

        # mixed semantic / perceptual
        dreamsim = compute_dreamsim_score(
            str(input_path),
            str(rendered_path),
        )

    siglip = clamp01(siglip)
    lpips = clamp01(lpips)
    dreamsim = clamp01(dreamsim)

    score = (
        cfg.siglip_multiplier * siglip
        + cfg.lpips_multiplier * lpips
        + cfg.dreamsim_multiplier * dreamsim
    )

    return VisualResult(
        score=score,
        siglip=siglip,
        lpips=lpips,
        dreamsim=dreamsim,
        reason=(
            f"visual_score={score:.3f}; "
            f"siglip={siglip:.3f}; "
            f"lpips={lpips:.3f}; "
            f"dreamsim={dreamsim:.3f}"
        ),
    )