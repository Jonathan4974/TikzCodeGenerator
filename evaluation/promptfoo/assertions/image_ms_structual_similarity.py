from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.compute_ms_ssim import compute_image_ms_ssim


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))
    reference_image = vars_.get("reference_image")

    if not reference_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.input_image or vars.reference_image",
        }

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Reference image does not exist: {reference_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(tex_code=output, output_path=generated_image)

            similarity = compute_image_ms_ssim(
                image_a=reference_image,
                image_b=generated_image
            )

        return {
            "pass": similarity >= threshold,
            "score": similarity,
            "reason": f"MS-SSIM similarity={similarity:.4f}",
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"MS-SSIM failed: {e}\n{traceback.format_exc()}",
        }