from pathlib import Path
import tempfile
import traceback

from pf_utils.tikz_rendering import render_tex_to_png
from pf_utils.compute_ssim import compute_image_ssim


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    reference_image = vars_.get("reference_image")
    reference_image = Path(reference_image)

    config = context.get("config")
    threshold = float(config.get("threshold", 0.75))

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            generated_image = tmp_dir / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            score = compute_image_ssim(
                image_a=reference_image,
                image_b=generated_image,
            )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"Image SSIM={score:.4f}"
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Rendering or similarity failed: {e}\n{traceback.format_exc()}",
        }