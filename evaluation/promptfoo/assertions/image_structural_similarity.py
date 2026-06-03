from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.compute_ssim import compute_image_ssim


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    input_image = vars_.get("input_image")
    threshold = float(vars_.get("similarity_threshold", 0.75))

    input_image = Path(input_image)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            generated_image = tmp_dir / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            score = compute_image_ssim(
                image_a=input_image,
                image_b=generated_image,
            )

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": f"Image SSIM={score:.4f}, threshold={threshold:.4f}",
            "namedScores": {
                "image_ssim": score,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Rendering or similarity failed: {e}\n{traceback.format_exc()}",
        }