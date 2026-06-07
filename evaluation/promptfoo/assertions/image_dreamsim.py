from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.dreamsim_similarity import (
    compute_dreamsim_distance,
    dreamsim_distance_to_similarity,
)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})

    input_image = vars_.get("input_image")
    threshold = float(vars_.get("dreamsim_threshold", 0.75))

    if not input_image:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Missing vars.input_image. Available vars: {list(vars_.keys())}",
        }

    input_image = Path(input_image)

    if not input_image.exists():
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Input image does not exist: {input_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            distance = compute_dreamsim_distance(
                image_a=input_image,
                image_b=generated_image,
            )
            score = dreamsim_distance_to_similarity(distance)

        return {
            "pass": score >= threshold,
            "score": score,
            "reason": (
                f"DreamSim similarity={score:.4f}, "
                f"DreamSim distance={distance:.4f}, "
                f"threshold={threshold:.4f}"
            ),
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"DreamSim failed: {e}\n{traceback.format_exc()}",
        }