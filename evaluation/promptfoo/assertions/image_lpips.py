from pathlib import Path
import tempfile
import traceback

from pf_utils.tikz_rendering import render_tex_to_png
from pf_utils.lpips_similarity import (
    compute_lpips_distance,
    lpips_distance_to_similarity,
)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    threshold = float(config.get("threshold", 0.75))
    reference_image = vars_.get("reference_image")
    net_type = str(config.get("net_type", vars_.get("lpips_net_type", "alex")))

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

            distance = compute_lpips_distance(
                image_a=reference_image,
                image_b=generated_image,
                net_type=net_type,
            )

            similarity = lpips_distance_to_similarity(distance)

        return {
            "pass": similarity >= threshold,
            "score": similarity,
            "reason": (
                f"LPIPS similarity={similarity:.4f}, "
                f"LPIPS distance={distance:.4f}"
            )
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"LPIPS failed: {e}\n{traceback.format_exc()}",
        }