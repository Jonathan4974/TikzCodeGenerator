from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png
from utils.lpips_similarity import compute_lpips_distance


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    reference_image = vars_.get("reference_image") or vars_.get("input_image")

    threshold = float(
        config.get(
            "threshold",
            vars_.get("lpips_threshold", 0.30),
        )
    )

    net_type = str(
        config.get(
            "net_type",
            vars_.get("lpips_net_type", "alex"),
        )
    )

    if not reference_image:
        return {
            "pass": False,
            "score": 999.0,
            "reason": "Missing vars.input_image or vars.reference_image",
        }

    reference_image = Path(reference_image)

    if not reference_image.exists():
        return {
            "pass": False,
            "score": 999.0,
            "reason": f"Reference image does not exist: {reference_image}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_image = Path(tmp_dir) / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            distance = compute_lpips_distance(
                image_a=reference_image,
                image_b=generated_image,
                net_type=net_type,
            )

        return {
            "pass": distance <= threshold,
            "score": distance,
            "reason": (
                f"LPIPS distance={distance:.4f}, "
                f"threshold={threshold:.4f}, "
                f"net_type={net_type}"
            ),
            "namedScores": {
                "lpips_distance": distance,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 999.0,
            "reason": f"LPIPS failed: {e}\n{traceback.format_exc()}",
        }