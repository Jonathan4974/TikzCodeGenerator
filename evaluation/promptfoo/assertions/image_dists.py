from pathlib import Path
import tempfile
import traceback
import shutil

from utils.tikz_rendering import render_tex_to_png
from utils.dists_similarity import compute_dists_distance


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    reference_image = vars_.get("reference_image") or vars_.get("input_image")

    threshold = float(config.get("threshold", vars_.get("dists_threshold", 0.25)))

    debug_dir = Path(config.get("debug_dir", vars_.get("debug_dir", "/app/debug_dists")))
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not reference_image:
        return {
            "pass": False,
            "score": 999.0,
            "reason": "Missing vars.input_image or vars.reference_image",
        }

    reference_image = Path(reference_image)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            generated_image = tmp_dir / "generated.png"

            render_tex_to_png(
                tex_code=output,
                output_path=generated_image,
            )

            distance = compute_dists_distance(
                image_a=reference_image,
                image_b=generated_image,
            )

            # Debug-Dateien dauerhaft speichern
            test_id = reference_image.stem
            debug_generated = debug_dir / f"{test_id}_generated.png"
            debug_reference = debug_dir / f"{test_id}_reference.png"
            debug_output = debug_dir / f"{test_id}_output.tex"

            shutil.copyfile(generated_image, debug_generated)
            shutil.copyfile(reference_image, debug_reference)
            debug_output.write_text(output, encoding="utf-8")

        return {
            "pass": distance <= threshold,
            "score": distance,
            "reason": (
                f"DISTS distance={distance:.4f}, threshold={threshold:.4f}. "
                f"Debug generated image saved to {debug_generated}"
            ),
            "namedScores": {
                "dists_distance": distance,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 999.0,
            "reason": f"DISTS failed: {e}\n{traceback.format_exc()}",
        }