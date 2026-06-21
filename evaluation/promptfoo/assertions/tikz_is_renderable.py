from pathlib import Path
import tempfile
import traceback

from pf_utils.tikz_rendering import render_tex_to_png, TikzRenderError


def get_assert(output: str, context):
    metrics = {}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_png = Path(tmp_dir) / "rendered.png"

            render_tex_to_png(
                tex_code=output,
                output_path=output_png,
                metrics=metrics,
            )

            ok = output_png.exists() and output_png.stat().st_size > 0

        errors = metrics.get("latex_errors", 0)
        warnings = metrics.get("latex_warnings", 0)
        badboxes = metrics.get("latex_badboxes", 0)

        return {
            "pass": ok,
            "score": 1.0 if ok else 0.0,
            "reason": (
                f"renderable={ok}; "
                f"errors={errors}; warnings={warnings}; badboxes={badboxes}"
            ),
            "namedScores": {
                "latex_errors": errors,
                "latex_warnings": warnings,
                "latex_badboxes": badboxes,
            },
        }

    except TikzRenderError as e:
        metrics.update(getattr(e, "metrics", {}))

        return {
            "pass": False,
            "score": 0.0,
            "reason": f"not renderable: {e}",
            "namedScores": {
                "latex_errors": metrics.get("latex_errors", 0),
                "latex_warnings": metrics.get("latex_warnings", 0),
                "latex_badboxes": metrics.get("latex_badboxes", 0),
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"not renderable: {e}\n{traceback.format_exc()}",
            "namedScores": {
                "latex_errors": metrics.get("latex_errors", 0),
                "latex_warnings": metrics.get("latex_warnings", 0),
                "latex_badboxes": metrics.get("latex_badboxes", 0),
            },
        }