from pathlib import Path
import tempfile
import traceback

from utils.tikz_rendering import render_tex_to_png


def get_assert(output: str, context):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_png = Path(tmp_dir) / "rendered.png"

            render_tex_to_png(
                tex_code=output,
                output_path=output_png,
            )

            if not output_png.exists() or output_png.stat().st_size == 0:
                return {
                    "pass": False,
                    "score": 0.0,
                    "reason": "Rendering finished, but no PNG was created.",
                }

        return {
            "pass": True,
            "score": 1.0,
            "reason": "TikZ/LaTeX code is renderable.",
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"TikZ/LaTeX code is not renderable: {e}\n{traceback.format_exc()}",
        }