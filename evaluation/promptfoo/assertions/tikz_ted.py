from pathlib import Path
import traceback

from pf_utils.tex_edit_distance import compute_ted


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config")

    reference_path = vars_.get("reference_code")
    threshold = float(config.get("threshold", 0.75))


    try:
        reference_path = Path(reference_path)

        if not reference_path.exists():
            return {
                "pass": False,
                "score": 0.0,
                "reason": f"tikz-ted: Reference file does not exist: {reference_path}",
            }

        reference_code = reference_path.read_text(encoding="utf-8")

        ted = compute_ted(
            generated_code=output,
            reference_code=reference_code,
        )

        ted_similarity = 1.0 / (1.0 + ted)

        return {
            "pass": ted_similarity >= threshold,
            "score": ted_similarity,
            "reason": (
                f"TED={ted_similarity:.4f}, "
                f"reference={reference_path.name}"
            )
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"TED failed: {e}\n{traceback.format_exc()}",
        }