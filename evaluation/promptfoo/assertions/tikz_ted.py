from pathlib import Path
import traceback

from pf_utils.ted_metric import compute_ted, ted_to_similarity


BAD_TED_DISTANCE = 999.0


def failed(reason: str):
    return {
        "pass": False,
        "score": 0.0,
        "namedScores": {
            "ted_similarity": 0.0,
            "ted_distance": BAD_TED_DISTANCE,
        },
        "reason": reason,
    }


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {}) or {}

    reference_path = vars_.get("reference_code")
    threshold = float(config.get("threshold", 0.75))
    language = str(vars_.get("ted_language", config.get("language", "en")))

    if not reference_path:
        return failed(
            f"Missing vars.reference_code. Available vars: {list(vars_.keys())}"
        )

    try:
        reference_path = Path(reference_path)

        if not reference_path.exists():
            return failed(
                f"tikz-ted: Reference file does not exist: {reference_path}"
            )

        reference_code = reference_path.read_text(encoding="utf-8")

        ted = compute_ted(
            generated_code=output,
            reference_code=reference_code,
            language=language,
        )

        ted_similarity = ted_to_similarity(ted)

        return {
            "pass": ted_similarity >= threshold,
            "score": ted_similarity,
            "namedScores": {
                "ted_similarity": ted_similarity,
                "ted_distance": ted,
            },
            "reason": (
                f"TED similarity={ted_similarity:.4f}, "
                f"TED distance={ted:.4f}, "
                f"reference={reference_path.name}"
            ),
        }

    except Exception as e:
        return failed(f"TED failed: {e}\n{traceback.format_exc()}")