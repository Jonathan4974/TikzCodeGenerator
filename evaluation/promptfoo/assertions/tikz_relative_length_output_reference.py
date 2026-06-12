from pathlib import Path


def load_reference_text(reference_text: str) -> str:
    if reference_text.startswith("file://"):
        path = Path(reference_text.removeprefix("file://"))
        return path.read_text(encoding="utf-8")

    path = Path(reference_text)
    return path.read_text(encoding="utf-8")


def length_error_to_similarity(relative_error: float) -> float:
    return 1.0 / (1.0 + relative_error)


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    reference_text = vars_.get("reference_text")

    config = context.get("config")
    threshold = float(config.get("threshold", 0.75))

    if not reference_text:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Missing vars.reference_text",
        }

    try:
        reference = load_reference_text(reference_text)

        output_len = len(output.strip())
        reference_len = len(reference.strip())

        if reference_len == 0:
            return {
                "pass": False,
                "score": 0.0,
                "reason": "Reference text is empty",
            }

        relative_error = abs(output_len - reference_len) / reference_len
        similarity = length_error_to_similarity(relative_error)
        ratio = output_len / reference_len

        return {
            "pass": similarity >= threshold,
            "score": similarity,
            "reason": (
                f"Relative length similarity={similarity:.4f}, "
                f"relative error={relative_error:.4f}, "
                f"output_len={output_len}, "
                f"reference_len={reference_len}, "
                f"ratio={ratio:.4f}"
            ),
            "namedScores": {
                "output_length": output_len,
                "reference_length": reference_len,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Length comparison failed: {e}",
        }