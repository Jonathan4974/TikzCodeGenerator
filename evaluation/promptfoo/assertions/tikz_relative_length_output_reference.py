from pathlib import Path


def load_reference_text(reference_text: str) -> str:
    if reference_text.startswith("file://"):
        path = Path(reference_text.removeprefix("file://"))
        return path.read_text(encoding="utf-8")

    path = Path(reference_text)
    return path.read_text(encoding="utf-8")


def get_assert(output: str, context):
    vars_ = context.get("vars", {})
    config = context.get("config", {})

    reference_text = vars_.get("reference_text")

    threshold = float(
        config.get(
            "threshold",
            vars_.get("relative_length_threshold", 0.25),
        )
    )

    if not reference_text:
        return {
            "pass": False,
            "score": 999.0,
            "reason": "Missing vars.reference_text",
        }

    try:
        reference = load_reference_text(reference_text)
        print(reference)
        output_len = len(output)
        reference_len = len(reference)

        if reference_len == 0:
            return {
                "pass": False,
                "score": 999.0,
                "reason": "Reference text is empty",
            }

        relative_error = abs(output_len - reference_len) / reference_len
        ratio = output_len / reference_len

        return {
            "pass": relative_error <= threshold,
            "score": relative_error,
            "reason": (
                f"Relative length error={relative_error:.4f}, "
                f"threshold={threshold:.4f}, "
                f"output_len={output_len}, "
                f"reference_len={reference_len}, "
                f"ratio={ratio:.4f}"
            ),
            "namedScores": {
                "relative_length_error": relative_error,
                "length_ratio": ratio,
                "output_length": output_len,
                "reference_length": reference_len,
            },
        }

    except Exception as e:
        return {
            "pass": False,
            "score": 999.0,
            "reason": f"Length comparison failed: {e}",
        }