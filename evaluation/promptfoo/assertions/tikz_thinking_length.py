def get_assert(output: str, context: dict) -> dict:
    provider_response = context.get("providerResponse") or {}
    metadata = provider_response.get("metadata") or context.get("metadata") or {}

    try:
        length = max(0, int(metadata.get("thinking_length") or 0))
    except (TypeError, ValueError):
        length = 0

    return {
        "pass": True,
        "score": 1.0,
        "reason": f"thinking_length={length} characters",
        "namedScores": {"thinking_length": float(length)},
    }