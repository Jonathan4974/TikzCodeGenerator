from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    score: float
    penalty: float
    reason: str


def diagnostic_reward_func(cfg, errors: int, warnings: int, badboxes: int) -> DiagnosticResult:
    penalty = (
        cfg.error_multiplier * errors
        + cfg.warning_multiplier * warnings
        + cfg.badboxes_multiplier * badboxes
    )

    score = cfg.diagnostic_base_max_score - penalty
    score = max(cfg.diagnostic_base_min_score, min(cfg.diagnostic_base_max_score, score))

    return DiagnosticResult(
        score=score,
        penalty=penalty,
        reason=(
            f"errors={errors}; warnings={warnings}; badboxes={badboxes}; "
            f"penalty={penalty:.3f}; score={score:.3f}"
        ),
    )