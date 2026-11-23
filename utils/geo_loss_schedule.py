from __future__ import annotations


DEFAULT_GEO_LOSS_WARMUP_START = 1e-3
DEFAULT_GEO_LOSS_WARMUP_STEPS = 500


def compute_geo_loss_warmup_scale(
    *,
    global_step: int,
    warmup_steps: int = DEFAULT_GEO_LOSS_WARMUP_STEPS,
    start: float = DEFAULT_GEO_LOSS_WARMUP_START,
) -> float:
    if warmup_steps <= 0:
        return 1.0

    clamped_start = float(max(0.0, min(start, 1.0)))
    clamped_step = max(int(global_step), 0)
    if clamped_step >= warmup_steps:
        return 1.0

    progress = clamped_step / float(warmup_steps)
    return clamped_start + (1.0 - clamped_start) * progress
