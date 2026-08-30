"""Shared helpers for diagnostic counters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping


def aggregate_rejections(
    pipeline: Mapping[str, int], risk: Mapping[str, int]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Sum same-named reasons while preserving their stage attribution."""
    by_stage = {
        "pipeline": dict(pipeline),
        "risk": dict(risk),
    }
    total: Counter[str] = Counter()
    total.update(pipeline)
    total.update(risk)
    return dict(total), by_stage
