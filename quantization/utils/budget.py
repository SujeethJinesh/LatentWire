"""Budget helpers for per-sample byte caps."""
from __future__ import annotations


def compute_slack(cap_bytes: float, actual_bytes: float) -> float:
    if cap_bytes is None:
        return 0.0
    slack = float(cap_bytes) - float(actual_bytes)
    return slack if slack > 0 else 0.0
