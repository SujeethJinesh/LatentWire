"""Q-KVComm-like baseline: layerwise mixed-precision schedule under a byte cap."""
from __future__ import annotations

from typing import List


def layerwise_schedule(num_layers: int, budget_bytes: float, bytes_int8: float, bytes_int4: float) -> List[str]:
    """Greedy schedule: allocate int8 to early layers until budget, then int4."""
    if num_layers <= 0:
        return []
    schedule = ["int4"] * num_layers
    if budget_bytes is None:
        return schedule
    remaining = float(budget_bytes)
    for i in range(num_layers):
        cost = bytes_int8
        if remaining >= cost:
            schedule[i] = "int8"
            remaining -= cost
        else:
            schedule[i] = "int4"
    return schedule
