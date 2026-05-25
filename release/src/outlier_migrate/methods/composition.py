"""Extension stub for rotation plus budget composition."""

from __future__ import annotations

import numpy as np


def compose_rotation_with_budget(
    weights: np.ndarray,
    channel_scores: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the interface for ParoQuant-style rotation plus M11b budgeting."""

    raise NotImplementedError("ICLR follow-up: implement rotation plus budget composition")
