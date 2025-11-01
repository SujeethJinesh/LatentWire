"""Checkpoint management for learned projections."""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .models import LearnedProjection


def get_projection_path(dim_a: int, dim_b: int, base_dir: Optional[Path] = None) -> Path:
    """
    Get the standard path for a projection checkpoint.

    Args:
        dim_a: Source dimension
        dim_b: Target dimension
        base_dir: Base directory for checkpoints (default: cross_model/runs/learned_projection)

    Returns:
        Path to projection checkpoint
    """
    if base_dir is None:
        base_dir = Path(__file__).parent / "runs" / "learned_projection"

    return base_dir / f"projection_{dim_a}_to_{dim_b}.pt"


def save_projection(projection: LearnedProjection, dim_a: int, dim_b: int,
                   base_dir: Optional[Path] = None) -> Path:
    """
    Save a learned projection to disk.

    Args:
        projection: Trained LearnedProjection module
        dim_a: Source dimension
        dim_b: Target dimension
        base_dir: Base directory for checkpoints

    Returns:
        Path where projection was saved
    """
    save_path = get_projection_path(dim_a, dim_b, base_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(projection.state_dict(), save_path)
    print(f"  ✓ Saved projection to {save_path}")

    return save_path


def load_projection(dim_a: int, dim_b: int, device: str = "cuda", dtype=None,
                   base_dir: Optional[Path] = None) -> Optional[LearnedProjection]:
    """
    Load a learned projection from disk.

    Args:
        dim_a: Source dimension
        dim_b: Target dimension
        device: Device to load model on
        dtype: Data type for model weights
        base_dir: Base directory for checkpoints

    Returns:
        Loaded LearnedProjection or None if not found
    """
    load_path = get_projection_path(dim_a, dim_b, base_dir)

    if not load_path.exists():
        print(f"  ✗ No projection found at {load_path}")
        return None

    try:
        projection = LearnedProjection(dim_a, dim_b)

        if dtype is not None:
            projection = projection.to(device=device, dtype=dtype)
        else:
            projection = projection.to(device)

        projection.eval()

        state = torch.load(load_path, map_location=device, weights_only=False)
        projection.load_state_dict(state)

        print(f"  ✓ Loaded projection from {load_path}")
        return projection

    except Exception as e:
        print(f"  ✗ Failed to load projection: {e}")
        return None


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save experiment results to JSON file.

    Args:
        results: Dictionary of results to save
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Saved results to {output_path}")
