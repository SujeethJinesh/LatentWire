"""Command-line entrypoints for LatentWire workflows."""

from .train import main as train_main  # noqa: F401
from .eval import main as eval_main    # noqa: F401
from .run_ablation import main as ablation_main  # noqa: F401

