#!/usr/bin/env python3
"""
Main entry point for cross-model activation communication experiments.

Reproduces "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).

This is a fully functional entry point that runs the complete experiment suite.
"""

import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experimental" / "learning"))

# Import and run the original main function
# This gives us full functionality while maintaining the clean cross_model structure
from unified_cross_model_experiments import main as run_all_experiments


if __name__ == "__main__":
    print("=" * 80)
    print("CROSS-MODEL EXPERIMENTS (via cross_model/run_experiments.py)")
    print("=" * 80)
    print()

    # Run the complete experiment suite
    run_all_experiments()
