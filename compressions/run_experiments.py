#!/usr/bin/env python3
"""
Main entry point for compression ablation experiments.

Runs comprehensive ablations testing different compression architectures,
ratios, and loss combinations on SQuAD Q&A.

Usage:
    # Run all ablations (4 GPUs in parallel)
    python compressions/run_experiments.py

    # Run single ablation
    python -c "from compressions.experiments import run_single_ablation;
               from compressions import CompressionConfig;
               config = CompressionConfig(architecture='gist', target_length=64);
               run_single_ablation(config, gpu_id=0)"
"""

import sys
from pathlib import Path

# Add experimental/learning to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "experimental" / "learning"))

# Import the full experimental ablations runner
from compression_ablations import run_all_ablations

if __name__ == "__main__":
    print("=" * 80)
    print("COMPRESSION ABLATION EXPERIMENTS")
    print("=" * 80)
    print()
    print("Testing architectures: cross_attention, conv, pooling, gist")
    print("Compression ratios: 32, 64, 128 tokens")
    print("Task: SQuAD Q&A")
    print()
    print("=" * 80)
    print()

    # Run the complete ablation suite
    run_all_ablations()
