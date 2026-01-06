#!/usr/bin/env python3
"""
Main entry point for the finalized LatentWire system.
Imports and exposes key components from consolidated directories.
"""

import sys
import os

# Add consolidated directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'consolidated_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'consolidated_eval'))

# Import core components
from models import Encoder, Adapter, LMWrapper
from losses import k_token_ce_from_prefix, kd_first_k_prefix_vs_text
from config import Config
from eval import main as eval_main

def main():
    """Main entry point for the system."""
    print("LatentWire System - Finalized Version")
    print("======================================")
    print("\nAvailable commands:")
    print("  python main.py train    - Run training")
    print("  python main.py eval     - Run evaluation")
    print("  ./RUN_ALL.sh           - Run complete pipeline")
    print("\nFor detailed usage, see README.md")

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "eval":
            eval_main()
        elif command == "train":
            print("Training functionality - use RUN_ALL.sh")
        else:
            print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()