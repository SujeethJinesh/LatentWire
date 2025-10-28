#!/bin/bash
# Simple wrapper to run cross_model_ablation.py with output capture
python experimental/learning/cross_model_ablation.py 2>&1 | tee experimental/learning/ablation_simple_results.txt
