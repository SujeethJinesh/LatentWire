#!/usr/bin/env bash
"""
Run enhanced cross-model alignment experiments with comprehensive logging.
Incorporates 2024 research: InfoNCE, CKA, multi-layer alignment, 10K samples.
"""

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/enhanced_experiments}"
SCRIPT_NAME="enhanced_unified_experiments.py"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_LAUNCH_BLOCKING=1  # For debugging CUDA issues

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/enhanced_experiments_${TIMESTAMP}.log"

echo "==================================================================================" | tee "$LOG_FILE"
echo "ENHANCED CROSS-MODEL ALIGNMENT EXPERIMENTS (2024 Research)" | tee -a "$LOG_FILE"
echo "==================================================================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Key Enhancements Based on 2024 Literature:" | tee -a "$LOG_FILE"
echo "  ✓ InfoNCE contrastive loss (τ=0.07)" | tee -a "$LOG_FILE"
echo "  ✓ CKA similarity metric (superior to SVCCA)" | tee -a "$LOG_FILE"
echo "  ✓ Soft contrastive learning with similarity labels" | tee -a "$LOG_FILE"
echo "  ✓ Multi-layer alignment [8, 16, 24]" | tee -a "$LOG_FILE"
echo "  ✓ 10,000 training samples (10x increase)" | tee -a "$LOG_FILE"
echo "  ✓ Batch size 16 (4x increase for contrastive)" | tee -a "$LOG_FILE"
echo "  ✓ 10 epochs with cosine annealing" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "GPU Allocation:" | tee -a "$LOG_FILE"
echo "  - CPU: Procrustes alignment baseline" | tee -a "$LOG_FILE"
echo "  - GPU 0: Linear adapter with InfoNCE" | tee -a "$LOG_FILE"
echo "  - GPU 1: Affine adapter with InfoNCE" | tee -a "$LOG_FILE"
echo "  - GPU 0 (sequential): LoRA adapter with InfoNCE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Expected Results (from literature):" | tee -a "$LOG_FILE"
echo "  - CKA scores: 0.6-0.7 (vs 0.3 baseline)" | tee -a "$LOG_FILE"
echo "  - Generation loss: <2.0 (vs 3.4 baseline)" | tee -a "$LOG_FILE"
echo "  - No mode collapse (contrastive prevents)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Starting enhanced experiments..." | tee -a "$LOG_FILE"
echo "==================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the enhanced experiment script with comprehensive logging
{
    python experimental/learning/$SCRIPT_NAME
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==================================================================================" | tee -a "$LOG_FILE"
echo "ENHANCED EXPERIMENTS COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================================================" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - Configuration: $OUTPUT_DIR/config_*.json" | tee -a "$LOG_FILE"
echo "  - Procrustes: $OUTPUT_DIR/procrustes_results_*.json" | tee -a "$LOG_FILE"
echo "  - Linear adapter: $OUTPUT_DIR/linear_results_*.json" | tee -a "$LOG_FILE"
echo "  - Affine adapter: $OUTPUT_DIR/affine_results_*.json" | tee -a "$LOG_FILE"
echo "  - LoRA adapter: $OUTPUT_DIR/lora_results_*.json" | tee -a "$LOG_FILE"
echo "  - Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Key Metrics to Check:" | tee -a "$LOG_FILE"
echo "  1. CKA scores (should reach 0.6+)" | tee -a "$LOG_FILE"
echo "  2. Generation loss convergence" | tee -a "$LOG_FILE"
echo "  3. Contrastive loss reduction" | tee -a "$LOG_FILE"
echo "  4. Sample diversity (no mode collapse)" | tee -a "$LOG_FILE"
echo "==================================================================================" | tee -a "$LOG_FILE"

# Create summary file with key results
SUMMARY_FILE="$OUTPUT_DIR/summary_${TIMESTAMP}.txt"
echo "Enhanced Experiment Summary - $TIMESTAMP" > "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Configuration:" >> "$SUMMARY_FILE"
echo "  - Training samples: 10,000" >> "$SUMMARY_FILE"
echo "  - Batch size: 16" >> "$SUMMARY_FILE"
echo "  - Epochs: 10" >> "$SUMMARY_FILE"
echo "  - Contrastive weight: 0.3" >> "$SUMMARY_FILE"
echo "  - Temperature: 0.07" >> "$SUMMARY_FILE"
echo "  - Alignment layers: [8, 16, 24]" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Check the following files for results:" >> "$SUMMARY_FILE"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null >> "$SUMMARY_FILE" || echo "No JSON results yet" >> "$SUMMARY_FILE"

echo "Summary written to: $SUMMARY_FILE" | tee -a "$LOG_FILE"