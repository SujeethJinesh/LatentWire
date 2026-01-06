# Linear Probe Baseline Setup Documentation

## Overview
The LinearProbeBaseline has been successfully copied to the finalization directory to meet reviewer requirements. This baseline provides a scientifically rigorous comparison using sklearn's LogisticRegression.

## File Locations
- **Main implementation**: `finalization/linear_probe_baseline.py` (622 lines)
- **Integration point**: `finalization/MAIN_EXPERIMENT.py` (imports and uses LinearProbeBaseline)
- **Verification script**: `finalization/verify_linear_probe.py`

## Key Features
1. **Scientific Rigor**: Uses sklearn's LogisticRegression (not neural networks)
2. **Memory Efficient**: Batch processing to handle large datasets
3. **Cross-Validation**: Proper stratified k-fold validation
4. **Reproducibility**: Save/load probe weights via joblib
5. **Integration**: Works with MAIN_EXPERIMENT.py's unified framework

## Dependencies
All required dependencies are already in `requirements.txt`:
- `scikit-learn>=1.0.0` (provides LogisticRegression)
- `joblib` (comes with scikit-learn, for saving/loading)
- `numpy>=1.21.0` (for array operations)
- `torch>=2.2.0` (for extracting hidden states from models)
- `transformers==4.45.2` (for model loading)

## Import Strategy
The code uses a fallback import strategy:
1. First tries to import from local `finalization/` directory
2. Falls back to `telepathy/` module if local import fails
3. This ensures compatibility both in finalization and main codebase

## Usage in MAIN_EXPERIMENT.py

```python
# When compression_type = "linear_probe"
compressor = LinearProbeCompressor()
compressor.initialize(
    hidden_dim=4096,  # For Llama-8B
    num_classes=2,    # For binary classification
    layer_idx=24      # Layer 24 often works well
)
```

## Verification
Run the verification script to check setup:
```bash
python3 verify_linear_probe.py
```

Note: Local verification will show warnings about missing torch/sklearn, which is expected since these are installed on HPC.

## HPC Execution
On the HPC cluster with full dependencies:
```bash
python MAIN_EXPERIMENT.py \
    --compression_type linear_probe \
    --dataset sst2 \
    --model_size 8B
```

## Architecture Details
1. **Extract**: Frozen embeddings from source model (e.g., Llama layer 24)
2. **Pool**: Hidden states (mean/last-token/first-token pooling)
3. **Train**: LogisticRegression with L2 regularization
4. **Evaluate**: On classification tasks (SST-2, AG News, TREC)

## Why This Baseline Matters
Reviewers require this baseline to test whether the sender model's hidden states already contain sufficient information for the task, without needing cross-model transfer. It provides a critical comparison point for the Telepathy/Bridge method.