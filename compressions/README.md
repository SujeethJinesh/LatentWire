# Compression Experiments

Comprehensive ablations for prompt/context compression on SQuAD Q&A task.

## Overview

This module tests different architectures and strategies for compressing long prompts into shorter representations while maintaining task performance. The goal is to reduce compute costs and memory usage without sacrificing quality.

## Structure

```
compressions/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── config.py                   # CompressionConfig dataclass
├── models.py                   # Compression architectures
├── dataset.py                  # SQuAD dataset wrapper
├── experiments/                # Individual experiment modules (future)
│   └── __init__.py
└── run_experiments.py          # Main entry point
```

## Compression Architectures

### 1. Cross-Attention Compressor
**Reference:** "Perceiver: General Perception with Iterative Attention" (Jaegle et al., 2021)

- **Method:** Learnable query tokens attend to full sequence via cross-attention
- **Pros:** Most flexible, can select information from anywhere
- **Cons:** More parameters, slower than convolutional
- **Params:** ~100M (for M=64, hidden_dim=4096)

### 2. Convolutional Compressor
**Reference:** "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)

- **Method:** Strided 1D depthwise convolution + adaptive pooling
- **Pros:** Fast, maintains local order, fewer parameters
- **Cons:** Limited receptive field, less flexible than attention
- **Params:** ~33M (for M=64, hidden_dim=4096)

### 3. Weighted Pooling Compressor
**Reference:** Adapted from "Attention is All You Need" (Vaswani et al., 2017)

- **Method:** Learned weighted average over fixed windows
- **Pros:** Simplest, most interpretable, fewest parameters
- **Cons:** Fixed window boundaries, less expressive
- **Params:** ~16M (for M=64, hidden_dim=4096)

### 4. Gist Compressor ⭐ **NEW**
**Reference:** "Learning to Compress Prompts with Gist Tokens" (Mu et al., NeurIPS 2023)
**Paper:** [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)

- **Method:** Learnable "gist" tokens that compress sequence via:
  1. Cross-attention to full sequence (compression)
  2. Self-attention among gist tokens (refinement)
  3. Feed-forward refinement
- **Key Innovation:** Trained via masked instruction finetuning with modified attention masks
- **Results (paper):** Up to 26× compression, 40% FLOPs reduction, 4.2% wall time speedup
- **Pros:** State-of-the-art compression, minimal quality loss, generalizes zero-shot
- **Cons:** Requires careful attention masking during training
- **Params:** ~134M (for M=64, hidden_dim=4096)

## Ablation Dimensions

### Compression Ratios
- **M=32:** 16× compression (512 → 32 tokens)
- **M=64:** 8× compression (512 → 64 tokens)
- **M=128:** 4× compression (512 → 128 tokens)

### Loss Functions
Default weights (balanced):
```python
loss_weights = {
    'teacher_forcing': 0.5,  # Supervised on answer generation
    'generation': 0.3,       # Match eval objective
    'contrastive': 0.2       # Prevent representation collapse
}
```

Ablations test different weightings to find optimal balance.

### Training Configuration
- **Model:** Llama-3.1-8B-Instruct (frozen, with LoRA adapters)
- **Dataset:** SQuAD v1.1 (10K train, 1K validation)
- **Batch Size:** 8 per GPU
- **Gradient Accumulation:** 4 steps (effective batch=32)
- **Optimizer:** AdamW (lr=1e-4 for compressor, 5e-5 for LoRA)
- **Scheduler:** Cosine with 5% warmup
- **Epochs:** 10

## Usage

### Run All Ablations (4 GPUs)
```bash
# Runs all architectures × compression ratios
python compressions/run_experiments.py
```

This tests:
- 4 architectures: cross_attention, conv, pooling, gist
- 3 compression ratios: M ∈ {32, 64, 128}
- Total: 12 experiments (assigned to 4 GPUs in parallel)

### Run Single Experiment
```python
from compressions import CompressionConfig
from experimental.learning.compression_ablations import run_single_ablation

config = CompressionConfig(
    architecture='gist',
    target_length=64,
    epochs=10,
    gpu_id=0
)

results = run_single_ablation(config, gpu_id=0)
print(f"F1: {results['final_f1']:.4f}")
print(f"Compression: {results['compression_ratio']}×")
```

### Custom Loss Weights
```python
config = CompressionConfig(
    architecture='cross_attention',
    target_length=64,
    loss_weights={
        'teacher_forcing': 0.7,  # More emphasis on supervised loss
        'generation': 0.2,
        'contrastive': 0.1
    }
)
```

## Evaluation Metrics

### Task Quality
- **F1 Score:** Token-level overlap between prediction and reference
- **Exact Match (EM):** Strict string match
- **Baseline F1/EM:** Uncompressed teacher model performance (upper bound)

### Efficiency
- **Compression Ratio:** `input_length / compressed_length` (e.g., 512/64 = 8×)
- **Parameters:** Compressor model size
- **Wall Time:** Inference time per sample (future)

### Example Results
```
Baseline (uncompressed):     F1: 0.82  EM: 0.68
Cross-Attention (M=64):      F1: 0.75  EM: 0.61  Compression: 8×
Convolutional (M=64):        F1: 0.72  EM: 0.58  Compression: 8×
Weighted Pooling (M=64):     F1: 0.69  EM: 0.54  Compression: 8×
Gist (M=64):                 F1: 0.77  EM: 0.63  Compression: 8×  ⭐
```

## Key Implementation Details

### Critical Fixes (from LOG.md)
1. **Teacher-forcing indexing:** Fixed off-by-one error (was `answer_start-1`, now `answer_start`)
2. **Removed broken KL distillation:** Position misalignment made it meaningless
3. **Added generation loss:** Matches eval objective (greed

y decoding)
4. **Contrastive loss:** Uses first token (CLS-like) instead of mean pooling

### Reproducibility
- Fixed seeds (42) for all randomness
- Deterministic CUDA operations
- Gradient clipping (max_norm=1.0)
- Dynamic padding (saves memory vs. global max padding)

## Output Structure

Results saved to:
```
runs/compression_ablations/
├── M32_cross_attention/
│   ├── checkpoint.pt
│   ├── config.json
│   ├── metrics.json
│   └── sample_predictions.json
├── M64_gist/
│   └── ...
└── results_summary.json
```

## Future Directions

1. **Hierarchical Compression:** Combine multiple architectures (gist + conv)
2. **Multi-Task Training:** Train on multiple datasets (SQuAD + HotpotQA + GSM8K)
3. **Adaptive Compression:** Learn when to compress vs. keep full context
4. **Cross-Model Compression:** Compress in one model, decode in another
5. **Quantization:** Further reduce wire cost via int8/int4 quantization

## References

1. Mu et al. "Learning to Compress Prompts with Gist Tokens" NeurIPS 2023. [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)
2. Jaegle et al. "Perceiver: General Perception with Iterative Attention" ICML 2021.
3. van den Oord et al. "WaveNet: A Generative Model for Raw Audio" 2016.
4. Vaswani et al. "Attention is All You Need" NeurIPS 2017.

## Contact

For questions or issues, see main repository README.
