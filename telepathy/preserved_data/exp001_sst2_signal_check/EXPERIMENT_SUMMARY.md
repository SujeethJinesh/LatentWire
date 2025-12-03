# Experiment 001: SST-2 Signal Check

**Date**: 2024-12-02
**Status**: SUCCESS
**Run ID**: `sst2_signal_check_20251202_212303`

---

## Objective

Validate that the Llama → Mistral bridge can transmit **any** semantic information before attempting complex tasks like GSM8K.

**Hypothesis**: If the bridge cannot transmit binary sentiment (positive/negative), it cannot transmit multi-step math reasoning.

---

## Method

### Architecture

| Component | Configuration |
|-----------|---------------|
| Source Model | Llama 3.1 8B Instruct |
| Target Model | Mistral 7B Instruct v0.3 |
| Source Layer | 16 (middle layer) |
| Bridge Type | **Continuous** (not VQ) |
| Soft Tokens | 32 |
| Perceiver Depth | 2 layers |
| Attention Heads | 8 |
| Normalization | RMS (not tanh) |

### Training

| Parameter | Value |
|-----------|-------|
| Dataset | GLUE SST-2 (Stanford Sentiment Treebank) |
| Train Samples | 33,675 (per GPU, 2 GPUs) |
| Validation Samples | 872 |
| Steps | 2,000 |
| Batch Size | 16 |
| Gradient Accumulation | 2 |
| Effective Batch | 32 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Diversity Loss Weight | 0.1 |

### Task Format

**Source (Llama reads)**:
```
Review: {text}
Sentiment:
```

**Target (Mistral generates)**:
```
Sentiment: [soft_tokens] positive/negative
```

---

## Results

### Final Evaluation (872 samples)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **93.46%** |
| Correct | 815 / 872 |
| Positive Accuracy | 91.44% |
| Negative Accuracy | 95.56% |

### Training Trajectory

| Step | Quick Eval Accuracy | LM Loss |
|------|---------------------|---------|
| 50 | - | 0.793 |
| 100 | - | 0.362 |
| 200 | 48.0% | - |
| 400 | **94.0%** | - |
| 600 | 92.0% | - |
| 800 | 52.0% (dip) | - |
| 1000 | 94.0% | - |
| 1200 | 96.0% | - |
| 1400 | 90.0% | - |
| 1600 | 96.0% | - |
| 1800 | 92.0% | - |
| 2000 | 92.0% | 0.126 |

### Key Observations

1. **Breakthrough at Step 400**: Accuracy jumped from 48% (random) to 94%
2. **Temporary dip at Step 800**: Recovered by Step 1000
3. **Stable plateau**: 90-96% from Step 1000 onward
4. **Z Variance**: Grew from 22 → 816 (healthy, not collapsed)
5. **Batch Div Loss**: ~0.99 (high but acceptable for binary task)

---

## Why This Worked (vs. Previous Failures)

### Previous Failures

| Attempt | Method | Failure Mode |
|---------|--------|--------------|
| V7 | Regression | Blurry averages, semantic drift |
| V12 | Diffusion (global) | Lost details |
| V13-14 | Diffusion (cross-attn) | Failed to converge |
| V15 (VQ) | Vector Quantization | Codebook collapse (7 times) |

### Key Changes That Enabled Success

1. **Continuous soft tokens** instead of VQ
   - No codebook = no collapse possible

2. **RMS normalization** instead of tanh
   - Preserves direction without saturation
   - `out = (x / ||x||_rms) * scale`

3. **Simpler task** (SST-2 vs GSM8K)
   - 67k training examples (vs 7.5k)
   - Binary classification (vs multi-step reasoning)
   - "Blurriness" acceptable (vs exact numbers)

4. **Batch diversity loss**
   - Penalizes high cosine similarity between batch items
   - Prevents mode collapse

---

## Significance

This is the **first successful demonstration** that:

1. Semantic information can transfer through the Llama → Mistral bridge
2. Continuous soft tokens work where discrete methods failed
3. The Perceiver architecture successfully compresses variable-length input to fixed-length latents

---

## Files in This Experiment

| File | Description |
|------|-------------|
| `config.json` | Training configuration |
| `eval_sst2_results.json` | Full evaluation results with sample predictions |
| `EXPERIMENT_SUMMARY.md` | This document |

### Source Code (at time of experiment)

- `telepathy/latent_bridge_v15.py` - Bridge architecture
- `telepathy/train_telepathy_sst2.py` - Training script
- `telepathy/eval_telepathy_sst2.py` - Evaluation script
- `run_sst2_signal_check.sh` - Run script

---

## Next Steps

1. **Run baselines** to validate result is genuine (not Mistral bias)
2. **Scale to harder tasks**: AG News (4-class), then QA
3. **Optimize architecture** if baselines confirm success

---

## Citation

If using this experiment in the paper:

```
Experiment 001: SST-2 Signal Check
- First successful semantic transfer through Llama→Mistral bridge
- 93.46% accuracy on binary sentiment classification
- Key innovations: Continuous soft tokens, RMS normalization, batch diversity loss
```
