# Experiment: AG News 4-Class (Corrected Prompts)

**Date**: 2025-12-03
**Run ID**: agnews_20251203_215159
**Status**: COMPLETE
**Purpose**: Fair comparison with consistent prompts between bridge and baselines

---

## Executive Summary

**Bridge exceeds Mistral text baseline by 18.4 percentage points.**

| Method | Accuracy | Samples |
|--------|----------|---------|
| **Bridge** | **88.9%** | 1000 |
| Llama Text | 74.5% | 200 |
| Mistral Text | 70.5% | 200 |
| Random | 25.0% | - |
| Noise | 0.0% | 200 |

---

## Key Correction

Previous runs had a prompt inconsistency. This run uses consistent prompts:
- `"Article: {text}\nTopic (world, sports, business, or science):"`

Both bridge training and baselines use the same format with explicit class choices.

---

## Per-Class Breakdown

| Class | Bridge | Mistral Text | Llama Text | Delta (vs Mistral) |
|-------|--------|--------------|------------|---------------------|
| World | 93.3% | 77% | 79% | +16.3pp |
| Sports | 92.0% | 85% | 100% | +7.0pp |
| Business | 78.5% | 97% | 97% | -18.5pp |
| **Science** | **89.3%** | **37%** | **35%** | **+52.3pp** |

### Critical Finding: Science Category Fix

Both Llama (35%) and Mistral (37%) struggle badly with Science in text mode.
The **bridge achieves 89.3%** - a 52+ percentage point improvement!

**Hypothesis**: The bridge learns a regularized representation space where Science/Technology articles cluster distinctly, overcoming the ambiguity that confuses both models in text mode.

---

## Training Configuration

```python
source_layer = 31      # Optimal from ablation
num_latents = 8        # Information Bottleneck optimal
diversity_weight = 0.1 # Prevents mode collapse
training_steps = 3000
batch_size = 16
```

---

## Training Dynamics

- Loss: 1.7 → 0.12
- Accuracy trajectory: 31% → 87% → 94% within ~800 steps
- Stable at 87-94% range

---

## Comparison with exp004 (Pre-Correction)

| Metric | exp004 (old) | exp006 (corrected) |
|--------|--------------|-------------------|
| Bridge Accuracy | 92.3% | 88.9% |
| Mistral Baseline | 70.0% | 70.5% |
| Delta | +22.3pp | +18.4pp |

The corrected prompts slightly reduced bridge accuracy but ensured fair comparison.

---

## Files

| File | Description |
|------|-------------|
| `eval_agnews_results.json` | Full evaluation with sample outputs |
| `agnews_baselines.json` | Baseline comparison data |
| `config.json` | Training configuration |
| `train.log` | Training execution log |
| `baselines_*.log` | Baseline evaluation log |
| `eval_*.log` | Final evaluation log |

---

## Significance

1. **Fair comparison**: Prompts are now consistent
2. **Massive improvement over text**: +18.4pp vs Mistral
3. **Science category fix**: 37% → 89% is the "hero result"
4. **Validates generalization**: 4-class works with same architecture

---

## Citation

```
AG News 4-Class Classification - Corrected Prompts (2025-12-03)
- Bridge: 88.9% (exceeds Mistral text 70.5% by +18.4pp)
- Science category: 37% → 89% (+52pp improvement)
- Fair comparison: Consistent prompts for bridge and baselines
- Configuration: Layer 31 + 8 tokens
```
