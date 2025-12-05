# Experiment: GSM8K Latent Chain-of-Thought (NEGATIVE RESULT)

**Date**: 2025-12-03
**Run ID**: gsm8k_20251203_221833
**Status**: COMPLETE - FAILED
**Purpose**: Test if latent tokens can enable mathematical reasoning

---

## Executive Summary

**The bridge completely fails at mathematical reasoning.**

| Method | Accuracy | Samples |
|--------|----------|---------|
| Llama Text | **76.5%** | 200 |
| Mistral Text | 48.5% | 200 |
| **Bridge (Latent CoT)** | **2.0%** | 500 |
| Random | 1.0% | 200 |
| Noise | 0.0% | 200 |

The bridge achieves only 2% accuracy - essentially random guessing.

---

## Architecture Attempted

"Latent Chain-of-Thought" - iterating through latent space to enable reasoning:

```
Question → Llama → 8 tokens → [Recurrent 4×] → 32 tokens → Mistral → Answer
```

The hypothesis was that iterating through latent space could enable "thinking."
**This hypothesis was wrong.**

---

## Training Dynamics

| Step Range | Accuracy |
|------------|----------|
| 0-500 | 0% |
| 500-1000 | 0% |
| 1000-2000 | 0% |
| 2000-3000 | 0-2% |
| 3000-5000 | 0-2% |

**Training never learned.** Accuracy stayed at 0% for essentially all 5000 steps.

---

## Mode Collapse Analysis

The model collapsed to outputting a small set of "round numbers":

| Gold Answers (varied) | Predicted (collapsed) |
|-----------------------|----------------------|
| 18, 3, 70000, 540, 20, 64, 260... | 10, 12, 100, 1000, 1200... |

Classic mode collapse - the model found a local minimum of outputting generic numbers.

---

## Why Classification Works But Reasoning Fails

| Aspect | Classification | Reasoning |
|--------|----------------|-----------|
| **Output entropy** | 1-2 bits (pos/neg, 1-of-4) | High (exact number) |
| **Information type** | Pattern matching | Sequential computation |
| **Error propagation** | Binary right/wrong | Compounding |
| **Latent requirements** | Feature encoding | Step-by-step manipulation |

### Theoretical Analysis

**Information-Theoretic**: Classification asks "which bucket?" (~1 bit). Math asks "compute this specific value" (many bits). The latent space can encode "which category" but cannot encode "execute these calculations."

**Computational**: Classification is a learned hash function (input → category). Math is sequential computation (input → operations → intermediate values → output). The bridge architecture is fundamentally a pattern matcher, not a calculator.

**Neuroscience Analogy**: Pattern recognition (classification) and sequential reasoning (math) use different cognitive processes. We built a pattern recognizer and expected it to compute.

---

## Key Insights

1. **The bridge architecture works for perception/classification** - proven by SST-2 and AG News
2. **The bridge architecture fails for multi-step reasoning** - proven by this experiment
3. **Latent Chain-of-Thought does not enable reasoning** - iterating latent tokens ≠ thinking
4. **This is an architectural limitation, not a training bug**

---

## Configuration

```python
source_layer = 31
cot_steps = 4              # Recurrent reasoning steps
soft_tokens_per_step = 8   # Tokens per step
total_latent_tokens = 32   # 4 × 8
training_steps = 5000
batch_size = 8
```

---

## Files

| File | Description |
|------|-------------|
| `eval_gsm8k_results.json` | Full evaluation with sample outputs |
| `gsm8k_baselines.json` | Baseline comparison data |
| `gsm8k_*.log` | Training and evaluation logs |

---

## Implications

1. **Latent spaces encode "what category" but not "how to compute"**
2. **Classification is lossy-compression-friendly; reasoning is not**
3. **Different architecture needed for reasoning tasks**

### Future Directions

For reasoning tasks, different approaches are needed:

1. **Hybrid**: Bridge for perception, text for reasoning
2. **COCONUT Curriculum**: Gradually replace text CoT with latent
3. **Calculator augmentation**: Bridge encodes problem, external tool computes
4. **Step-explicit latents**: One supervised latent block per reasoning step

---

## Citation

```
GSM8K Latent Chain-of-Thought (2025-12-03) - NEGATIVE RESULT
- Bridge: 2.0% (vs Mistral text 48.5%, Llama text 76.5%)
- Latent CoT architecture failed to learn reasoning
- Training accuracy: 0% for 5000 steps
- Diagnosis: Latent space cannot encode sequential computation
- Recommendation: Use hybrid approach or COCONUT curriculum
```
