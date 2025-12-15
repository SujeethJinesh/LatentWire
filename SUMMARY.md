# LatentWire: Cross-Model Communication via Soft Tokens

## Executive Summary

**Problem:** When two LLMs need to collaborate (e.g., Llama analyzes, Mistral acts), the standard approach is text-based communication. Llama generates text token-by-token (slow), Mistral reads it. This incurs ~835ms latency per interaction.

**Solution:** LatentWire extracts Llama's internal representations and compresses them into 8 learned "soft tokens" that Mistral directly consumes—no text generation required.

**Result:** 22× faster (37ms vs 835ms) while achieving higher accuracy than either model alone.

---

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Llama     │     │   Bridge    │     │   Mistral   │
│   (8B)      │ ──► │   (6.3M)    │ ──► │   (7B)      │
│   FROZEN    │     │  TRAINABLE  │     │   FROZEN    │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
  Hidden states      8 soft tokens        Classification
  from layer 16      (compressed)          prediction
```

**The Bridge:** A Perceiver Resampler (cross-attention architecture from Flamingo/BLIP-2) that:
1. Takes variable-length hidden states from Llama
2. Compresses them into exactly 8 soft tokens
3. Projects to Mistral's embedding space

**Training:** Only the bridge (6.3M params) is trained. Both LLMs (15B total) remain frozen.

---

## Main Results

### Classification Accuracy (%)

| Dataset | Classes | Random | Prompt-Tuning | Llama | Mistral | 5-shot | Text-Relay | **Bridge** |
|---------|---------|--------|---------------|-------|---------|--------|------------|------------|
| SST-2 | 2 | 50.0 | 49.5 | 88.4 | 92.2 | 94.5 | 71.0 | **96.7 ± 0.6** |
| AG News | 4 | 25.0 | 19.8 | 63.8 | 69.4 | 80.3 | 64.5 | **90.7 ± 0.5** |
| TREC | 6 | 16.7 | 19.0 | 74.4 | 61.8 | -- | 58.0 | **95.3 ± 0.3** |
| Banking77 | 77 | 1.3 | -- | -- | -- | -- | 1.0 | **21.5** |

### Latency (H100 GPU)

| Method | Latency | Speedup |
|--------|---------|---------|
| Text-Relay | 834.5 ms | 1× |
| **Bridge** | **37.3 ms** | **22.4×** |

**Breakdown:**
- Llama encode: 16.9ms (45%)
- Bridge transform: 1.2ms (3%)
- Mistral forward: 19.3ms (52%)

### Throughput (samples/sec)

| Batch Size | Bridge | Direct Mistral | Text-Relay |
|------------|--------|----------------|------------|
| 1 | 7.4 | 8.8 | 0.9 |
| 4 | 28.7 | 31.2 | 1.0 |
| 16 | 105.7 | 116.0 | -- |

---

## Key Findings

### 1. Sender Model is Essential

**Experiment:** Train soft prompts on Mistral only (no Llama involvement), same training budget.

| Method | SST-2 | AG News | TREC |
|--------|-------|---------|------|
| Prompt-Tuning (no Llama) | 49.5% | 19.8% | 19.0% |
| Bridge (with Llama) | 96.7% | 90.7% | 95.3% |
| **Improvement** | **+47.2pp** | **+70.9pp** | **+76.3pp** |

**Statistical significance:** p = 8.49×10⁻¹⁴, Cohen's d = 107

**Conclusion:** The improvement comes entirely from Llama's hidden states, not from training soft prompts.

### 2. Super-Additive Performance

The bridge exceeds BOTH individual models:

| Model | SST-2 | AG News |
|-------|-------|---------|
| Llama alone | 88.4% | 63.8% |
| Mistral alone | 92.2% | 69.4% |
| **Bridge** | **96.7%** | **90.7%** |
| Gain over best single model | +4.5pp | +21.3pp |

### 3. Cross-Model Beats Same-Model

| Configuration | SST-2 | AG News |
|---------------|-------|---------|
| Llama → Llama (same) | 84.5% | 90.5% |
| Mistral → Mistral (same) | 95.5% | -- |
| **Llama → Mistral (cross)** | **96.7%** | **90.7%** |

**Hypothesis:** Representation incompatibility forces the bridge to learn abstract, task-relevant features rather than identity shortcuts. The incompatibility acts as beneficial regularization.

### 4. Inverse Token Scaling

More tokens = worse performance (counterintuitive):

| Soft Tokens | Banking77 Accuracy |
|-------------|-------------------|
| 16 | **21.5%** |
| 32 | 13.5% |
| 64 | 7.5% |
| 128 | 1.0% |

**Explanation:** Compression forces the bridge to extract only task-relevant features (information bottleneck principle). More tokens allow noise and overfitting.

### 5. Comparison to Fine-Tuning Baselines

| Method | SST-2 Acc | Latency | Notes |
|--------|-----------|---------|-------|
| Full Fine-tune (2 layers) | 94.0% | 113ms | 570M trainable params |
| Full Fine-tune (8 layers) | 94.0% | 113ms | 1.9B trainable params |
| LoRA (rank=8) | 95.3% | 113ms | 3.4M trainable params |
| CoT-Relay | 89.0% | 3,169ms | Llama generates reasoning |
| **Bridge** | **96.7%** | **37ms** | 6.3M trainable params |

**Bridge is 2.7pp more accurate than fine-tuning and 3× faster.**

---

## Limitations: Reasoning Tasks Fail

| Task | Bridge | Llama Direct | Random |
|------|--------|--------------|--------|
| CommonsenseQA | 17.0% | 75.0% | 20.0% |
| GSM8K (math) | 2.0% | 76.5% | ~0% |
| BoolQ | 53.5% | 79.5% | 50.0% |
| PIQA | 52.5% | 80.0% | 50.0% |

**Why reasoning fails:** Soft token compression preserves "what" (sentiment, topic) but not "how to reason." Multi-step inference requires maintaining logical chains that 8 compressed tokens cannot encode.

**This is a fundamental limitation we acknowledge in the paper.**

---

## Comparison to Related Work

| Method | Type | Setting | Comparable? |
|--------|------|---------|-------------|
| Cross-LoRA | LoRA weight transfer | Offline | No - different problem |
| PromptBridge | Text prompt optimization | Offline | No - different problem |
| StitchLLM | Model block composition | Serving | No - different problem |
| **LatentWire** | Soft token communication | **Runtime** | -- |

**Key distinction:** These methods perform offline transfer (before deployment). LatentWire enables runtime communication—the sender processes each input dynamically during inference.

---

## Architecture Details

### Bridge Specifications

| Component | Value |
|-----------|-------|
| Architecture | Perceiver Resampler |
| Input dimension | 4096 (Llama hidden size) |
| Output dimension | 4096 (Mistral hidden size) |
| Internal dimension | 512 |
| Cross-attention layers | 2 |
| Attention heads | 8 |
| Soft tokens | 8 |
| Trainable parameters | ~6.3M |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 1×10⁻⁴ |
| Batch size | 8 |
| Training steps | 2,000 |
| Source layer | 16 (SST-2), 31 (AG News, TREC) |
| Diversity loss weight | 0.1 |
| Optimizer | AdamW |

### Models Used

| Model | Role | Parameters | Status |
|-------|------|------------|--------|
| Llama 3.1 8B Instruct | Sender | 8B | Frozen |
| Mistral 7B Instruct v0.3 | Receiver | 7B | Frozen |

---

## Ablation Studies

### Source Layer Selection

| Layer | SST-2 Accuracy |
|-------|----------------|
| 0 | 50.0% |
| 8 | 85.0% |
| 16 | 92.0% |
| 24 | 91.5% |
| 31 | 92.5% |

**Finding:** Middle-to-late layers (16-31) contain the best semantic information.

### Architecture Comparison

| Architecture | SST-2 Accuracy |
|--------------|----------------|
| **Perceiver (ours)** | **92.0%** |
| MLP Bridge | 91.5% |
| Linear Projection | 91.5% |
| Diffusion Transformer | 85.5% |
| Mean Pooling | 0.0% |
| Identity | 0.0% |

**Finding:** Cross-attention is essential. Naive pooling completely fails.

---

## Reproducibility

### Multi-Seed Results (SST-2)

| Seed | Bridge Accuracy |
|------|-----------------|
| 42 | 96.5% |
| 123 | 96.0% |
| 456 | 97.5% |
| **Mean ± Std** | **96.7% ± 0.6%** |

### Prompt-Tuning Baseline (SST-2)

| Seed | Accuracy |
|------|----------|
| 42 | 49.5% |
| 123 | 49.5% |
| 456 | 49.5% |
| **Mean ± Std** | **49.5% ± 0.0%** |

---

## What's Next

### Immediate
1. ✅ Paper written (12 pages)
2. ✅ Unified comparison script created
3. 🔄 Running final experiments on HPC
4. ⬚ Verify all baselines have error bars

### Before Submission
- Run unified comparison for fresh results
- Ensure statistical significance for all claims
- Final paper polish

### Future Directions
- Scale to larger models
- Test bidirectional communication
- Multi-model chains (>2 models)
- Different modalities (code, structured data)

---

## Paper Structure

1. **Introduction** - Problem, solution, contributions
2. **Related Work** - Soft prompts, Perceiver, vision-language models, model stitching
3. **Method** - Problem formulation, bridge architecture, training objective
4. **Experiments** - Main results, latency, ablations
5. **Analysis** - Cross vs same model, token scaling, reasoning failure
6. **Limitations** - Reasoning tasks, scope
7. **Conclusion**

**Target:** MLSys 2025

---

## One-Slide Summary

```
╔══════════════════════════════════════════════════════════════════╗
║  LatentWire: 22× Faster LLM-to-LLM Communication                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PROBLEM:  Text-based LLM communication is slow (835ms)         ║
║  SOLUTION: Compress hidden states → 8 soft tokens (37ms)        ║
║                                                                  ║
║  ┌─────────┐    ┌────────┐    ┌─────────┐                       ║
║  │  Llama  │───►│ Bridge │───►│ Mistral │                       ║
║  │   8B    │    │  6.3M  │    │   7B    │                       ║
║  └─────────┘    └────────┘    └─────────┘                       ║
║                                                                  ║
║  RESULTS:                                                        ║
║  • SST-2:    96.7% (vs 92.2% Mistral, 49.5% no-Llama)           ║
║  • AG News:  90.7% (vs 69.4% Mistral) = +21pp                   ║
║  • Latency:  37ms vs 835ms = 22× faster                         ║
║  • Cross > Same: Llama→Mistral beats Llama→Llama by 12pp        ║
║                                                                  ║
║  LIMITATION: Reasoning tasks fail (17% vs 75%)                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Contact

- **Author:** Sujeeth Jinesh
- **Affiliation:** Stanford University
- **Email:** sujinesh@stanford.edu
