# Experiment 002: SST-2 Baselines

**Date**: 2024-12-02
**Status**: COMPLETE
**Purpose**: Validate exp001 results by comparing against baselines

---

## Objective

Confirm that the bridge's 93.46% accuracy represents genuine semantic transfer, not an artifact of:
- Model bias (always outputting one label)
- Random chance
- Dataset imbalance

---

## Baselines Evaluated

| Baseline | Description | Accuracy |
|----------|-------------|----------|
| Random | Coin flip | 50.0% |
| Majority class | Always predict majority | 50.9% |
| Noise | Random soft tokens to Mistral | **0.0%** |
| Llama text | Llama given full review | **89.0%** |
| Mistral text | Mistral given full review | **93.5%** |

---

## Key Results

### Bridge Performance in Context

```
Random chance:     50.0%  ─┐
Majority class:    50.9%  ─┤ Lower bounds
Noise baseline:     0.0%  ─┘ (Mistral outputs garbage)

Llama text:        89.0%  ── Source model baseline

Bridge:            93.46% ── OUR RESULT
Mistral text:      93.5%  ── Upper bound
```

### Critical Findings

1. **Noise baseline = 0%**: Random soft tokens produce garbage output (not even random guessing). This proves soft tokens must carry meaningful information.

2. **Bridge matches Mistral text**: 93.46% vs 93.5% - the bridge achieves the same performance as giving Mistral the full text directly.

3. **Bridge exceeds Llama text**: 93.46% vs 89.0% - the bridge+Mistral combination outperforms Llama alone by 4.46 percentage points.

---

## Interpretation

The bridge is genuinely transmitting semantic information:
- Far above random (50%) and noise (0%)
- Matches the theoretical upper bound (Mistral text: 93.5%)
- Exceeds the source model (Llama: 89.0%)

The fact that Bridge+Mistral > Llama suggests the bridge successfully leverages Mistral's classification capabilities while using Llama's text understanding.

---

## Dataset Statistics

- Total samples: 872
- Positive: 444 (50.9%)
- Negative: 428 (49.1%)
- Dataset is balanced (majority baseline ≈ 50%)

---

## Files

| File | Description |
|------|-------------|
| `sst2_baselines.json` | Structured results |
| `baselines.log` | Full execution log |
| `EXPERIMENT_SUMMARY.md` | This document |

---

## Citation

```
Experiment 002: SST-2 Baselines
- Noise baseline: 0% (proves soft tokens carry information)
- Bridge matches Mistral text: 93.46% vs 93.5%
- Bridge exceeds Llama text: 93.46% vs 89.0%
```
