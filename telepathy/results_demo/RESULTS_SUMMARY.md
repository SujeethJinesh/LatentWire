# TELEPATHY BRIDGE - RESULTS SUMMARY

Generated: 2026-01-04 20:46:16

================================================================================

## Executive Summary

**Best Configuration**: bridge__sst2_default
**Best Accuracy**: 96.7%

## Execution Gate Decisions


### Gate1 Sender Necessary
- **Status**: ❌ FAILED
- **Description**: Bridge must significantly outperform prompt-tuning baseline
- **Recommendation**: INVESTIGATE

### Gate2 Cross Model Transfer
- **Status**: ✅ PASSED
- **Description**: Cross-model transfer achieves >80% accuracy on at least one dataset
- **Recommendation**: PROCEED

### Gate3 Compression Achieved
- **Status**: ❌ FAILED
- **Description**: Average compression ratio >= 4x
- **Recommendation**: OPTIMIZE
- **Current Value**: 2.12x

### Gate4 Latency Improved
- **Status**: ✅ PASSED
- **Description**: Average latency reduction > 20%
- **Recommendation**: PROCEED
- **Current Improvement**: 31.0%

## Main Results

| Method | SST-2 | AG News | TREC | GSM8K | Avg |
|--------|-------|---------|------|-------|-----|
| Telepathy Bridge | 96.7±0.6 | 90.7±0.5 | 95.3±0.2 | - | 94.2 |
| Prompt-Tuning | 49.5±0.0 | 19.8±7.5 | - | - | 34.7 |
| LoRA | 92.0±0.4 | - | - | - | 92.0 |
| Linear Probe | 84.5±0.4 | - | - | - | 84.5 |
| LLMLingua | - | - | - | - | - |
| Same-Model | - | - | - | - | - |
| Zero-Shot | 88.0 | 75.0 | 82.0 | - | 81.7 |
| Few-Shot (3) | 91.0 | 82.0 | 88.0 | - | 87.0 |

## Statistical Significance Tests


### Bridge vs Linear Probe_sst2
- Mean 1: 96.7% ± 0.0%
- Mean 2: 84.5% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Bridge vs LoRA_sst2
- Mean 1: 96.7% ± 0.0%
- Mean 2: 92.0% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Bridge vs Prompt-Tuning_agnews
- Mean 1: 90.7% ± 0.0%
- Mean 2: 19.8% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Bridge vs Prompt-Tuning_sst2
- Mean 1: 96.7% ± 0.0%
- Mean 2: 49.5% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Few-Shot vs Zero-Shot_agnews
- Mean 1: 82.0% ± 0.0%
- Mean 2: 75.0% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Few-Shot vs Zero-Shot_sst2
- Mean 1: 91.0% ± 0.0%
- Mean 2: 88.0% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

### Few-Shot vs Zero-Shot_trec
- Mean 1: 88.0% ± 0.0%
- Mean 2: 82.0% ± 0.0%
- t-statistic: nan
- p-value: nan
- **Significance**: n.s.

## Recommendations

⚠️ **Some execution gates require attention**

Priority improvements needed:
1. **Sender Model Impact**: Bridge not significantly better than prompt-tuning
   - Investigate encoder architecture
   - Increase training data diversity
3. **Compression Ratio**: Below 4x target
   - Reduce latent sequence length
   - Implement quantization techniques