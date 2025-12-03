# Experiment 002: SST-2 Baselines

**Date**: $(date +%Y-%m-%d)
**Status**: COMPLETE
**Purpose**: Validate exp001 results by comparing against baselines

---

## Baselines Evaluated

| Baseline | Description |
|----------|-------------|
| Random | 50% (trivial lower bound) |
| Majority class | Based on SST-2 label distribution |
| Noise | Random soft tokens to Mistral |
| Mistral text | Full text to Mistral (upper bound) |
| Llama text | Full text to Llama |

---

## Results

See `sst2_baselines.json` for detailed results.

---

## Interpretation

If bridge (93.46%) is significantly above noise baseline and close to text baseline,
the bridge is genuinely transmitting semantic information.
