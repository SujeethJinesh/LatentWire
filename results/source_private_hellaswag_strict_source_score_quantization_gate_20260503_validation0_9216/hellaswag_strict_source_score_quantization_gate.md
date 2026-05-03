# HellaSwag Strict Source-Score Quantization Gate

- positive method pass: `False`
- reviewer-control audit complete: `True`
- eval rows: `9216`
- candidate-only accuracy: `0.525499`
- best score-quantized variant: `source_argmax_1b`
- best score-quantized accuracy: `0.479384`
- best minus candidate-only: `-0.046115`
- best CI95 low vs candidate-only: `-0.052083`
- best packet bytes: `1B` raw / `4B` framed

## Interpretation

Train-calibrated source-score quantization does not beat the strict candidate-only packet on the 9216-row HellaSwag surface. This weakens the source-score branch as an ICLR-positive method but closes a reviewer gap: explicit score-vector and rank/margin code baselines at matched and larger byte budgets were tested against the current packet.

## Lay Explanation

We tried sending compressed versions of the source model's four answer scores instead of only its chosen answer. A small decoder was trained on HellaSwag train rows, then frozen. On the large validation surface, these score codes still did not beat the tiny candidate-only hint.

