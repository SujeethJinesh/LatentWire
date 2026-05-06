# HBSM

HBSM tests whether layer sensitivity in current hybrid reasoners can be
explained and predicted more cheaply than full forward-pass KL sweeps.

## Current Readiness

Status: **NEW / wounded novelty / Mac gates pending**.

Estimated completion:

- **10%** as a narrow mechanism paper: hypothesis and gates are scaffolded.
- **0%** as a broad sensitivity-discovery paper because recent KL Lens-style
  work narrows that novelty.

## Paper Story

HBSM is not a generic forward-only sensitivity paper. Its remaining wedge is a
mechanistic account for frontier hybrid reasoners, FP4-specific sensitivity,
and cheaper predictors based on weight/statistics rather than repeated
quantized forward passes.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_hbsm_20260506.md`
- `phase1/competitor_matrix.md`

B1 replicates sensitivity heterogeneity on current hybrids. B2 tests
no-forward-pass predictors. B3 tests the softmax-amplification mechanism.

## Current Mac Packet

Synthetic-only packet:

- `phase2/results/hbsm_synthetic_b1/`
- decision: `SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT`

This validates artifact mechanics only. It is not model evidence.

## Output Paths

```text
experimental/hbsm/results/hbsm_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## GPU Rule

No GPU validation until B1--B3 pass. If HORN passes and HBSM is redundant, fold
HBSM into HORN instead of keeping a separate paper.
