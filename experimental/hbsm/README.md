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

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1 \
  --expected-decision-prefix SYNTHETIC
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit boundary IDs and architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/` for boundary-flagged
layer definitions.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.
Required real controls are `perturbation_off`, `random_flags`, `layer_index`,
`parameter_count_norm`, and `boundary_only`. Real rows must also include
`top_decile_flag`, `random_top_decile`, and `train_test_split`, with matched
top-decile/random counts and both train/test split rows unless the packet
records a resource-limit note. Convert saved B1 sensitivity rows with
`experimental.shared.hybrid_trace_packet_builder --project hbsm --row-packet
...` before validation.

Validate the first real B1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_gate_b1_<YYYYMMDD>_<model_slug> \
  --mode real --project hbsm
```

## Output Paths

```text
experimental/hbsm/phase2/results/hbsm_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

Reproduce the current synthetic packet:

```bash
./venv_arm64/bin/python -m experimental.hbsm.phase2.hbsm_synthetic_b1_gate
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1 \
  --expected-decision-prefix SYNTHETIC
jq '.decision, .spearman_rho_kurtosis_vs_sensitivity, .boundary_top_decile_hits' \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1/summary.json
```

Expected decision: `SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT`.

## GPU Rule

No GPU validation until B1--B3 pass. If HORN passes and HBSM is redundant, fold
HBSM into HORN instead of keeping a separate paper.
