# HORN

HORN tests whether activation outliers propagate asymmetrically through hybrid
attention/SSM boundaries.

## Current Readiness

Status: **NEW / Mac gates pending**.

Estimated completion:

- **10%** as a positive-method paper: hypothesis and gates are scaffolded.
- **0%** as a systems-result paper: no precision allocation or native GPU
  validation exists.

## Paper Story

If attention-to-SSM and SSM-to-attention boundaries have different outlier and
noise-propagation behavior, then uniform activation precision is wasteful.
HORN asks whether one boundary direction needs higher precision than the other.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_horn_20260506.md`
- `phase1/competitor_matrix.md`

H1 measures boundary activation magnitude and kurtosis. H2 injects
FP4-equivalent noise around each boundary direction. H3 checks cross-model and
pure-architecture controls.

## Current Mac Packet

Synthetic-only packet:

- `phase2/results/horn_synthetic_h1/`
- decision: `SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT`

This validates artifact mechanics only. It is not model evidence.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_synthetic_h1 \
  --expected-decision-prefix SYNTHETIC
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Validate the first real H1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/results/horn_gate_h1_<YYYYMMDD>_<model_slug> \
  --mode real --project horn
```

## Output Paths

```text
experimental/horn/results/horn_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

## GPU Rule

No GPU validation until H1--H3 pass and a directional precision-allocation
recipe is frozen.
