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

H1 measures boundary activation magnitude and kurtosis. H2 injects
FP4-equivalent noise around each boundary direction. H3 checks cross-model and
pure-architecture controls.

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
