# HBSM Progress

## 2026-05-06

Status: **NEW / wounded novelty / Mac gates pending**.

Added and ran a deterministic synthetic B1/B2 packet:

- script: `phase2/hbsm_synthetic_b1_gate.py`
- packet: `phase2/results/hbsm_synthetic_b1/`
- decision: `SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT`
- cheap-predictor Spearman rho: `0.657`
- boundary top-decile hits: `2`

Interpretation: synthetic-only artifact validation. It fixes the B1/B2 readout
format but does not promote the branch or replace real hybrid layer-sensitivity
measurements.

Next exact gate: B1 sensitivity heterogeneity replication on current hybrid
models. Kill or fold into HORN if the mechanism wedge does not differentiate.
