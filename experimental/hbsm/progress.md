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

## 2026-05-06 Architecture Provenance Update

Added shared config-derived architecture maps at
`../shared/results/hybrid_architecture_maps_20260506/`. Real B1 packets must use
these explicit boundary IDs for boundary-flagged layers and include
`architecture_map_hash`; cheap predictors will be compared against these fixed
flags plus random/layer-index/size/norm baselines.

## 2026-05-06 Model Eligibility Update

Added metadata-only model eligibility at
`../shared/results/hybrid_model_eligibility_20260506/`. HBSM can now name the
frontier-hybrid targets and their approximate weight sizes, but it cannot run B1
without loaded model weights and forward sensitivity measurements.

## 2026-05-06 Packet Builder Update

`../shared/hybrid_trace_packet_builder.py` now supports `--project hbsm` from a
JSON row packet. Real B1 packets must include the perturbation-off no-op plus
random, layer-index, parameter-count/norm, and boundary-only controls before the
checker will accept them.

The checker also requires both `boundary_flag=true` and `boundary_flag=false`,
finite sensitivity/predictor/size/norm fields, and near-zero drift for
`perturbation_off` rows. This keeps HBSM's first real packet tied to a genuine
sensitivity table rather than a control-only schema artifact.
It now also requires train/test split coverage and matched counts for
`top_decile_flag=true` and `random_top_decile=true`.

## 2026-05-06 Reviewer Pack Update

Added `paper/reviewer_pack.md` and wired the stricter B1 packet blocker into the
COLM shell. The paper now states that HBSM is not camera-ready as a standalone
paper until B1--B3 separate the mechanism from existing sensitivity tools.

## 2026-05-06 Decision-Grade Packet Hardening

Tightened the real B1 contract after COLM-style review. A real HBSM packet now
must include hash-shaped prompt and architecture provenance, aggregate
sensitivity/enrichment fields in `summary.json`, matched top-decile/random
counts, train/test counts, and the existing no-op plus baseline controls.
Resource-limited runs are diagnostic only and must use
`RESOURCE_LIMITED_NOT_PROMOTABLE`.

Decision: **B1 PROMOTION NOW REQUIRES A COMPLETE REAL SENSITIVITY TABLE**. The
next exact gate remains a live layer-sensitivity sweep on current hybrid
reasoners.
