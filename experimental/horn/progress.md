# HORN Progress

## 2026-05-06

Status: **NEW / Mac gates pending**.

Added and ran a deterministic synthetic H1 packet:

- script: `phase2/horn_synthetic_h1_gate.py`
- packet: `phase2/results/horn_synthetic_h1/`
- decision: `SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT`
- SSM-to-attention / attention-to-SSM max ratio: `3.775`
- SSM-to-attention / attention-to-SSM kurtosis ratio: `7.139`

Interpretation: synthetic-only artifact validation. It fixes the H1 readout
format but does not promote the branch or replace real hybrid boundary dumps.

Next exact gate: H1 activation magnitude and kurtosis characterization using
shared boundary-inspection utilities.

## 2026-05-06 Architecture Provenance Update

Added shared config-derived architecture maps at
`../shared/results/hybrid_architecture_maps_20260506/`. Real H1 packets must use
these explicit boundary IDs and include `architecture_map_hash`; substring-only
module classification is no longer admissible for promotion.
