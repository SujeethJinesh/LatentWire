# Anchor-Relative Sparse Packet Gate

- date: `2026-04-29`
- artifact: `results/anchor_relative_sparse_packet_gate_20260429_smoke/`
- script: `scripts/build_anchor_relative_sparse_packet_gate.py`
- test: `tests/test_build_anchor_relative_sparse_packet_gate.py`
- scale rung: cross-family smoke / falsification

## Purpose

This gate tests a genuinely different packet family from scalar Wyner-Ziv and
canonical RASP. The source sends sparse candidate-anchor atoms: `(candidate
anchor id, quantized score)` pairs under a fixed byte budget. The target decodes
by applying those sparse atoms to its public candidate anchors. This is intended
to test whether an anchor-relative, sparse-dictionary style interface can solve
the bidirectional cross-family failure that scalar WZ and canonical RASP do not.

## Controls

The gate includes:

- target-only;
- matched sparse anchor packet;
- constrained shuffled-source packet;
- answer-masked source packet;
- random valid same-byte sparse packet;
- target-derived prior sidecar;
- anchor-id permutation.

Pass requires one budget per direction to beat target by at least `+0.10`, beat
the best source-destroying control by at least `+0.05`, and keep every control
within `target + 0.03`.

## Result

The gate fails bidirectionally.

| Direction | Budget | Sparse | Target | Best control | Pass |
|---|---:|---:|---:|---:|---:|
| core -> holdout | 2 | 0.242 | 0.250 | 0.453 | false |
| core -> holdout | 4 | 0.250 | 0.250 | 0.271 | false |
| core -> holdout | 6 | 0.125 | 0.250 | 0.283 | false |
| core -> holdout | 8 | 0.250 | 0.250 | 0.375 | false |
| holdout -> core | 2 | 0.496 | 0.250 | 0.250 | true |
| holdout -> core | 4 | 0.248 | 0.250 | 0.250 | false |
| holdout -> core | 6 | 0.270 | 0.250 | 0.250 | false |
| holdout -> core | 8 | 0.373 | 0.250 | 0.262 | true |

The result repeats the same asymmetry pattern as scalar WZ and canonical RASP:
holdout-to-core has clean positive rows, but core-to-holdout does not beat the
target prior and some anchor controls dominate the matched packet.

## Interpretation

Static anchor-relative sparse packets are not enough to claim bidirectional
cross-family communication on the current source-private surface. The failure is
useful because it rules out a shallow "relative coordinates fix everything"
story. The next cross-family mechanism should add a learned receiver or
target-preserving query bottleneck, or move to a richer source surface where
dense relative oracles first show headroom.

## Next Gate

Do not tune this exact static AR-SIP family for another cycle. The next
highest-value options are:

1. endpoint TTFT/E2E systems frontier for the existing positive packet; or
2. learned query-bottleneck / Q-Former-style receiver with zero-init target
   preservation, using AR-SIP as an interpretable sparse sidecar baseline.
