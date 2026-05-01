# Source-Private ARC-Challenge Shared-Basis Endpoint, 2026-05-01

## Status

- validation artifact:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- test artifact:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/`
- seed-stability artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/`
- anchor-relative follow-up:
  `paper/source_private_arc_challenge_anchor_relative_basis_20260501.md`
- source-latent diagnostic artifact:
  `results/source_private_arc_challenge_source_latent_endpoint_gate_20260501_qwen05_bge_validation/`
- systems trace artifact:
  `results/source_private_arc_challenge_systems_trace_20260501/`
- code:
  `scripts/run_source_private_arc_challenge_fixed_packet_gate.py`,
  `scripts/build_source_private_arc_challenge_seed_stability.py`, and
  `scripts/run_source_private_arc_challenge_source_latent_endpoint_gate.py`
- references:
  `references/570_arc_challenge_common_basis_endpoint_refs_20260501.md`

## Method

The source is local Qwen2.5-0.5B choice-text log-likelihood. Unlike the earlier
BGE bridge, the packet basis is now a public hashed text basis that both source
and receiver can compute from the question and candidate choices. The source
selects a candidate without answer fields, builds a fixed `12B` sparse
random-projection packet in the shared hashed basis, and the receiver decodes
against public candidate-side hashed residuals.

This directly addresses the common-basis problem: the earlier BGE packet was a
useful bridge, but the source did not naturally emit BGE residuals. The hashed
basis is a deliberately simple shared coordinate system.

## Results

| Split | N | Target | Matched 12B | Shuffled | Same-byte text | Derangement | CI95 low vs target |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 299 | 0.244 | 0.388 | 0.251 | 0.348 | 0.197 | 0.070 |
| test | 1172 | 0.265 | 0.344 | 0.265 | 0.311 | 0.215 | 0.044 |

Projection-seed stability also passes:

| Split | Seeds | Pass | Matched mean/min/max | Min lift vs target | Min lift vs text | Min CI95 low |
|---|---:|---:|---:|---:|---:|---:|
| validation | 5 | 5/5 | 0.388 / 0.388 / 0.388 | 0.144 | 0.040 | 0.070 |
| test | 5 | 5/5 | 0.344 / 0.343 / 0.344 | 0.078 | 0.032 | 0.038 |

Mac-local systems trace:

- source scoring on official test: `251.0 ms/question`;
- receiver sparse decode p50/p95: `31.7/104.3 us`;
- payload/framed record: `12B/15B`;
- single-request cacheline/DMA: `64B/128B`;
- batch-64 line/DMA per request: `15.0B/16.0B`;
- peak process RSS: `7261.3 MiB`.

## Negative Endpoint Diagnostic

The stricter Qwen-hidden-to-BGE ridge endpoint does not pass validation:
matched/target/text is `0.281/0.244/0.348`, and the CI95 lower bound versus
target is `-0.035`. This weakens the naive hidden-state alignment branch and
motivates either a denoising receiver or stronger common-basis learning before
promoting a hidden-state endpoint.

## Interpretation

This is now a clean ARC public contribution: a source-computable fixed-byte
packet in an agreed public basis, not a receiver-only BGE packet. The
anchor-relative follow-up further strengthens the common-basis story by showing
the result survives a public coordinate chart over train anchors. It still does
not close the full ICLR latent endpoint claim because the source decision is
Qwen choice log-likelihood, not a learned hidden-state communication policy.

## Next Gate

Run either:

- a second public benchmark with the shared-basis endpoint, or
- a stronger hidden-state endpoint with OT/Procrustes/denoising receiver
  regularization, then native NVIDIA/vLLM systems rows. The Mac-local ARC
  systems trace is now present, but TTFT/TPOT/goodput/HBM native rows remain
  pending.
