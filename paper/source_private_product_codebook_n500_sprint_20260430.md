# Source-Private Product-Codebook n500 Sprint

Date: 2026-04-30

## Status

Current paper readiness: stronger than before this sprint, but still not
comfortable ICLR full-paper ready. COLM workshop readiness is strong.

Current story: source-private residual communication with decoder side
information. The target has public prompt/candidate state; the source sends a
rate-capped private packet; gains must vanish when the source is destroyed.

Exact blocker: the frozen verifier row is reliable but protocol-shaped, and
the failed anchor-relative/crosscoder receiver means we still need a less
table-shaped, systems-aware method contribution.

## Lay Explanation

This experiment asks whether the source can send a tiny compressed correction,
not a full private log. Product-codebook packets split the source-side residual
vector into subspaces and send one learned centroid index per subspace. The
target already has public candidate information and checks which candidate is
closest to the reconstructed correction.

## Method

The n500 gate uses the existing product-codebook/PQ packet path:

- train split: `768`
- eval split: `500`
- feature dimension: `512`
- candidate view: `slot`
- remap seeds: `101`, `103`, `107`
- budget: `4` bytes
- source packet: product-codebook indices
- baselines: scalar Wyner-Ziv, protected residual, QJL residual,
  rotation-sign, target-only, query-aware text at budget
- controls: label-shuffled ridge, constrained shuffled source,
  answer-masked source, permuted codes, random same-byte
- metrics: accuracy, paired CI, payload bytes/tokens, p50/p95 decode latency,
  cached/table-lookup decode mismatch

Artifacts:

- `results/source_private_product_codebook_packet_gate_n500_20260430/`
- `results/source_private_product_codebook_uncertainty_n500_20260430/`
- `results/source_private_product_codebook_decode_frontier_n500_20260430/`

## Results

The functional n500 product-codebook gate passes all three remaps.

| Remap | Budget | PQ | Target | Best PQ Control | Scalar WZ | PQ-Control | PQ-Scalar | p50 Decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4B | 0.482 | 0.250 | 0.268 | 0.424 | +0.214 | +0.058 | 0.360 ms |
| 103 | 4B | 0.508 | 0.250 | 0.262 | 0.502 | +0.246 | +0.006 | 0.378 ms |
| 107 | 4B | 0.520 | 0.250 | 0.252 | 0.504 | +0.268 | +0.016 | 0.368 ms |

Paired uncertainty also passes:

- pass rows: `3/3`
- remaps with pass: `[101, 103, 107]`
- min CI95 low versus target-only: `+0.174`
- min CI95 low versus best PQ destructive control: `+0.154`
- min CI95 low versus scalar WZ: `-0.032`

Interpretation: PQ proves source-causal lift and stays near scalar WZ, but does
not strictly beat scalar on every paired comparison. The claim should be a
systems/codec tradeoff, not universal accuracy dominance.

The decode frontier passes:

- pass rows: `3/3`
- max cached receiver p50: `0.0212 ms`
- max request-public table decode p50: `0.3694 ms`
- max resident table lookup p50: `0.0177 ms`
- zero canonical/cached/table prediction mismatches
- min cached speedup versus prior recorded Python path: `17.38x`

Interpretation: once target-side public state is cached, the PQ receiver is a
fast table lookup rather than an expensive neural verifier. This gives the
paper a systems-facing method contribution distinct from the 2-byte verifier.

## Contribution Impact

This promotes product-codebook packets as the third technical contribution:

1. Strict source-private benchmark and destructive controls.
2. Frozen model-mediated 2-byte verifier receiver.
3. Compression-native product-codebook packet codec with n500 remap stability,
   paired uncertainty, and cached target-side decode frontier.
4. Hardware-aware packet accounting and source-boundary systems trace.

The claim remains scoped. This is not protocol-free cross-model latent
reasoning. It is a compact source-private residual codec with decoder side
information and explicit systems accounting.

## Remaining ICLR Gap

For a comfortable ICLR full paper, the next required evidence is:

- frozen verifier n500 or batched GPU verifier telemetry
- native NVIDIA/vLLM/KV baseline telemetry for TTFT, TPOT, goodput, and KV byte
  movement
- a less synthetic benchmark or cross-family model pair beyond the diagnostic
  surface
- PQ/TurboResidual stress with top-codeword knockout and possibly OPQ/protected
  basis variants under the same n500 pass rule

Blocker needing user help eventually: NVIDIA GPU access for production serving
and KV/TurboQuant/KIVI/C2C-style baseline telemetry. No SSH was used.
