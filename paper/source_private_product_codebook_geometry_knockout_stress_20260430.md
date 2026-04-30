# Source-Private Product-Codebook Geometry Knockout Stress

Date: 2026-04-30

## Status

Current readiness: materially stronger, but still not a comfortable ICLR
full-paper package. COLM workshop readiness remains strong.

Current paper story: source-private residual communication with decoder side
information. The source sends a rate-capped vector-code packet; the target uses
public candidate state to decode a correction. The key systems claim is
positive transfer per byte under strict source-destroying controls.

Exact submission blocker addressed here: canonical 4-byte PQ passed n500
source-causal controls, but payloads were nearly unique and public-mean
top-codeword knockout was weak. Reviewers could call canonical PQ a compact
example-ID channel. This gate tests whether OPQ/protected bases mitigate that
lookup-risk without losing source-causal lift.

Layman explanation: the earlier 4-byte message worked, but almost every example
got a unique 4-byte code. Here we scramble/rotate the coordinate system before
building the codebook. If the method still works while more examples share
codes, that is stronger evidence that the code is a reusable communication
scheme rather than a private ID tag.

## Gate

- script:
  `scripts/build_source_private_product_codebook_geometry_knockout_stress.py`
- updated geometry helpers:
  `scripts/build_source_private_product_codebook_geometry_gate.py`
- tests:
  `tests/test_build_source_private_product_codebook_geometry_knockout_stress.py`
  and `tests/test_build_source_private_product_codebook_geometry_gate.py`
- artifact:
  `results/source_private_product_codebook_geometry_knockout_stress_n500_20260430/`
- setup: n500 eval, 768 train examples, 4-byte packets, remaps
  `101/103/107`, `feature_dim=512`, slot candidate view
- variants:
  - canonical contiguous PQ
  - utility-balanced PQ
  - OPQ-Procrustes
  - utility-initialized OPQ-Procrustes
  - protected Hadamard PQ
  - utility-protected Hadamard PQ

The protected Hadamard variants use sign/permutation plus normalized Hadamard
rotation before PQ. This is intentionally hardware-friendly: it resembles the
structured rotations used in low-bit inference, and can be implemented with
sign flips, permutations, and fast Walsh-Hadamard transforms rather than a dense
matrix multiply.

## Pass Rule

A noncanonical geometry variant must:

- pass the source-private controls,
- keep source accuracy within `0.02` of canonical PQ at the same remap/budget,
- and either improve public-mean top-codeword lift removal by at least `0.05`
  or reduce unique matched payloads by at least `25` at n500 while the reused
  payload subset still beats target by at least `0.10`.

## Headline Results

The gate passes: `11/15` noncanonical rows pass mitigation, with all three
remaps covered. All `18/18` rows pass source controls and adversarial
top-codeword knockout.

| Variant | Remaps | Source accuracy | Best control | Main mitigation |
|---|---:|---:|---:|---|
| canonical PQ | 101/103/107 | 0.482-0.520 | 0.262-0.268 | baseline remains near-unique |
| utility OPQ | 101/103/107 | 0.480-0.514 | 0.258-0.268 | public-mean knockout removes 1.49-1.60x lift |
| protected Hadamard | 101/103/107 | 0.498-0.514 | 0.252-0.268 | unique payloads drop by 73-95 |
| utility-protected Hadamard | 101/103/107 | 0.504-0.516 | 0.262-0.268 | unique payloads drop by 93-114 |

Canonical PQ unique payloads:

- remap 101: `500/500`
- remap 103: `499/500`
- remap 107: `498/500`

Protected Hadamard unique payloads:

- remap 101: `412/500`, collision subset accuracy `0.527`
- remap 103: `404/500`, collision subset accuracy `0.537`
- remap 107: `425/500`, collision subset accuracy `0.457`

Utility-protected Hadamard unique payloads:

- remap 101: `386/500`, collision subset accuracy `0.513`
- remap 103: `394/500`, collision subset accuracy `0.497`
- remap 107: `405/500`, collision subset accuracy `0.494`

Utility OPQ does not reduce uniqueness, but it changes the knockout behavior:
public-mean top-codeword replacement removes `1.485-1.600x` of matched lift
across all remaps while preserving accuracy within `0.006` of canonical.

## Interpretation

Promote:

- Geometry-mitigated PQ is now a distinct technical contribution over canonical
  PQ. It preserves the n500 source-private control pass while addressing the
  lookup-ID objection in two complementary ways.
- Utility OPQ makes the top-margin byte much more public-mean sensitive,
  supporting the claim that geometry changes the distribution of useful
  source evidence across codewords.
- Protected Hadamard substantially reduces singleton payload behavior while
  keeping reused-payload accuracy far above target-only.
- The protected Hadamard design has a systems angle: structured sign/permutation
  and Hadamard transforms are more accelerator-friendly than arbitrary dense
  rotations.

Do not overclaim:

- Utility OPQ still has near-unique payloads, so its claim is public-mean
  sensitivity, not collision robustness.
- Protected Hadamard does not pass public-mean knockout; its claim is lower
  uniqueness with preserved source-causal lift.
- This remains a source-private residual-code method on a controlled task, not
  broad protocol-free cross-model latent reasoning.

## Readiness Impact

This strengthens the third technical contribution substantially:

1. strict source-private communication benchmark and controls,
2. frozen target-verifier packet consumption,
3. geometry-mitigated product-codebook residual packets with n500 source-causal
   lift, cached decode, byte-causality diagnostics, and reduced lookup risk.

Comfortable ICLR still needs at least one of:

- frozen verifier n500 or batched verifier evidence,
- native GPU/vLLM/KV TTFT/TPOT/goodput telemetry,
- a less synthetic benchmark or broader cross-family receiver,
- a competitor table against C2C/KVComm/prompt-compression systems baselines
  with explicit bytes and exposed-state accounting.

Next exact gate: build a compact systems comparison table for canonical PQ,
utility OPQ, protected Hadamard, scalar WZ, text relay, KV baselines, and
verifier packets, including bytes, cached decode, public-state assumptions, and
what private state is exposed.
