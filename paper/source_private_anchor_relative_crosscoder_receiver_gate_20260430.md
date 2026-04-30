# Source-Private Anchor-Relative Crosscoder Receiver Gate

Date: 2026-04-30

## Status

Current paper readiness: COLM-workshop plausible if framed as source-private
residual coding with strict controls; not comfortable as an ICLR full paper yet.

Current story: a source model has private evidence, sends a tiny rate-capped
packet, and a target model/receiver uses public prompt-side information plus
the packet to choose among candidates. The positive claim must survive
source-destroying controls and report real byte/latency accounting.

Exact blocker: the strongest positive row is still protocol-shaped. A reviewer
can argue that it is a controlled lookup/verifier protocol rather than a
learned, protocol-free cross-model communication method.

## Lay Explanation

This experiment asks whether one model can send another model a tiny private
hint. The target already sees the public question and the answer choices. The
source alone sees the hidden clue. If fake hints, shuffled hints, answer-only
hints, or public-only hints work just as well, then the method is not real
private communication.

## Gate Implemented

New harness:
`scripts/run_source_private_anchor_relative_crosscoder_receiver_gate.py`.

Focused test:
`tests/test_run_source_private_anchor_relative_crosscoder_receiver_gate.py`.

The harness evaluates a learned receiver over source packets with:

- exact ordered-ID parity across all conditions
- public question plus candidate pool as decoder side information
- candidate-pool recall fixed at `1.0`
- `target_only`, zero-source, shuffled-source, random same-byte,
  answer-only, answer-masked, public-only sidecar, target-derived sidecar,
  feature-ID permutation, top-feature knockout, matched-byte structured text,
  and full diagnostic oracle rows
- bytes, tokenized packet length, p50/p95 condition latency, paired bootstrap
  against target-only and best control

Pass rule: matched packet must beat target-only and best source-destroying
control by at least `0.15`, every source-destroying control must stay within
`target+0.02`, matched-byte structured text must not explain the gain, oracle
must be at least `0.95`, paired CI95 lower bounds must be at least `0.10`, and
exact ordered-ID parity must hold.

## Decisive n256 Results

Artifacts:

- `results/source_private_anchor_relative_crosscoder_receiver_n256_20260430/core_to_holdout_seed29/`
- `results/source_private_anchor_relative_crosscoder_receiver_n256_20260430/holdout_to_core_seed29/`

Both directions preserve exact ID parity and candidate-pool recall `1.0`.

| Direction | Budget | Pass | Matched | Target | Best Control | Text | Oracle | CI95 Low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core -> holdout | 4B | `False` | 0.277 | 0.250 | 0.266 | 0.258 | 0.641 | -0.039 |
| core -> holdout | 8B | `False` | 0.270 | 0.250 | 0.258 | 0.207 | 0.742 | -0.043 |
| holdout -> core | 4B | `False` | 0.309 | 0.250 | 0.277 | 0.281 | 0.762 | -0.012 |
| holdout -> core | 8B | `False` | 0.301 | 0.250 | 0.266 | 0.262 | 0.828 | -0.020 |

Top-feature knockout did not reduce matched accuracy in any n256 row. That is a
strong interpretability failure: the receiver is not depending on a small
private feature set in the way a useful crosscoder-style packet should.

## Cheap DFS Grid

I also ran a n128 debug grid over direct hashed, anchor-relative, and learned
anchor-relative packet features with `diag_only` and `semantic` candidate
views. No row passed. The most informative rows were:

- `learned_anchor_relative + diag_only`: oracle reached `0.969`, but matched
  was only `0.266` and shuffled-source reached `0.281`, so the interface can
  represent the answer but cannot transmit it from private source evidence.
- `hashed + diag_only`: oracle reached `1.000` and matched reached `0.320`,
  but public-only/text controls and top-feature knockout were too close.
- semantic views did not create a positive receiver; oracle was often near
  target-only except for direct hashed semantic, where matched collapsed.

Interpretation: this branch is currently an encoder/interface failure, not a
simple byte-budget failure.

## Decision

Prune the current anchor-relative/crosscoder receiver as a headline positive
method. Keep the harness as a reviewer-facing negative boundary and as the
template for the next learned receiver.

This result strengthens the paper only if we present it honestly: relative
representations and sparse crosscoders are useful inspiration, but this
implementation does not yet solve source-private cross-family communication.

## Next Exact Gate

The highest-value next gate is not wider anchor tuning. It is:

`source_private_verifier_n500_and_turboresidual_receiver_20260430`

1. Scale the existing frozen Qwen binary-verifier positive row to n500 with
   seed repeat and the same combined controls.
2. Add a TurboResidual/PQ packet branch that uses product-quantized residuals
   with decoder side information, then compare against the verifier row and
   matched-byte structured text under the same control suite.
3. Report Mac-local CPU/MPS latency if feasible, but keep production GPU/vLLM
   claims blocked until NVIDIA serving telemetry is available.

Blocker needing user help eventually: access to an NVIDIA GPU machine for
vLLM/FlashAttention/DistServe-style TTFT, TPOT, goodput, and KV-baseline
telemetry. No SSH was used for this gate.
