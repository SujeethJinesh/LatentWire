# Source-Private Product-Codebook Knockout Stress

Date: 2026-04-30

## Status

Current readiness: stronger than the previous n500 PQ row, but still not a
comfortable ICLR full-paper claim. COLM workshop readiness remains strong.

Current paper story: the source observes private evidence and sends a tiny
rate-capped residual packet. The target has public prompt/candidate side
information and decodes the packet as a correction, not as full trace relay.

Exact submission blocker: the PQ branch must survive reviewer concerns that a
4-byte packet is just a hidden lookup ID. This gate tests whether the decoded
candidate decision is sensitive to the source-selected codeword that contributes
most to the gold-vs-nearest-wrong margin.

Layman explanation: for each question, we find which byte in the tiny message
most helps the correct answer beat the closest wrong answer. Then we damage that
byte and check whether performance drops. If damaging the important byte hurts
more than damaging a random byte, the packet is carrying usable evidence.

## Gate

- script: `scripts/build_source_private_product_codebook_knockout_stress.py`
- test: `tests/test_build_source_private_product_codebook_knockout_stress.py`
- artifact:
  `results/source_private_product_codebook_knockout_stress_n500_20260430/`
- benchmark: n500, train 768, eval 500, feature dim 512, remaps 101/103/107,
  4-byte product-codebook packet, slot candidate view
- conditions:
  - `target_only`
  - `product_codebook_source`
  - `top_codeword_removed_worst`
  - `top_codeword_removed_mean`
  - `random_codeword_removed_mean`
  - `random_codeword_removed_random`
  - `top_codeword_only`
  - `mean_payload`

The top-codeword selector is an analysis tool, not a deployable receiver: it
uses the decoded distance margin to identify the packet byte with the largest
gold-vs-nearest-wrong contribution. The `top_codeword_removed_worst` row then
replaces that byte with the code that most damages the gold margin. The
`top_codeword_removed_mean` row replaces it with a train-public mean code.

## Results

The adversarial knockout gate passes across all three remaps, but the stricter
public-mean knockout gate does not pass.

| Remap | Source | Target | Top-worst | Top-mean | Random-mean | Mean payload | Top-worst lift removed | Top-mean lift removed | Public pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.482 | 0.250 | 0.002 | 0.458 | 0.478 | 0.220 | 2.069 | 0.103 | false |
| 103 | 0.508 | 0.250 | 0.002 | 0.456 | 0.508 | 0.264 | 1.961 | 0.202 | false |
| 107 | 0.520 | 0.250 | 0.004 | 0.482 | 0.500 | 0.234 | 1.911 | 0.141 | false |

Paired CI lower bounds for matched PQ versus top-worst knockout are positive:

- remap 101: mean delta `+0.480`, CI95 low `+0.436`
- remap 103: mean delta `+0.506`, CI95 low `+0.468`
- remap 107: mean delta `+0.516`, CI95 low `+0.474`

The public-mean replacement is much weaker:

- remap 101: mean delta `+0.024`, CI95 low `-0.006`
- remap 103: mean delta `+0.052`, CI95 low `+0.028`
- remap 107: mean delta `+0.038`, CI95 low `+0.016`

Payload entropy/collision diagnostics are also now recorded. Matched source
payloads are almost all unique at n500:

- remap 101: `500/500` unique payloads, entropy `8.97` bits
- remap 103: `499/500` unique payloads, entropy `8.96` bits
- remap 107: `498/500` unique payloads, entropy `8.96` bits

## Interpretation

Promote:

- The PQ packet is margin-sensitive: an oracle replacement of the byte with the
  largest candidate-margin contribution collapses accuracy from `0.482-0.520`
  to `0.002-0.004`.
- The row is source-bound under the prior n500 destructive controls and paired
  uncertainty gate.
- The decode path remains systems-positive from the prior decode-frontier run:
  cached/table lookup matches the canonical decoder and is sub-millisecond on
  Mac CPU.

Do not overclaim:

- The public-mean top-codeword removal does not remove enough lift to pass the
  stricter interpretability rule.
- The high payload uniqueness means reviewers can still call the 4-byte PQ
  packet a compact per-example code unless we add a harder generalization or
  protected-basis gate.
- The `top_codeword_only` row is post-hoc/oracle and should not be presented as
  a deployable low-byte packet.

## Readiness Impact

This strengthens the PQ contribution as a compression-native systems row with a
causal stress diagnostic, but it does not close the ICLR blocker by itself. The
paper should claim:

1. strict source-private communication benchmark and controls,
2. frozen target-verifier packet consumption,
3. product-codebook residual packet with n500 source-causal lift and fast cached
   decode,
4. diagnostic evidence that PQ decisions depend on source-selected codewords.

The comfortable ICLR path still needs at least one of:

- frozen verifier n500 or batched verifier evidence,
- native GPU/vLLM/KV systems telemetry,
- OPQ/protected-basis PQ variant that reduces unique-payload lookup risk,
- cross-family or less synthetic receiver generalization.

Next exact gate: run an OPQ/protected-basis product-codebook stress at n500 and
require either comparable accuracy with lower payload uniqueness/collision risk
or stronger public-mean knockout sensitivity.
