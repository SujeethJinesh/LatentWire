# Conditional PQ Corruption-To-Noop Receiver Gate

Date: `2026-05-04`

## Status

- COLM_v2: still plausible as a scoped source-private packet paper, but this
  branch is not the cross-family positive result.
- ICLR: still blocked by lack of a positive held-out-family method that beats
  destructive controls with paired uncertainty.
- Current story: conditional PQ has strong same-family byte-scale evidence,
  but held-out-family transfer is failing because corrupted packet-like inputs
  remain competitive unless the receiver collapses to no-op.

## Method

Implemented:

- `scripts/build_source_private_conditional_pq_corruption_noop_receiver_gate.py`
- `tests/test_build_source_private_conditional_pq_corruption_noop_receiver_gate.py`
- artifacts under
  `results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/`

The packet interface stays fixed: source-private conditional PQ payloads with
no source text, source KV, hidden vectors, raw logits, or answer key exposed to
the receiver. The change is receiver-side only. A tiny public-candidate receiver
is trained so matched packets target the answer slot and corrupted/no-source
packets target the target prior/no-op slot.

Controls include target-only, label-shuffled encoder, constrained wrong-row
source, same-answer-slot wrong-row source, answer-masked source,
public-condition-only, permuted codes, random same-byte packet, deranged public
basis, candidate roll, and opaque-slot basis.

## Commands

The main bidirectional gate used `n=256`, `train=768`, semantic basis,
`public_zscore`, `4` payload bytes, and `utility_protected_hadamard`.
I swept receiver no-op weights `0.10`, `0.05`, and `0.01`.

Example:

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/build_source_private_conditional_pq_corruption_noop_receiver_gate.py \
  --output-dir results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/core_to_holdout_semantic_public_zscore_n256_w01 \
  --train-examples 768 --eval-examples 256 \
  --train-start-index 10000 --eval-start-index 0 \
  --train-seed 30 --eval-seed 29 \
  --train-family-set core --eval-family-set holdout \
  --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --basis-view semantic \
  --source-topk 64 --target-topk 32 \
  --conditioning-mode public_zscore \
  --budget-bytes 4 --variant utility_protected_hadamard \
  --remap-slot-seed 101 --seed 30 \
  --receiver-noop-weight 0.1 \
  --bootstrap-samples 1000
```

## Results

All gates fail.

| Direction | No-op weight | Source | Target | Best control | Best control acc. | CI95 low vs best |
|---|---:|---:|---:|---|---:|---:|
| core -> holdout | `0.10` | `0.250000` | `0.250000` | label_shuffled_encoder | `0.250000` | `0.000000` |
| holdout -> core | `0.10` | `0.250000` | `0.250000` | label_shuffled_encoder | `0.250000` | `0.000000` |
| core -> holdout | `0.05` | `0.246094` | `0.250000` | permuted_codes | `0.304688` | `-0.089844` |
| holdout -> core | `0.05` | `0.253906` | `0.250000` | permuted_codes | `0.253906` | `-0.011719` |
| core -> holdout | `0.01` | `0.285156` | `0.250000` | random_same_byte | `0.367188` | `-0.148535` |
| holdout -> core | `0.01` | `0.300781` | `0.250000` | label_shuffled_encoder | `0.351562` | `-0.105566` |

Training diagnostics explain the failure:

- at `w=0.10`, the receiver predicts the prior for almost every train row, so
  all eval conditions collapse to target-only;
- at `w=0.05`, the receiver still mostly no-ops and source lift disappears;
- at `w=0.01`, matched source lift partially returns, but random same-byte,
  permuted-code, and label-shuffled controls return too.

The diagnostic headroom remains asymmetric:

| Direction | Unquantized source prediction | Target innovation oracle |
|---|---:|---:|
| core -> holdout | `0.500000` | `1.000000` |
| holdout -> core | `0.253906` | `1.000000` |

So the blocker is not just receiver corruption handling. In one direction
there is source signal before quantization/receiver decoding, but the byte
packet and receiver do not preserve it cleanly. In the other direction the
train-fitted source encoder itself has no held-out signal.

## Decision

Demote corruption-to-no-op conditional PQ as the next cross-family positive
branch. Keep the implementation because it is a reusable diagnostic for any
future packet family: it cleanly distinguishes "controls are high" from "the
receiver suppresses all packets."

Promote the next branch away from this exact held-out-family conditional-PQ
surface. The current highest-value Mac-local gate is the HellaSwag
protected-top-2/rival packet branch identified by the oracle decomposition:
there is large source top-2 headroom, but shallow switchers failed. The next
method should transmit a compact rival/frontier code and test it against
source-row shuffle, source-score shuffle, code permutation, candidate roll,
target-derived code, label permutation, and same-byte random controls.

## Lay Explanation

We taught the receiver: "trust the real tiny message, but ignore fake tiny
messages." If the fake-message penalty was strong, the receiver ignored
everything. If the penalty was weak, fake messages became useful again. That
means this receiver is not separating real source evidence from packet-shaped
noise, so this branch is not strong enough for ICLR.
