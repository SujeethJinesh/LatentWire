# Masked-Consistency Balanced Diag-Only Public-Separation Gate

- date: `2026-04-30`
- gate: `source_private_masked_consistency_public_separation`
- artifacts: `results/source_private_masked_consistency_diag_only_public_gate_20260430/`
- status: fail; prune current masked-consistency receiver as a headline ICLR contribution on the balanced surface

## Cycle Start

1. Current ICLR readiness and distance: strong scoped positive-method evidence,
   not yet comfortable broad latent-transfer ICLR. The distance is one less
   protocol-shaped learned receiver or native serving systems evidence.
2. Current story: a source-private packet can communicate hidden diagnostic
   evidence to a target with public candidate side information; the frozen
   Qwen binary verifier proves model-mediated consumption under strict controls.
3. Exact blocker: the strongest receiver still uses an explicit public decoder
   table, while the learned receiver must survive balanced public-only controls.
4. Current live branch: learned masked-consistency receiver over source-private
   syndrome bytes.
5. Highest-priority gate: n500 balanced `diag_only` public-separation.
6. Scale-up rung: medium confirmation / pruning gate.

## Layman Version

The hand-coded packet method can send a tiny clue that exactly names the right
diagnostic handle. This test asks whether the learned receiver can discover that
same trick without a hand-written equality decoder, while a public-only model
that never sees the clue still fails.

It does not. The public-only model stays below target, so the task is clean, but
the learned receiver only moves from `0.250` target accuracy to about
`0.30-0.34`, far below the `0.400` promotion threshold.

## Commands

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval29_train30_diag_only_packet \
  --train-examples 512 --eval-examples 500 --train-seed 30 --eval-seed 29 \
  --train-start-index 10000 --eval-start-index 0 --train-family-set all \
  --eval-family-set all --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --budget-bytes 6 --seed 30 --candidate-view diag_only \
  --no-require-pass

env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval31_train32_diag_only_packet \
  --train-examples 512 --eval-examples 500 --train-seed 32 --eval-seed 31 \
  --train-start-index 10000 --eval-start-index 0 --train-family-set all \
  --eval-family-set all --diagnostic-table-mode plausible_decoys \
  --feature-dim 512 --budget-bytes 6 --seed 32 --candidate-view diag_only \
  --no-require-pass

./venv_arm64/bin/python scripts/summarize_source_private_masked_consistency_public_gate.py \
  --packet-run-dir results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval29_train30_diag_only_packet \
  --public-run-dir results/source_private_diag_only_public_ablation_20260430/n500_seed29_diag_only_public_same_eval \
  --packet-run-dir results/source_private_masked_consistency_diag_only_public_gate_20260430/n500_eval31_train32_diag_only_packet \
  --public-run-dir results/source_private_diag_only_public_ablation_20260430/n500_seed31_diag_only_public_same_eval \
  --output-dir results/source_private_masked_consistency_diag_only_public_gate_20260430/summary
```

The first attempted run omitted `OPENBLAS_NUM_THREADS=1`; sampling showed the
processes were spending almost all time in many tiny OpenBLAS matrix-vector
calls. I killed those over-threaded jobs and reran with one BLAS thread, which
reduced each n500 run to about five seconds on the Mac.

## Results

Aggregate artifact:
`results/source_private_masked_consistency_diag_only_public_gate_20260430/summary/`

Headline:

- pass gate: `False`
- rows: `2`
- passed rows: `0`
- all same eval hash: `True`
- all train/eval disjoint: `True`
- all public-only rows near/below target: `True`
- max public-only lift over target: `-0.072`
- min learned lift over target: `+0.052`
- min learned lift over best destructive control: `+0.052`

Rows:

| Eval seed | Train seed | N | Learned | Hamming | Target | Best control | Public-only | Learned lift | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 29 | 30 | 500 | 0.336 | 0.350 | 0.250 | 0.252 | 0.178 | +0.086 | `False` |
| 31 | 32 | 500 | 0.302 | 0.214 | 0.250 | 0.250 | 0.142 | +0.052 | `False` |

Diagnostic stress:

- budget `16` on seed 29 improves learned accuracy only to `0.366`, still below
  the `0.400` threshold;
- `--fit-intercept` at budget `6` reaches `0.346`, also below threshold;
- oracle packet decoding is `1.000`, so the receiver can use the interface when
  the packet is correct;
- matched source packets are not equal to oracle packets, with mean bit
  distance about `0.243-0.273` on the two n500 rows.

## Interpretation

This is a useful pruning result, not a paper claim.

The balanced `diag_only` surface is clean: public-only classifiers stay below
target and source-destroying controls remain near target for the learned
receiver. The failure is in the learned source encoder/interface. It does not
produce a packet close enough to the correct diagnostic-handle packet. Increasing
the byte budget and adding an intercept do not close the gap.

Promotion decision:

- do not use the current masked-consistency receiver as one of the three
  headline ICLR contributions;
- keep it as a method-depth negative result showing that learned receivers need
  stronger source-control objectives than one-step ridge syndrome prediction;
- preserve the direct diagnostic packet and frozen binary verifier as the
  source-causal side-information contributions.

## Next Gate

Highest-value next method branch:

```text
source_private_packet_trace_card_v2_20260501
```

on Mac, plus a new learned receiver only if it changes the interface:

- posterior/flow-consistency receiver over candidate logits with source-control
  negatives;
- Q-Former/Perceiver-style query bottleneck over packet bits and public
  candidate features;
- discrete product-code / TurboQuant-inspired protected residual packet with a
  learned decoder and explicit public-only separation.

For ICLR, the next learned branch must beat the direct packet or provide a clear
systems/robustness advantage. More tuning of this masked-consistency ridge
encoder is low expected value.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_summarize_source_private_masked_consistency_public_gate.py
```

Outcome: `2 passed in 0.05s`.
