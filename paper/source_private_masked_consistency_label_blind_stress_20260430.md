# Source-Private Masked Consistency Label-Blind Stress

- date: `2026-04-30`
- gate: `source_private_masked_consistency_receiver_label_blind_stress`
- artifacts: `results/source_private_masked_consistency_receiver_label_blind_20260430/`
- status: pass as an anti-lookup / public-side-information boundary test

## Question

Does the learned masked-consistency receiver still look positive only because
train/eval examples share synthetic IDs or because the candidate table exposes a
public lookup key?

Layman version: we first checked whether the receiver can use a tiny real clue
when the candidate descriptions are meaningful. Then we replaced those
descriptions with anonymous, randomly reordered candidate slots. If the method
were just memorizing slot numbers or labels, it would still work. It did not.

## Harness Changes

I added two reviewer-facing hardening controls:

- `--train-start-index` / `--eval-start-index` in the hidden-repair benchmark
  and masked-consistency receiver, so decisive train and eval rows can have
  disjoint synthetic IDs and labels.
- `--remap-slot-seed` for the masked-consistency receiver, so opaque slot-only
  candidate views can be randomly reordered per example before fitting and
  evaluation.

The summary now records per-condition exact-ID parity, train/eval ID
intersection count, and paired CI high for the opaque-collapse check.

## Commands

Disjoint full-view anchors:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_full \
  --train-examples 512 --eval-examples 256 --train-seed 29 --eval-seed 30 \
  --train-start-index 0 --eval-start-index 10000 \
  --feature-dim 512 --budget-bytes 6 --seed 29 --candidate-view full \
  --no-require-pass

./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed31_32_full \
  --train-examples 512 --eval-examples 256 --train-seed 31 --eval-seed 32 \
  --train-start-index 0 --eval-start-index 10000 \
  --feature-dim 512 --budget-bytes 6 --seed 31 --candidate-view full \
  --no-require-pass
```

Opaque remapped-slot stress:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_slot_remap901 \
  --train-examples 512 --eval-examples 256 --train-seed 29 --eval-seed 30 \
  --train-start-index 0 --eval-start-index 10000 \
  --feature-dim 512 --budget-bytes 6 --seed 29 --candidate-view slot \
  --remap-slot-seed 901 --no-require-pass

./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed31_32_slot_remap907 \
  --train-examples 512 --eval-examples 256 --train-seed 31 --eval-seed 32 \
  --train-start-index 0 --eval-start-index 10000 \
  --feature-dim 512 --budget-bytes 6 --seed 31 --candidate-view slot \
  --remap-slot-seed 907 --no-require-pass
```

Semantic diagnostic:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/disjoint_n256_seed29_30_semantic \
  --train-examples 512 --eval-examples 256 --train-seed 29 --eval-seed 30 \
  --train-start-index 0 --eval-start-index 10000 \
  --feature-dim 512 --budget-bytes 6 --seed 29 --candidate-view semantic \
  --no-require-pass

./venv_arm64/bin/python scripts/summarize_source_private_masked_consistency_label_blind_stress.py \
  --output-dir results/source_private_masked_consistency_receiver_label_blind_20260430/summary
```

## Results

Aggregate artifact:
`results/source_private_masked_consistency_receiver_label_blind_20260430/summary/`

Headline:

- pass gate: `True`
- full-view n256 anchors: `2/2` pass
- opaque slot-remap n256 rows: `2/2` collapse
- all decisive train/eval ID intersections: `0`
- all exact-ID parity: `True`
- min full-view lift vs target: `+0.664`
- max opaque slot learned lift vs target: `+0.012`
- max opaque slot Hamming lift vs target: `+0.023`
- max opaque slot paired CI95 high vs target: `+0.066`
- semantic-view diagnostic lift vs target: `+0.746`

Rows:

| View | Seed pair | Learned | Hamming | Target | Lift | Controls |
|---|---|---:|---:|---:|---:|---|
| full | 29/30 | 0.914 | 0.930 | 0.250 | +0.664 | max +0.000 |
| full | 31/32 | 0.957 | 0.988 | 0.250 | +0.707 | max +0.000 |
| semantic | 29/30 | 0.996 | 0.156 | 0.250 | +0.746 | max +0.000 |
| opaque slot remap | 29/30 | 0.234 | 0.180 | 0.250 | -0.016 | max +0.004 |
| opaque slot remap | 31/32 | 0.262 | 0.273 | 0.250 | +0.012 | max +0.016 |

## Interpretation

This closes the immediate lookup objection for the learned receiver:

- the positive full-view result survives with disjoint train/eval IDs;
- the source packet is useless when public candidate descriptions are replaced
  by anonymous remapped slots;
- destructive controls remain near target in all decisive rows.

The semantic row is especially useful. It removes the explicit
`handles_repair_diag` key and still passes, but it keeps public candidate intent
and issue text. That means the receiver can use public semantic side
information plus source-private bytes, not only an exact diagnostic-code table.

Claim boundary:

- This is not protocol-free latent transfer.
- This is not a model-agnostic proof that opaque latents are aligned.
- It is a side-information communication method: source-private bytes are
  decoded against public candidate semantics.

## Reviewer Impact

The strongest technical contribution is now:

> A learned, control-regularized source-private byte receiver that preserves
> packet utility on disjoint examples, collapses under opaque remapped candidate
> views, and remains positive when explicit diagnostic keys are removed but
> public candidate semantics remain.

This is materially stronger than the previous receiver memo because the result
is no longer tied to same-ID train/eval artifacts.

## Next Gate

Run `source_private_masked_consistency_receiver_disjoint_n500_20260430`:

- `n=500`, disjoint train/eval IDs;
- full view, semantic view, and opaque slot-remap view;
- two seeds if feasible on Mac;
- add a public-only learned receiver ablation that is trained without packet
  features, to separate public semantic solvability from packet-conditioned
  recovery.

If that passes, the remaining ICLR blocker shifts from anti-lookup to breadth:
one cross-family/model pair and production-style systems telemetry.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_masked_consistency_receiver_smoke.py \
  tests/test_summarize_source_private_masked_consistency_label_blind_stress.py
```

Outcome: `7 passed in 2.61s`.
