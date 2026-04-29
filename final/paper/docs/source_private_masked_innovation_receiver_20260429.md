# Source-Private Masked Innovation Receiver

- date: `2026-04-29`
- artifact: `results/source_private_masked_innovation_receiver_20260429/`
- script: `scripts/run_source_private_masked_innovation_receiver.py`
- test: `tests/test_run_source_private_masked_innovation_receiver.py`
- scale rung: smoke plus first strict cross-family falsification

## Purpose

This gate tests whether a more creative receiver can go beyond hand-coded
diagnostic packets. The source packet is built from private innovation:
`matched source features - answer-masked source features`. The receiver decodes
against anchor-relative target innovation:
`candidate representation - target-prior candidate representation`.

The design is inspired by masked latent prediction, flow/velocity residuals,
relative representations, and sparse crosscoder-style feature bottlenecks. The
gate is still synthetic/tool-trace based; it is a method discriminator, not a
paper claim by itself.

## Commands

Same-distribution smoke:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_innovation_receiver.py \
  --output-dir results/source_private_masked_innovation_receiver_20260429/smoke_all_seed3_4 \
  --train-examples 128 \
  --eval-examples 64 \
  --train-family-set all \
  --eval-family-set all \
  --feature-dim 256 \
  --anchor-count 64 \
  --source-topk 48 \
  --target-topk 24 \
  --budgets 4 8 \
  --train-seed 3 \
  --eval-seed 4 \
  --mask-repeats 1 \
  --calibration-examples 32
```

First strict cross-family direction:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_innovation_receiver.py \
  --output-dir results/source_private_masked_innovation_receiver_20260429/core_to_holdout_seed29_30 \
  --train-examples 256 \
  --eval-examples 128 \
  --train-family-set core \
  --eval-family-set holdout \
  --feature-dim 256 \
  --anchor-count 64 \
  --source-topk 48 \
  --target-topk 24 \
  --budgets 4 8 12 \
  --train-seed 29 \
  --eval-seed 30 \
  --mask-repeats 1 \
  --calibration-examples 48
```

## Results

| Surface | Budget | Pass | Matched | Target | Best destructive | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| all-to-all smoke | 4 | `true` | 0.766 | 0.250 | 0.281 | 1.000 |
| all-to-all smoke | 8 | `true` | 0.922 | 0.250 | 0.266 | 1.000 |
| core-to-holdout | 4 | `false` | 0.258 | 0.250 | 0.258 | 1.000 |
| core-to-holdout | 8 | `false` | 0.250 | 0.250 | 0.250 | 1.000 |
| core-to-holdout | 12 | `false` | 0.250 | 0.250 | 0.250 | 1.000 |
| core-to-holdout shared-text | 4 | `false` | 0.266 | 0.250 | 0.250 | 1.000 |
| core-to-holdout shared-text | 8 | `false` | 0.250 | 0.250 | 0.250 | 1.000 |

The same-distribution smoke is clean: zero-source, answer-masked,
public-only-innovation, shuffled-atoms, random same-byte, target-derived,
answer-only, and matched-byte text controls stay at or near target.

The cross-family result fails decisively. Oracle remains `1.000`, so the
anchor-relative innovation code can represent the answer. The learned
source-private innovation map does not transfer from core families to holdout
families.

Follow-up shared-text view: a stricter semantic variant uses a shared lexical
feature namespace for source evidence and candidate intent, while masking
candidate diagnostic handles. This also fails core-to-holdout: `0.266` at
`4` bytes and `0.250` at `8` bytes. Controls remain clean and oracle remains
`1.000`, so the failure is again source-innovation transfer rather than packet
capacity or target-side candidate decoding.

## Interpretation

This is not a promoted contribution yet. It is an important negative boundary:
masked sparse innovation works as a same-distribution receiver but does not
solve held-out-family communication. The result weakens the hypothesis that a
simple JEPA/flow-inspired innovation projection plus anchor-relative code is
enough. The next live variant must change the representation or supervision,
not just tune budgets.

## Next Gate

Do not run the reverse direction until a new diagnostic changes the hypothesis.
Two adjacent variants now fail for the same reason: anchor-relative and
shared-text source innovation maps both collapse to target on core-to-holdout
despite oracle headroom. The next method gate should either:

- add fold-heldout/shared-dictionary calibration with explicit feature knockout,
  or
- convert this into a true crosscoder-style shared/unique sparse feature model
  before rerunning core-to-holdout.

Promotion remains blocked until both cross-family directions pass with paired
uncertainty and clean source-destroying controls.
