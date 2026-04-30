# Source-Private Masked Consistency Receiver

- date: `2026-04-30`
- gate: `source_private_masked_consistency_receiver_smoke`
- artifacts: `results/source_private_masked_consistency_receiver_smoke_20260430/`
- status: pass as a learned receiver over 6-byte syndrome packets; not yet a
  fully table-free semantic receiver

## Question

Can we replace a hand-written nearest-neighbor packet decoder with a learned
one-step receiver that uses packet bytes plus public candidate side information,
while forcing destroyed packets to fall back to the target prior?

Layman version: the source sends a tiny noisy clue. The target learns how to
use a real clue, how to tolerate a partially erased clue, and how to ignore a
fake clue.

## Method

I added:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_smoke_20260430/n64_seed29_30_budget6 \
  --train-examples 256 --eval-examples 64 --train-seed 29 --eval-seed 30 \
  --feature-dim 256 --budget-bytes 6 --seed 29 --candidate-view full \
  --no-require-pass

./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed29_30_budget6 \
  --train-examples 512 --eval-examples 256 --train-seed 29 --eval-seed 30 \
  --feature-dim 512 --budget-bytes 6 --seed 29 --candidate-view full \
  --no-require-pass

./venv_arm64/bin/python scripts/run_source_private_masked_consistency_receiver_smoke.py \
  --output-dir results/source_private_masked_consistency_receiver_smoke_20260430/n256_seed31_32_budget6 \
  --train-examples 512 --eval-examples 256 --train-seed 31 --eval-seed 32 \
  --feature-dim 512 --budget-bytes 6 --seed 31 --candidate-view full \
  --no-require-pass
```

Then aggregated:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_masked_consistency_receiver.py \
  --output-dir results/source_private_masked_consistency_receiver_smoke_20260430/summary
```

The receiver is a ridge-trained one-step candidate scorer. Training views:

- clean matched 6-byte learned syndrome packet -> gold candidate
- masked matched packet -> same gold candidate
- target-only / zero-source -> target prior
- shuffled, answer-masked, random same-byte, target-derived sidecar,
  answer-only, same-byte hidden-log text, and wrong-projection packets ->
  target prior

Inputs are public candidate features, target-prior features, packet/candidate
bit agreement, packet/candidate vector compatibility, and packet presence/mask
features. It does not receive private text, example IDs, family IDs, or answer
labels at inference.

## Results

Aggregate artifact:
`results/source_private_masked_consistency_receiver_smoke_20260430/summary/`

Headline:

- pass gate: `True`
- runs: `3`
- n256 runs: `2`
- min n256 learned matched accuracy: `0.957`
- min n256 lift vs target: `+0.707`
- min n256 lift vs best control: `+0.676`
- min n256 CI95 low vs target: `+0.652`
- min n256 CI95 low vs best control: `+0.617`
- n256 learned-minus-Hamming range: `-0.020` to `+0.016`

Rows:

| Run | n | learned | deterministic Hamming | target | best control | lift vs control | CI low vs control |
|---|---:|---:|---:|---:|---|---:|---:|
| seed 29/30 | 64 | 0.969 | 0.969 | 0.250 | shuffled 0.281 | +0.688 | +0.578 |
| seed 29/30 | 256 | 0.977 | 0.961 | 0.250 | wrong-projection 0.258 | +0.719 | +0.664 |
| seed 31/32 | 256 | 0.957 | 0.977 | 0.250 | wrong-projection 0.281 | +0.676 | +0.617 |

The key diagnostic is that deterministic Hamming leaks on some controls in the
first n256 run (`shuffled_source=0.332`, `wrong_projection=0.328`), while the
learned receiver suppresses them to near target (`0.246` and `0.258`). That is
the strongest new evidence: the learned receiver is not only copying the
nearest-neighbor decoder; it is learning a target-preserving control boundary.

## Interpretation

This promotes the learned receiver branch from smoke to medium confirmation.
It addresses a real reviewer concern: the target no longer relies solely on a
hand-written Hamming decoder. The learned receiver preserves nearly all
deterministic packet utility and can suppress destructive-control leakage.

Claim boundary: this is still a receiver over public candidate/code features.
It is stronger than a table lookup, but not yet protocol-free semantic latent
transfer. The next ICLR-strengthening step is label-blind or held-out
candidate-feature stress for this learned receiver, followed by larger `n=500`
or a cross-family setting.

## Subagent Inputs

- Literature scout: D3PM/MaskGIT/consistency/JEPA justify discrete corruption,
  one-step denoising, and target-state prediction rather than source-text
  reconstruction.
- Harness audit: reuse learned-syndrome packet generation and masked-PQ
  receiver scaffolding, but avoid PQ distance tables and private-text inputs.
- Skeptical reviewer: require target, zero, shuffled, random same-byte,
  answer-only, answer-masked, target-derived, wrong-projection, same-byte text,
  exact-ID parity, and deterministic decoder comparison.

## Decision

Promote as a new technical contribution:

> A control-regularized masked-consistency receiver that consumes learned
> source-private syndrome packets and learns when to use or ignore source
> messages.

This is the most promising method-depth update since the direct Qwen target
decoder and label-blind anti-lookup defense.

## Next Gate

```text
source_private_masked_consistency_receiver_label_blind_stress_20260430
```

Run the same receiver with label-blind/opaque candidate views and public-table
derangements. Pass for the claim boundary is not high accuracy; pass is
collapse of opaque packets plus preservation on the normal public candidate
view. If that holds, widen to `n=500` and one held-out family split.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_masked_consistency_receiver_smoke.py \
  tests/test_summarize_source_private_masked_consistency_receiver.py
```

Outcome: `4 passed in 2.40s`.
