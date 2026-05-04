# Target Self-Resonance HellaSwag Chunk-Encoder Gate

Date: 2026-05-04

## Readiness Status

- ICLR full paper: still not ready.
- COLM workshop: improved as an honest positive-capacity plus failed
  generalization story.
- Exact blocker: the learned target-side encoder is not yet stable enough
  across held-out slices to become the paper's positive method.

## Paper Story Update

The per-example oracle self-resonance gate showed that compact soft prefixes can
recreate the target model's own full-context behavior. This gate asks the next
reviewer-critical question: can one shared encoder, trained only on official
train rows, emit useful target-native slots on unseen validation rows?

The answer is mixed. The simple chunk-residual encoder is numerically stable
after RMS normalization and gradient clipping, and it can beat chunk-mean and
slots-only on one held-out slice. But it fails the adjacent-slice repeat because
slots-only remains a stronger KL baseline. This weakens the simple
chunk-residual encoder as the ICLR method.

## What Changed

- Added `scripts/build_target_self_resonance_hellaswag_chunk_encoder_gate.py`.
- Added `tests/test_build_target_self_resonance_hellaswag_chunk_encoder_gate.py`.
- Added explicit nonfinite telemetry after an initial NaN failure exposed an
  overly aggressive residual gate.
- Stabilized training with:
  - small sigmoid residual scaling;
  - row-wise RMS normalization to the target embedding manifold;
  - gradient clipping;
  - pass-gate blocking on nonfinite KL or score rows.

## Commands

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_chunk_encoder_gate.py \
  --output-dir results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation32_48 \
  --train-start 0 \
  --train-rows 64 \
  --eval-start 32 \
  --eval-rows 16 \
  --prefix-len 8 \
  --hidden-dim 128 \
  --epochs 5 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --norm-weight 0.001 \
  --bootstrap-samples 1000 \
  --device auto \
  --dtype float32 \
  --max-length 256 \
  --max-mean-kl 0.15

./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_chunk_encoder_gate.py \
  --output-dir results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation48_64 \
  --train-start 0 \
  --train-rows 64 \
  --eval-start 48 \
  --eval-rows 16 \
  --prefix-len 8 \
  --hidden-dim 128 \
  --epochs 5 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --norm-weight 0.001 \
  --bootstrap-samples 1000 \
  --device auto \
  --dtype float32 \
  --max-length 256 \
  --max-mean-kl 0.15
```

## Results

| Slice | Pass | Learned Agreement | Learned KL | Chunk Agreement | Chunk KL | Slots Agreement | Slots KL | KL Gain vs Chunk | KL Gain vs Slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train `32`, validation `32:48` | `false` | `0.625000` | `0.094576` | `0.625000` | `0.093210` | `0.625000` | `0.097726` | `-0.001365` | `+0.003151` |
| validation `32:48` | `true` | `0.687500` | `0.081292` | `0.625000` | `0.093210` | `0.687500` | `0.090259` | `+0.011918` | `+0.008967` |
| validation `48:64` | `false` | `0.750000` | `0.074528` | `0.687500` | `0.075388` | `0.687500` | `0.067913` | `+0.000860` | `-0.006615` |

The 32-train-row run used the same stabilized code path and failed, so the
small pass on validation `32:48` should not be overclaimed. The 64-train-row
encoder has `239,361` trainable parameters. The emitted `8` Qwen soft slots
are `14,336` raw fp16 bytes. This is a target-side context-compression gate,
not a source-private communication result.

## Interpretation

Promoted: RMS-normalized target-native slots are trainable on Mac and can
generalize weakly from official train to validation.

Weakened: a plain chunk-mean residual encoder is not enough for ICLR. It does
not reliably beat the slots-only target-cache control across adjacent slices.

Alive: oracle-prefix distillation and query-resampler / ICAE-style encoders.
The oracle capacity result remains strong; the failure is the encoder class,
not target controllability.

Saturated: shallow Qwen-to-Phi switchers remain saturated. This result does
not justify returning to that branch.

Cut if needed: do not put this simple chunk-residual encoder forward as a core
technical contribution. Keep it as an ablation showing why the target-slot
interface needs a stronger encoder.

## Lay Explanation

Last time we manually tuned special soft tokens for each question. This time
we trained one small encoder on training questions and asked it to create those
soft tokens for new questions. It helped on one small group but not the next
one, because a generic learned slot baseline was still too competitive. That
means the idea is not dead, but the simple encoder is too weak.

## Next Exact Gate

Run oracle-prefix distillation or a query-resampler encoder:

- generate oracle optimized prefixes on a small official-train slice;
- train an encoder to imitate those prefixes and also match target logits;
- evaluate held-out rows against chunk-mean, slots-only, zero, random,
  shuffled-row, and candidate-deranged controls;
- only after this clears adjacent slices should we add source-conditioned
  residual slots and cross-family Phi/Qwen controls.

Systems note: this branch is not a systems win yet. The relevant systems claim
requires either smaller quantized slot packets or native GPU comparison against
C2C/KVComm/TurboQuant-style KV/vector sharing.
