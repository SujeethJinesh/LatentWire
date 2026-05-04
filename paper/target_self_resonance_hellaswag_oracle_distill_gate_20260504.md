# Target Self-Resonance HellaSwag Oracle-Distill Gate

Date: 2026-05-04

## Readiness Status

- ICLR full paper: still not ready.
- COLM workshop: stronger as an honest target-resonance capacity and
  falsification story, but still missing a positive learned method.
- Exact blocker: oracle-prefix distillation has not yet produced a held-out
  target-side encoder that beats the target-only slots cache.

## Paper Story Update

The per-example oracle self-resonance gate remains the live capacity result:
compact soft prefixes can make a frozen target model behave similarly to its
own full-context run. The previous chunk-residual encoder was too weak. This
gate tested whether distilling those per-example oracle prefixes from official
train rows gives a stronger shared encoder.

The result is still negative. The oracle teachers fit the train rows well, but
the student encoder does not beat a static target-only slots baseline on held
out validation. Stronger student training improves KL over raw chunk means, but
not enough to separate from target-cache effects.

## What Changed

- Added `scripts/build_target_self_resonance_hellaswag_oracle_distill_gate.py`.
- Added `tests/test_build_target_self_resonance_hellaswag_oracle_distill_gate.py`.
- Added a train-only oracle prefix generation path:
  - optimize oracle soft prefixes on official HellaSwag train rows;
  - train a shared encoder with both logit KL and oracle-prefix imitation;
  - train a slots-only target-cache baseline under the same objective;
  - evaluate held-out validation without optimizing validation prefixes.

## Commands

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_oracle_distill_gate.py \
  --output-dir results/target_self_resonance_hellaswag_oracle_distill_gate_20260504_qwen05_train16_validation64_72 \
  --train-start 0 \
  --train-rows 16 \
  --eval-start 64 \
  --eval-rows 8 \
  --prefix-len 8 \
  --hidden-dim 128 \
  --oracle-steps 30 \
  --oracle-lr 0.005 \
  --epochs 5 \
  --lr 0.0001 \
  --oracle-weight 0.05 \
  --device auto \
  --dtype float32 \
  --max-length 256

./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_oracle_distill_gate.py \
  --output-dir results/target_self_resonance_hellaswag_oracle_distill_gate_20260504_qwen05_train16_validation64_72_stronger_student \
  --train-start 0 \
  --train-rows 16 \
  --eval-start 64 \
  --eval-rows 8 \
  --prefix-len 8 \
  --hidden-dim 128 \
  --oracle-steps 30 \
  --oracle-lr 0.005 \
  --epochs 10 \
  --lr 0.001 \
  --weight-decay 0.0 \
  --oracle-weight 0.2 \
  --device auto \
  --dtype float32 \
  --max-length 256
```

## Results

| Run | Pass | Distill Agreement | Distill KL | Chunk Agreement | Chunk KL | Slots Agreement | Slots KL | KL Gain vs Chunk | KL Gain vs Slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default student | `false` | `0.375000` | `0.128162` | `0.375000` | `0.105277` | `0.250000` | `0.106234` | `-0.022885` | `-0.021928` |
| stronger student | `false` | `0.375000` | `0.097225` | `0.375000` | `0.105277` | `0.375000` | `0.089671` | `+0.008052` | `-0.007554` |

Both runs used `16` official train rows and evaluated validation `64:72`.
The oracle train prefixes fit well: mean train KL fell from `0.107092` to
`0.000983`. The failure is therefore not target controllability; it is held-out
student generalization beyond a target-only slots cache.

The emitted packet remains `8` Qwen soft slots, or `14,336` raw fp16 bytes.
The stronger student encoder has `253,705` trainable parameters. This is still
a target-side self-compression diagnostic, not cross-model communication.

## Interpretation

Promoted: the oracle teacher surface is real and train rows can be fitted to
very low target KL.

Weakened: direct oracle-prefix distillation from chunk means is not enough. It
can improve over chunk KL with stronger student training, but it does not beat
the slots-only target cache.

Alive: query-resampler / ICAE-style encoders, source-conditioned residual slots
on top of a stronger target interface, and common-basis regularizers such as
relative representations or sparse/crosscoder features.

Saturated: shallow receiver switchers and plain chunk-residual encoders remain
lower priority.

Cut if needed: do not present oracle-prefix distillation as a contribution
unless a later larger/seeded run beats slots-only. Keep it as an ablation that
explains why the interface needs real query bottlenecks or source-conditioned
residual information.

## Lay Explanation

We first made excellent custom soft-token answers for training examples. Then
we trained one small encoder to imitate those custom soft tokens on new
examples. It got somewhat better when trained harder, but a generic learned
slot baseline still carried more useful target-side behavior. So the problem
is not that soft tokens cannot work; the problem is that this simple encoder
does not yet know how to create the right ones for new questions.

## Next Exact Gate

Run a query-resampler / ICAE-style target interface:

- use learned query slots that attend over the target prompt/chunk summaries
  rather than only applying slot-wise residual MLPs;
- train on official train rows with KL plus oracle-prefix/headroom telemetry;
- evaluate validation against chunk, slots-only, zero, random, wrong-row, and
  candidate-deranged controls;
- only if it beats slots-only on adjacent slices should we add source-conditioned
  residual slots and strict cross-family Phi/Qwen controls.

Systems note: the systems claim remains pending. On Mac we can measure bytes,
hardware-rounded traffic, encode/decode proxy time, source exposure labels, and
paired quality controls. NVIDIA/vLLM/SGLang are still required for TTFT, TPOT,
HBM traffic, continuous batching, and fair C2C/KVComm/TurboQuant comparison
rows.
