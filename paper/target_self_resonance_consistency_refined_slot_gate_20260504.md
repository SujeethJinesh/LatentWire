# Target Self-Resonance Consistency-Refined Slot Gate

Date: 2026-05-04

## Status

This gate is not an ICLR-ready positive result. It is a bounded follow-up after
the direct source-hidden residual slot gate failed to move answers. The new
branch adds one target-native refinement step, but the held-out result does not
separate from wrong-source or candidate-roll controls.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The current story is that LatentWire has a strong source-private
packet harness and strict controls, while target-native latent receivers remain
the live positive-method gap. The exact ICLR blocker is still a frozen receiver
that uses source-conditioned evidence to beat target-only, wrong-source,
target-derived, source-index/rank, and same-budget baselines with paired
uncertainty.

## Why This Gate

The previous source-hidden residual gate nudged KL but did not change answers.
This gate tests a more structured idea inspired by one-step consistency or
denoising refinement:

```text
frozen Qwen slots
+ TinyLlama source-hidden residual
+ one learned refinement update conditioned on Qwen's current candidate scores
-> Qwen candidate scoring
```

The compressed target path never sees the original HellaSwag context text.

Lay explanation: instead of sending Qwen one static hidden clue, we first let
Qwen score the answer choices from the hidden clue, then let a small learned
module revise the clue once after seeing Qwen's current score state. If this
worked, it would look like the source message helps Qwen correct itself.

## Artifacts

- script:
  `scripts/build_target_self_resonance_hellaswag_consistency_refined_slot_gate.py`
- test:
  `tests/test_build_target_self_resonance_hellaswag_consistency_refined_slot_gate.py`
- artifact:
  `results/target_self_resonance_hellaswag_consistency_refined_slot_gate_20260504_tiny_to_qwen05_train64_validation88_96/`
- references:
  `references/707_target_self_resonance_consistency_refined_slot_refs_20260504.md`

## Command

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_consistency_refined_slot_gate.py \
  --output-dir results/target_self_resonance_hellaswag_consistency_refined_slot_gate_20260504_tiny_to_qwen05_train64_validation88_96 \
  --train-start 0 \
  --train-rows 64 \
  --eval-start 88 \
  --eval-rows 8 \
  --slot-epochs 3 \
  --residual-epochs 3 \
  --refine-epochs 3 \
  --lr 1e-4 \
  --refine-lr 1e-4 \
  --initial-residual-gate -8 \
  --initial-refine-gate -8 \
  --source-contrastive-weight 0 \
  --hidden-feature-mode top2_delta \
  --bootstrap-samples 1000 \
  --device auto \
  --source-lm-device auto \
  --dtype float32 \
  --source-lm-dtype float16 \
  --local-files-only true \
  --run-date 2026-05-04
```

## Result

```text
consistency-refined accuracy:      0.375000
frozen target-slot accuracy:       0.250000
source residual no-refine accuracy:0.250000
wrong-source refine accuracy:      0.375000
candidate-roll refine accuracy:    0.375000
source top1 label accuracy:        0.500000
source top1/top2 oracle accuracy:  1.000000
refined mean KL:                   0.178246
no-refine mean KL:                 0.178143
paired delta vs frozen:            0.125000
CI95 low vs frozen:                0.000000
pass gate:                         false
```

The method flips one additional row relative to frozen slots, but the same
answer movement appears under wrong-source, candidate-roll, and shuffled-refine
controls. The refinement step also worsens KL versus the no-refine source
residual path.

## Decision

Demote the current one-step consistency-refined slot branch. The failure is
not just low accuracy; it is a specificity failure. A reviewer would reasonably
say the refinement module learned a generic score perturbation rather than
using TinyLlama evidence.

This weakens generic "refinement over target slots" as a contribution. It does
not kill target-native latent transfer because the source top1/top2 oracle is
`1.000000` on this slice, but it says the next method must extract that
ambiguity under stronger controls instead of adding another unconstrained
residual step.

## What This Rules Out

- One shallow learned refinement update over target slots is not enough.
- Tiny KL movement is not evidence of communication when wrong-source and
  candidate-roll controls match answer movement.
- The current high-dimensional hidden feature is not a systems win:
  `16404` fp16 bytes of source feature state, before any target-side compute.

## What Remains Live

- The strict HellaSwag hidden-innovation packet remains the strongest positive
  surface: it passes validation `0:9216` but not the terminal tail.
- Score-only and source-score quantization controls are saturated and should
  stay as reviewer controls, not the positive method.
- A stronger next branch should either learn a compact sparse/common-basis
  source code with atom-shuffle and wrong-row controls, or use supervised
  target-native oracle-prefix distillation with hard source-destroying controls.

## Next Exact Gate

Do not widen benchmark scope on this branch. The next exact Mac-local gate
should be a compact source-conditioned common-basis receiver that has to beat:

```text
frozen target slots
zero source
wrong source
candidate roll
target-derived source features
refinement-step shuffle
source top1/source-rank controls
same-budget text or candidate-only packet
```

If that also fails, simplify the ICLR plan around the strict packet protocol,
systems accounting, and limitations, while reserving NVIDIA runs for C2C /
KVComm / TurboQuant-style systems comparisons.
