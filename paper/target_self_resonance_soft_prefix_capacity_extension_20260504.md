# Target Self-Resonance Soft-Prefix Capacity Extension

Current paper readiness: COLM remains plausible as a scoped source-private
packet/evaluation artifact; ICLR remains blocked. Estimated ICLR readiness is
roughly 40% after this run because the target resonance control surface is now
more credible, but the learned source-private encoder is still missing.

Current paper story: score-level packets are saturated on Qwen-to-Phi, but a
frozen target model can be steered into full-context behavior by compact
continuous prefixes. The live method branch is therefore target-native
selective logit resonance, not more score switching.

Exact gap blocking submission: convert per-example oracle-optimized soft
prefixes into a held-out source-derived encoder that beats chunk, slots-only,
zero, random, shuffled-row, candidate-deranged, source-row-shuffle, and
same-byte controls with paired uncertainty.

## Purpose

The previous target self-resonance oracle capacity probe passed on HellaSwag
validation `0:32`. Learned encoders then failed on later slices, especially
around validation `48:64`, raising the question:

> Is the target control surface itself unstable, or are our reusable encoders
> too weak?

This extension reruns the per-example oracle soft-prefix gate on validation
`32:48` and `48:64` using the same frozen Qwen2.5-0.5B-Instruct target, `8`
soft prefix tokens, and the same full-prompt score distribution as the
behavioral reference.

Lay explanation: first Qwen answers with the full HellaSwag question. Then the
question is removed, and we tune only eight hidden tokens until Qwen gives the
same answer distribution. This does not yet teach another model to send the
tokens, but it checks whether such tokens can exist reliably.

## Runs

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py \
  --output-dir results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation32_48 \
  --row-start 32 \
  --row-limit 16 \
  --prefix-len 8 \
  --steps 30 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py \
  --output-dir results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation48_64 \
  --row-start 48 \
  --row-limit 16 \
  --prefix-len 8 \
  --steps 30 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

## Result

Both new slices pass.

| Slice | Rows | Optimized agreement | Optimized KL | Chunk agreement/KL | Best destructive agreement | Best destructive KL |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 1.000000 | 0.000060 | 0.625000 / 0.093210 | 0.687500 | 0.098435 |
| 48 | 16 | 1.000000 | 0.000181 | 0.687500 / 0.075388 | 0.750000 | 0.097099 |

Across all four now-run slices, validation `0:64`:

| Condition | Rows | Accuracy | Full-prompt agreement | Mean KL to full prompt |
|---|---:|---:|---:|---:|
| full prompt | 64 | 0.453125 | 1.000000 | 0.000000 |
| optimized soft prefix | 64 | 0.421875 | 0.937500 | 0.003533 |
| chunk-mean prefix | 64 | 0.312500 | 0.546875 | 0.106579 |
| zero prefix | 64 | 0.312500 | 0.593750 | 0.142256 |
| random same-norm prefix | 64 | 0.296875 | 0.531250 | 0.128185 |
| shuffled optimized prefix | 64 | 0.281250 | 0.609375 | 0.134414 |
| candidate derangement | 64 | 0.250000 | 0.046875 | 0.563535 |

The important result is not gold-label accuracy. The optimized prefix is
trained to match Qwen's own full-prompt behavior, which may itself be wrong.
The core capacity metric is agreement/KL to the full-prompt distribution.

## Interpretation

Promote target self-resonance as a real capacity surface. The target can often
be driven into its full-context decision state using `8` continuous tokens
without seeing the original context text. The adjacent `48:64` pass matters
because prior learned encoders failed there, so the bottleneck is now more
clearly encoder generalization and source-specificity, not target reachability.

This does not prove cross-model communication. The optimized prefixes are
per-example oracle variables and cost `14,336` raw fp16 bytes for Qwen's hidden
dimension. They are a teacher/oracle surface for the next encoder, not a
systems-efficient packet yet.

## Decision

The next highest-priority method branch should be a selective
logit-resonance encoder:

- train on official train rows to predict target-native soft prefixes;
- optimize logit KL to the target full-prompt distribution;
- add top1/top2 behavioral loss only as a stabilizer, not as label leakage;
- match only answer-relevant score/logit subspaces, not all hidden states;
- normalize emitted slots to target embedding RMS;
- include wrong-row, source-row-shuffle, candidate-roll, target-derived,
  zero-source, same-byte random, chunk, slots-only, and shuffled-prefix
  controls;
- pass only if held-out agreement/KL beats target-only controls and answer
  accuracy improves over fixed hybrid/source-index controls with paired CI.

Do not spend more mainline effort on shallow score packets unless a new
source-specific feature explains why source top1/top2 should override the
target.

## Artifacts

- Existing script:
  `scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py`
- New results:
  `results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation32_48/`
  and
  `results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation48_64/`
- References:
  `references/712_target_self_resonance_capacity_extension_refs_20260504.md`
