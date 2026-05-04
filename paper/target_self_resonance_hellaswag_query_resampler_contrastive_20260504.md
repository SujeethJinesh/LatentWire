# Target Self-Resonance Query-Resampler Contrastive Rescue

## Status

This is a negative method gate. It rules out the simplest rescue of the
target-only query-resampler branch.

Current paper readiness remains below ICLR full-paper standard. The target
oracle-prefix experiments show capacity/headroom, but reusable learned
self-compression still does not separate from target-cache and wrong-row
controls.

## Gate

Script:
`scripts/build_target_self_resonance_hellaswag_query_resampler_gate.py`

Tests:
`tests/test_build_target_self_resonance_hellaswag_query_resampler_gate.py`

Artifact:
`results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train64_validation72_80_contrastive/`

The previous query-resampler failed because a wrong-row prompt prefix was too
competitive and the learned query attention was diffuse. This rescue adds two
bounded training terms:

- a wrong-row contrastive KL penalty, asking matched prompt slots to match the
  full-prompt distribution better than mismatched prompt slots;
- a normalized attention-entropy penalty, asking query slots to attend less
  uniformly over prompt tokens.

The compressed evaluation path remains:

```text
8 learned soft slots + fixed anchor + candidate continuation
```

The target model still does not receive the original context text in the
compressed path.

## Result

Train rows `0:64`, validation rows `72:80`, `8` prefix slots, hidden dimension
`128`, `8` epochs:

| condition | agreement | mean KL |
|---|---:|---:|
| query-resampler | 0.500000 | 0.557950 |
| chunk mean | 0.625000 | 0.426502 |
| slots-only | 0.500000 | 0.217971 |

Gate result: fail.

Key diagnostics:

- best KL control: `slots_only_query`;
- KL gain versus best control: `-0.339979`;
- KL gain versus chunk mean: `-0.131448`;
- mean normalized attention entropy: `1.000000`;
- mean attention max: `0.031230`;
- peak RSS on Mac: `4283.5` MiB;
- runtime: `290.43` s.

Training diagnostics show the rescue did not create useful mismatch separation:

| diagnostic | initial | final |
|---|---:|---:|
| positive train KL | 0.117869 | 0.124080 |
| wrong-row train KL | 0.118629 | 0.124080 |
| contrastive loss | 0.718087 | 0.718460 |
| attention entropy | 0.997079 | 1.000000 |

The negative/wrong-row KL ended equal to the matched KL, and attention became
fully diffuse. The added objective therefore did not make the slots
prompt-specific.

## Decision

Demote the simple target-only query-resampler branch. The evidence now says:

- per-example optimized soft prefixes remain alive only as an oracle capacity
  diagnostic;
- plain query bottlenecks and this contrastive rescue do not create a reusable
  held-out target interface;
- wrong-row and target-only controls are too strong;
- further target-only resampler tuning is lower priority than adding explicit
  source-conditioned residual information over a matched target baseline.

The promoted next method gate is a source-conditioned residual-slot gate:
freeze or reuse a target-only slot baseline, add a small source-conditioned
residual packet, and require source-present slots to beat zero-source,
wrong-source, shuffled-source, target-derived, and candidate-deranged controls
at the same byte/slot budget.

## Lay Explanation

We tried to teach the hidden summary tokens to work only for the correct
question and to stop working for the wrong question. It did not happen: the
summary tokens stayed vague, looked almost the same for matched and mismatched
questions, and the model was better served by a target-only cached prefix.

## Next Exact Gate

Implement `scripts/build_target_self_resonance_hellaswag_source_residual_slot_gate.py`.
The pass rule should require a positive paired gain over the frozen target-slot
baseline and the best destructive control on the same held-out slice before any
adjacent-slice or cross-family widening.
