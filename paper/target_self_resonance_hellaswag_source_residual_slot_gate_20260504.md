# Target Self-Resonance HellaSwag Source-Residual Slot Gate

## Status

This is a weak positive signal but a failed method gate. It should not be
framed as an ICLR-ready positive method.

Current paper readiness remains below ICLR full-paper standard. The experiment
shows that a compact source-derived code can sometimes steer a frozen target
slot interface, but the current residual-slot controller does not beat the
direct source-label shortcut and does not have positive paired uncertainty.

## Gate

Script:
`scripts/build_target_self_resonance_hellaswag_source_residual_slot_gate.py`

Tests:
`tests/test_build_target_self_resonance_hellaswag_source_residual_slot_gate.py`

Artifacts:

- `results/target_self_resonance_hellaswag_source_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation72_80/`
- `results/target_self_resonance_hellaswag_source_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation72_80_gate2p5/`

Setup:

- source: cached TinyLlama HellaSwag score summaries;
- target: frozen Qwen2.5-0.5B-Instruct;
- train rows: `64` cached official-train source rows;
- eval rows: frozen HellaSwag validation `72:80`;
- target interface: `8` target-native soft slots;
- source packet: top-1/top-2/margin score summary, `2B` raw / `5B` framed;
- compressed target path:

```text
frozen target slots + residual(source packet) + fixed anchor + candidate
```

The compressed path does not receive source text, source KV, raw source hidden
vectors, or a raw source score vector.

## Controls

The gate includes:

- `frozen_target_slots`;
- `zero_source_residual`;
- `wrong_source_residual`;
- `candidate_roll_source_residual`;
- `target_derived_residual`;
- `random_same_norm_residual`;
- `source_top1_label_control`;
- `candidate_derangement`.

The strict pass rule requires matched source to beat frozen target slots with
paired uncertainty, improve KL, remain finite, and beat every destructive
control including the direct source-top1 label shortcut.

## Result

Default bounded-gate run:

| condition | accuracy | agreement | mean KL |
|---|---:|---:|---:|
| frozen target slots | 0.375000 | 0.500000 | 0.380034 |
| source residual slots | 0.375000 | 0.500000 | 0.324309 |
| source top1 label control | 0.750000 | 0.750000 | 0.124459 |

The source residual improved KL but did not change accuracy.

Higher residual-gate rescue, `initial_residual_gate=-2.5`:

| condition | accuracy | agreement | mean KL |
|---|---:|---:|---:|
| frozen target slots | 0.375000 | 0.500000 | 0.406056 |
| source residual slots | 0.500000 | 0.625000 | 0.395800 |
| zero source residual | 0.375000 | 0.500000 | 0.426537 |
| wrong source residual | 0.375000 | 0.500000 | 0.479964 |
| target-derived residual | 0.375000 | 0.500000 | 0.493029 |
| random same-norm residual | 0.375000 | 0.500000 | 0.457089 |
| source top1 label control | 0.750000 | 0.750000 | 0.124459 |

The higher-gate rescue gives one held-out help and zero harms versus frozen
target slots:

- source residual accuracy: `0.500000`;
- frozen target-slot accuracy: `0.375000`;
- paired mean delta versus frozen: `+0.125000`;
- paired CI95 low versus frozen: `0.000000`;
- KL gain versus frozen: `+0.010256`;
- residual gate after training: `0.074866`;
- residual RMS after training: `0.017190`;
- peak RSS: `4602.125` MiB.

Gate result: fail.

## Decision

Promote the broader source-conditioned residual-interface branch, but demote
this exact continuous residual-slot implementation as insufficient.

What changed scientifically:

- The source packet contains useful task signal on this slice: source-top1 is
  `0.750000`.
- Matched source residual can steer the target: `0.500000` versus `0.375000`
  frozen target slots, with wrong-source, zero-source, target-derived, and
  random controls at `0.375000`.
- The effect is not yet strong enough for a paper: it is below direct
  source-top1 label copy and the paired CI lower bound is not positive.

## Lay Explanation

We gave the target model a small hidden baseline summary, then added a tiny
source-derived correction. With a larger correction knob, the source correction
fixed one example and did not break any examples. That is encouraging, but the
simple source answer shortcut was still much better, so this is not yet the
method we can publish.

## Next Exact Gate

Build a quantized source-conditioned candidate repair or codebook residual
gate. The next method should use the same matched/wrong/zero/target-derived
controls, but it should more directly convert top-1/top-2 source evidence into
target candidate preference changes instead of asking a generic continuous
slot residual to discover that mapping from only `64` rows.
