# Target Self-Resonance Source-Hidden Residual Slot Gate

Date: 2026-05-04

## Status

This gate is not an ICLR-ready positive result. It directly tests the live
source-conditioned target-native receiver branch, but it fails to separate from
zero-source and frozen target-slot controls on the held-out validation slice.

## Why This Gate

The previous target-only soft-prefix gates proved capacity but not reusable
communication. The previous score-residual gate gave TinyLlama only a shallow
top-2/margin packet. This gate gives the bridge a richer source signal:
TinyLlama candidate hidden summaries are mapped into a Qwen target-native
soft-prefix residual.

In plain terms, the experiment asks:

```text
Can TinyLlama send Qwen a small learned hidden-message that helps Qwen answer,
without showing Qwen the original HellaSwag context text?
```

## Artifacts

- script:
  `scripts/build_target_self_resonance_hellaswag_source_hidden_residual_slot_gate.py`
- test:
  `tests/test_build_target_self_resonance_hellaswag_source_hidden_residual_slot_gate.py`
- mean/top1-delta artifact:
  `results/target_self_resonance_hellaswag_source_hidden_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation80_88/`
- top2-delta stable artifact:
  `results/target_self_resonance_hellaswag_source_hidden_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation80_88_top2_stable/`
- references:
  `references/706_target_self_resonance_source_hidden_residual_slot_refs_20260504.md`

## Main Result

The stricter top2-delta variant:

```text
source-hidden residual accuracy: 0.375000
frozen target-slot accuracy:     0.375000
zero-source hidden accuracy:     0.375000
source top1 label accuracy:      0.125000
source top1/top2 oracle:         0.625000
source-hidden residual mean KL:  0.165418
frozen target-slot mean KL:      0.166265
paired delta vs frozen:          0.000000
CI95 low vs frozen:              0.000000
pass gate:                       false
```

The mean/top1-delta variant also fails:

```text
source-hidden residual accuracy: 0.375000
frozen target-slot accuracy:     0.375000
zero-source hidden accuracy:     0.375000
source-hidden residual mean KL:  0.137524
frozen target-slot mean KL:      0.133810
paired delta vs frozen:          0.000000
pass gate:                       false
```

## Decision

Demote the direct high-dimensional hidden-to-residual-MLP branch as currently
implemented. The top2-delta variant shows a tiny KL improvement, but no answer
movement and no source-specific separation: zero-source, wrong-source,
candidate-roll, and frozen-slot controls all land on the same accuracy.

This does not kill target-native latent transfer. It weakens the idea that a
small residual MLP over frozen slots is enough. The source top1/top2 oracle at
`0.625000` still says there is useful source-side information on the slice.
The receiver is not extracting it.

## What This Rules Out

- A raw TinyLlama hidden summary plus shallow residual MLP is not yet a
  positive held-out receiver.
- A tiny KL reduction is not acceptable as a paper contribution when accuracy
  is unchanged and zero-source ties the method.
- This branch is not a systems win: the current hidden feature is `16404` fp16
  bytes, larger than the `14336` fp16-byte Qwen soft-prefix path.

## What Remains Live

The live branch should move from direct hidden-to-slot regression to one of:

- a source-conditioned candidate repair head trained to resolve top1/top2
  ambiguity under wrong-source and target-derived controls;
- an oracle-prefix distillation target, where source features learn to predict
  per-example target-native oracle prefixes instead of only matching logits;
- a PCA/SAE/shared-feature bottleneck before the target-slot decoder, so the
  bridge is forced into a smaller, interpretable common basis;
- a consistency-refined slot interface: initialize slots from source evidence,
  then run one or two target-native denoising/refinement steps before scoring.

## Lay Explanation

TinyLlama often narrows the answer down to the right two choices, but the
bridge still cannot tell Qwen how to use that clue. The hidden message nudges
Qwen's internal scores a little, but it does not change which answer Qwen
chooses, and a zero-source message works just as well on this slice.

## Next Exact Gate

Run a source-conditioned candidate repair or oracle-prefix-distillation gate
that explicitly targets the top1/top2 ambiguity. Required controls:

```text
frozen target slots
zero source
wrong source
candidate roll
target-derived packet
source top1 label
source top1/top2 oracle
candidate derangement
```

Do not widen benchmark scope until one source-conditioned receiver produces a
strict accuracy delta over frozen slots and destructive controls with paired
uncertainty.
