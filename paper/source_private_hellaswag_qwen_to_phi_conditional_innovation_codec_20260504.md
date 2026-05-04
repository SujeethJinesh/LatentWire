# HellaSwag Qwen-To-Phi Conditional Innovation Codec Gate

Date: 2026-05-04

## Status

- Current paper readiness: COLM workshop evidence is stronger after this gate
  because another shallow cross-family repair is ruled out; ICLR remains
  blocked by the lack of a strict positive cross-family method.
- Current story: LatentWire can express source-private fixed-byte packet
  boundaries and strong controls, but the Qwen-to-Phi receiver still fails to
  convert source evidence into reliable HellaSwag gains.
- Exact gap: the receiver selected a no-op fixed-hybrid policy, so this branch
  does not yet prove cross-model communication.

## Gate

- script:
  `scripts/build_source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate.py`
- test:
  `tests/test_build_source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate.py`
- artifact:
  `results/source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate_20260504_validation1024_2048/`
- references:
  `references/701_hellaswag_qwen_to_phi_conditional_innovation_codec_refs_20260504.md`

The gate fits a Phi-side ghost predictor of Qwen's four-candidate score
geometry on official HellaSwag train rows, then transmits a tiny quantized
residual innovation code from Qwen to the Phi receiver. It evaluates on frozen
validation `1024:2048` with source-row shuffle, random-code, code-value
permutation, candidate-roll, target-derived, ghost-only, and label-permutation
controls.

## Result

Fail.

| Metric | Value |
| --- | ---: |
| calibration rows | `1487` |
| eval rows | `768` |
| selected rate | `1B` raw / `4B` framed |
| fixed hybrid accuracy | `0.467448` |
| conditional innovation accuracy | `0.467448` |
| delta vs fixed hybrid | `0.000000` |
| CI95 low vs fixed hybrid | `0.000000` |
| candidate-only accuracy | `0.455729` |
| ghost-only accuracy | `0.309896` |
| best destructive control accuracy | `0.388021` |
| overrides vs fixed hybrid | `0` |

The official-train selector found that all useful settings reduced to fixed
hybrid. Higher-rate candidates sometimes made one override on official dev,
but did not improve the decision surface. On eval, the selected packet has
positive margin over candidate-only and ghost-only controls, but that is only
because it is exactly fixed hybrid.

## Decision

Rule out the current linear ghost plus discrete residual innovation codec as a
positive ICLR branch. The conditional side-information idea is still alive, but
the receiver must be materially stronger than a shallow ridge decoder and must
prove nonzero source-conditioned overrides.

Promote the next gate to a target-side learned receiver/interface that first
passes target self-compression/resonance controls, then adds source packets
with strict wrong-source and target-derived controls. The method must report
nonzero beneficial overrides, larger frozen slices, seed repeats, and one
cross-family falsification pair.

## Lay Explanation

We asked Phi to guess what Qwen would think, then let Qwen send only a tiny
correction. The best safe model ignored the correction and kept using the old
fixed Qwen answer packet. That means this specific correction format did not
create useful cross-model communication.
