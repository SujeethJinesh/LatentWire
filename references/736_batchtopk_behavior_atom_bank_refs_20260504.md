# BatchTopK Behavior Atom Bank Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

The linear behavior-atom Sparse Resonance Packet branch produced a useful
matched lift on ARC, but candidate-roll and top-atom-knockout controls stayed
competitive. This branch changes the atom basis rather than tuning the receiver:
it trains a tiny BatchTopK-style sparse atom bank against target behavior loss,
then emits the same top-k quantized atom packets used by the strict ARC harness.

## Primary Sources And Novelty Boundary

- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) relax
  per-example TopK to batch-level sparsity, giving adaptive active-latent counts
  at a controlled average sparsity. LatentWire uses this as a packet-basis
  training prior, not as a novelty claim.
- [Crosscoders](https://transformer-circuits.pub/2024/crosscoders/) learn a
  shared sparse dictionary across activation spaces and are a direct prior for
  shared/private atom bases. LatentWire differs only if packet atoms are shown
  to improve receiver task behavior under source-private destructive controls.
- [Dedicated Feature Crosscoders](https://openreview.net/pdf?id=ZB84SvrZB8)
  and Anthropic's [model diffing tooling](https://www.anthropic.com/research/diff-tool)
  sharpen the shared-vs-model-dedicated feature threat. A LatentWire paper cannot
  claim novelty from shared/private sparse features alone.
- [Robustly identifying concepts introduced during chat fine-tuning using
  crosscoders](https://arxiv.org/abs/2504.02922) warns that crosscoder-private
  features can be artifacts of sparsity pressure. This motivates atom-shuffle,
  coefficient-shuffle, top-atom-knockout, seed-stability, and held-out
  candidate-alignment controls.
- [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944)
  train sparse features to approximate component behavior rather than just
  reconstruct activations. This supports behavior-loss packet atoms as a more
  appropriate prior than reconstruction-only SAE atoms.
- [Cache-to-Cache](https://openreview.net/pdf?id=LeatkxrBCi) and
  [KVComm](https://arxiv.org/abs/2510.03346) are direct high-bandwidth
  LLM-to-LLM communication competitors through cache/KV transfer. LatentWire's
  defensible difference is low-rate packet transfer with source-private
  destructive controls and utility-per-byte accounting.
- [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/) keeps the base
  LM frozen and learns continuous prefix vectors. LatentWire must distinguish
  itself by using per-example source-derived packets rather than static
  task-specific continuous prompts.

## Scout Outcome

The first Mac-local BatchTopK behavior atom implementation was added as an
`--atom-basis-mode batchtopk_behavior` option in the strict ARC behavior-atom
harness. Two `.debug/` n8 scouts were run:

| Variant | Packet | Matched | Target | Best required control | Helps | Harms | Decision |
|---|---|---:|---:|---|---:|---:|---|
| BatchTopK top-1 | rank16 top1 q4 | 0.3750 | 0.3750 | qwen_substituted_packet, 0.6250 | 1 | 1 | fail |
| BatchTopK top-2 | rank16 top2 q4 | 0.3750 | 0.3750 | qwen_substituted_packet, 0.6250 | 2 | 2 | fail |

The atom bank fits the tiny train behavior target (`fit_r2 ~= 0.79` for top-1
and `0.91` for top-2), but it does not improve held-out matched accuracy and
adds harms. This weakens the naive "train any sparse behavior atom bank" branch.
It does not rule out DFC/crosscoder atoms, because this scout did not use a
paired shared/private source-target decomposition or a stronger event-triggered
integration loss.

## Consequence

The next highest-value atom branch should not be a larger version of this naive
BatchTopK encoder. If continuing sparse atoms, add a paired source/target
decomposition or behavior-transcoder target-side atom feasibility probe first,
then require candidate-roll and top-atom-knockout sensitivity before n16/n32
promotion.
