# Frozen Receiver Smoke References

- date: `2026-04-30`
- purpose: references for the frozen embedding/activation receiver smoke and
  the next contrastive receiver branch.

## BGE / FlagEmbedding

- source: https://huggingface.co/BAAI/bge-small-en
- source: https://arxiv.org/abs/2311.13534
- blocker helped: tests whether a general-purpose frozen embedding model can
  replace the explicit semantic-anchor lexicon.
- mechanism idea: use mean-pooled frozen sentence embeddings as the public
  candidate dictionary feature space.
- next experiment change: frozen BGE alone is insufficient; use BGE/Qwen
  features as inputs to a trained contrastive receiver instead.
- role: ablation / baseline.

## Qwen3 Dense Models

- source: https://huggingface.co/Qwen/Qwen3-0.6B
- source: https://arxiv.org/abs/2505.09388
- blocker helped: tests whether recent small LLM activations provide a better
  non-symbolic candidate feature space.
- mechanism idea: extract frozen hidden-state means from a current dense Qwen3
  model under local CPU/MPS constraints.
- next experiment change: the MPS backend failed for the tiny smoke, and CPU
  activations did not pass at n8; keep Qwen activations as future inputs, not a
  promoted result.
- role: ablation / model-family coverage.

## I-JEPA / V-JEPA Framing

- source: https://arxiv.org/abs/2301.08243
- source: https://arxiv.org/abs/2404.08471
- blocker helped: motivates predicting target-useful representations rather
  than reconstructing source text.
- mechanism idea: train a small packet receiver with a candidate-latent
  prediction or denoising objective over frozen features.
- next experiment change: move from off-the-shelf frozen embeddings to a
  contrastive or JEPA-style receiver trained with source-destroying negatives.
- role: inspiration / next-method framing.

## QJL / TurboQuant

- source: https://arxiv.org/abs/2406.03482
- source: https://arxiv.org/abs/2504.19874
- blocker helped: keeps the receiver branch tied to byte-bounded communication
  rather than unbounded latent transfer.
- mechanism idea: learned/frozen features should be transmitted as sign or
  residual-coded packets under matched bytes, with random same-byte controls.
- next experiment change: add QJL-style sign packets as a nonlearned matched
  baseline for the contrastive receiver.
- role: baseline / systems inspiration.

## Bottom Line

Frozen embeddings are a useful reviewer ablation but not yet a headline
technical contribution. The promising next branch is a trained source-control
contrastive receiver over frozen features, evaluated under the existing
source-private packet controls.
