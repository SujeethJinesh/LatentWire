# HellaSwag Rank/Score-Channel Control References

Date: 2026-05-03

## Purpose

This memo records the novelty boundary and next-method literature after adding
source-rank/index-only and score-channel-roll controls to the HellaSwag
hidden-innovation packet gate.

## Local decision result

The stricter HellaSwag first-slice gate passes:

- selected hidden-innovation packet: `0.512695`;
- best label-copy: `0.463867`;
- source-rank/index-only bagged control: `0.461914`;
- score-only bagged control: `0.461914`;
- zero-hidden control: `0.461914`;
- score-channel-roll hidden control: `0.252930`;
- paired CI95 low versus best label-copy: `+0.026367`;
- paired CI95 low versus source-rank/index-only: `+0.033203`;
- pass gate: `True`.

The result strengthens the communication-control story but does not remove the
ICLR blocker: the strict controls must be rerun across larger frozen slices and
at least one strict receiver-family or cross-family pair.

## Closest method boundaries

- Relative representations:
  https://arxiv.org/abs/2209.15430
  - Supports anchor-relative/common-basis thinking for latent spaces. It is not
    by itself a fixed-byte source-private communication protocol.
- SAE feature universality:
  https://arxiv.org/abs/2410.06981
  - Supports the hypothesis that sparse feature spaces can be compared across
    LLMs. LatentWire must still show downstream packet utility under destructive
    controls.
- Crosscoders / model diffing:
  https://www.anthropic.com/research/crosscoder-model-diffing
  and
  https://transformer-circuits.pub/2024/crosscoders/index.html
  - Strong inspiration for shared/exclusive features. The novelty risk is high
    unless the contribution is framed as source-private fixed-byte transfer and
    not just feature discovery.
- Universal logit distillation:
  https://arxiv.org/abs/2402.12030
  - Relevant for cross-tokenizer source/receiver matching, especially
    TinyLlama-to-Qwen. It is a distillation objective, not a private packet
    interface.
- Gist tokens:
  https://arxiv.org/abs/2304.08467
  - Learned compression tokens are prior art. LatentWire must avoid claiming
    soft/prefix-token novelty and must separate target self-compression from
    source communication.
- Consistency models:
  https://arxiv.org/abs/2303.01469
  - Useful analogy for packet corruption/repair objectives. Not a direct
    baseline unless implemented as a receiver robustness loss.
- TurboQuant:
  https://arxiv.org/abs/2504.19874
  - Motivates rotation plus extreme quantization. For LatentWire it is a
    systems/compression boundary, not a novelty claim.
- QJL:
  https://arxiv.org/abs/2406.03482
  - Relevant for sign-sketching low-byte residual information and for byte-floor
    comparisons against transmitting dense states.

## Novelty claim that survives this control

The defensible claim is:

1. per-example source-conditioned packet communication;
2. fixed-byte source-private payload with no source text, KV cache, raw hidden
   vector, or raw score transmission;
3. destructive controls showing that the packet is not reducible to source
   label copy, candidate rank/index, source score, zero hidden state,
   wrong-row hidden state, candidate-order hidden corruption, or rolled
   score-channel metadata.

The claim still does not yet cover cross-family generalization.

## Next exact gate

Run the strict rank/score-channel controls across HellaSwag held-out slices.
If they pass, run a receiver-family/cross-family falsification. If they fail,
move to the ARC sparse common-feature innovation packet branch:

- top-k signed innovation in a public/common basis;
- packet budgets `3B`, `6B`, `11B`;
- controls: source-label, source-index, quantized source-score, same-byte text,
  target-derived packet, zero-source, row-shuffle, candidate-roll,
  candidate-score-roll, feature-id shuffle, and label-shuffle.
