# HellaSwag Hidden-Innovation Heldout Slice References

## Primary Sources

- HellaSwag benchmark: https://arxiv.org/abs/1905.07830
- Prefix-Tuning: https://arxiv.org/abs/2101.00190
- Prompt Tuning / scale effects: https://arxiv.org/abs/2104.08691
- P-Tuning v2: https://arxiv.org/abs/2110.07602
- Relative representations / latent communication: https://arxiv.org/abs/2209.15430
- Sparse autoencoders for language-model features: https://arxiv.org/abs/2309.08600
- Universal sparse autoencoders: https://arxiv.org/abs/2502.03714
- Sparse crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
- C2C cache-to-cache communication: https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing: https://openreview.net/forum?id=F7rUng23nw
- QJL 1-bit quantized Johnson-Lindenstrauss transform: https://arxiv.org/abs/2406.03482
- TurboQuant online vector quantization: https://arxiv.org/abs/2504.19874

## Novelty Boundary

This heldout-slice gate is not claiming to invent prompt/prefix tuning,
common-basis representation learning, sparse autoencoders, crosscoders,
cache-to-cache communication, selective KV sharing, or low-bit vector
quantization.

The safe novelty claim is narrower: a source model can compute private hidden
innovation internally and emit only a fixed `2B` raw / `5B` framed
candidate/confidence packet. The receiver does not ingest source text, source
KV, raw source hidden vectors, raw source scores, soft prompt vectors, prefix
tokens, or adapter weights. The heldout slice tests whether this packet
remains useful when the evaluation rows move beyond the repeatedly inspected
validation-first1024 slice.

## Prefix-Token Distinction

Prefix tuning, prompt tuning, and P-Tuning learn task-level continuous prompt
state that is inserted into the model context or internal prefix state.
LatentWire's packet is a per-example discrete boundary record decoded into a
candidate decision with public candidate side information. It is therefore a
communication/accounting interface rather than a parameter-efficient tuning
method.

## Next Method Implication

If full validation or a predeclared multi-slice stress also passes, the paper
can present HellaSwag as the hard non-science headline benchmark. If it fails,
the correct next branch is sparse/common-basis hidden innovation: replace dense
hidden residuals with anchor-relative, SAE, crosscoder, or quantization-shaped
features, then require atom/basis shuffles to collapse.
