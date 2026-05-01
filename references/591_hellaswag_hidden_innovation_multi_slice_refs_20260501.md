# HellaSwag Hidden-Innovation Multi-Slice References

## Primary Sources

- HellaSwag benchmark: https://arxiv.org/abs/1905.07830
- Prefix-Tuning: https://arxiv.org/abs/2101.00190
- Prompt Tuning / scale effects: https://arxiv.org/abs/2104.08691
- P-Tuning v2: https://arxiv.org/abs/2110.07602
- Relative representations / latent communication: https://arxiv.org/abs/2209.15430
- Sparse autoencoders for language-model features: https://arxiv.org/abs/2309.08600
- SAE scaling: https://arxiv.org/abs/2406.04093
- SAE feature universality: https://arxiv.org/abs/2410.06981
- Sparse crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
- C2C cache-to-cache communication: https://arxiv.org/abs/2510.03215
- KVCOMM online cross-context KV-cache communication: https://arxiv.org/abs/2510.12872
- QJL 1-bit quantized Johnson-Lindenstrauss transform: https://arxiv.org/abs/2406.03482
- TurboQuant online vector quantization: https://arxiv.org/abs/2504.19874

## Novelty Boundary

The multi-slice gate does not claim to invent prompt or prefix tuning,
representation alignment, sparse autoencoders, crosscoders, cache-to-cache
communication, selective KV sharing, Johnson-Lindenstrauss sketches, or vector
quantization. Those works define the adjacent spaces reviewers will compare
against.

The safe claim is narrower and stronger: under a source-private boundary, a
source model computes hidden innovation internally and emits only a fixed
`2B` raw / `5B` framed candidate/confidence packet. The receiver does not
ingest source text, source KV, raw source hidden vectors, raw source scores,
soft prompt vectors, prefix tokens, or adapter weights. The new evidence is
that this exact packet clears contiguous HellaSwag validation rows `0:3072`
against label-copy, score-only, zero-hidden, wrong-example, candidate-roll,
and jackknife controls.

## Prefix-Token Distinction

Prefix tuning and related prompt-tuning methods learn task-level continuous
state that is inserted into a model as virtual tokens, prompt embeddings, or
prefix-like internal state. LatentWire instead sends a per-example discrete
boundary record after a source model has inspected the candidate set. It is
therefore a communication and systems-accounting interface, not a
parameter-efficient tuning method.

## SAE / Common-Basis Implication

SAE, crosscoder, and relative-representation work remain valuable as next
method branches because they offer ways to build shared coordinate systems for
hidden features. The current multi-slice result is dense and same-source-model
specific; if remaining slices or cross-family gates fail, the next highest-EV
branch is to replace dense hidden residuals with sparse/common-basis atoms and
require basis-shuffle or atom-knockout controls to collapse.

## Systems Boundary

C2C and KVCOMM transmit or manipulate model KV/cache state, while QJL and
TurboQuant compress high-dimensional vectors or KV/cache state. LatentWire's
current HellaSwag row transmits only the final fixed-byte decision packet. This
supports a byte/exposure systems claim now, but not a native throughput claim
until vLLM/SGLang/NVIDIA rows measure TTFT, TPOT, goodput, GPU memory, and
HBM/PCIe-or-NVLink traffic.
