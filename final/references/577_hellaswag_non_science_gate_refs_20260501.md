# References: HellaSwag Non-Science Gate

Date: 2026-05-01

## Primary Benchmark

- HellaSwag: https://arxiv.org/abs/1905.07830
  - Relevance: four-choice adversarial commonsense sentence-completion
    benchmark used for the new non-science positive slice.
  - Boundary: the current result is on a frozen `1024`-row validation slice;
    full validation remains required.

- HellaSwag dataset card/files: https://huggingface.co/datasets/Rowan/hellaswag
  - Relevance: local bridge materializes the public train/validation splits and
    confirms that public test labels are unavailable/empty.
  - Boundary: source packets must not use `label`, `activity_label`,
    `source_id`, `split`, `split_type`, or `ind`.

- CommonsenseQA: https://arxiv.org/abs/1811.00937
  - Relevance: preceding non-science diagnostic that motivated HellaSwag after
    same-byte text saturated short answer labels.
  - Boundary: CommonsenseQA remains diagnostic rather than a strict headline
    positive.

## Communication and Systems Comparators

- Cache-to-Cache (C2C): https://openreview.net/forum?id=LeatkxrBCi
  - Relevance: closest direct model-to-model communication framing, using
    projected/fused KV-cache state.
  - Boundary: C2C communicates dense internal cache state; LatentWire's current
    HellaSwag result sends a `2B` task packet plus public candidate side
    information and exposes no source KV cache.

- KVComm: https://openreview.net/forum?id=F7rUng23nw
  - Relevance: selective KV sharing for efficient LLM communication.
  - Boundary: KVComm is a KV-cache sharing/compression family; the HellaSwag
    row is an extreme-rate decision packet, not a cache reuse method.

- QJL: https://arxiv.org/abs/2406.03482
  - Relevance: Johnson-Lindenstrauss plus sign-bit quantization is relevant for
    future public-basis and distribution-sketch packet designs.
  - Boundary: QJL is a vector/KV quantization method; this result does not
    claim KV-cache compression.

- TurboQuant: https://arxiv.org/abs/2504.19874
  - Relevance: online vector quantization and QJL residual correction are useful
    systems inspiration for byte-efficient latent/vector packet ISAs.
  - Boundary: TurboQuant targets high-dimensional vector/KV compression; the
    current HellaSwag packet is a few-byte source-private benchmark protocol.

- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  - Relevance: defines the native serving systems context for future TTFT,
    TPOT, throughput/goodput, and KV-memory measurements.
  - Boundary: the current run is Mac-local Python/NumPy/PyTorch evidence only.

- LLMLingua: https://arxiv.org/abs/2310.05736
  - Relevance: prompt/text compression family that motivates same-byte text
    relay controls.
  - Boundary: HellaSwag's `2B` packet beats the local same-byte text relay, but
    we still need to show where text catches up at larger byte budgets.

## Novelty Boundary

Safe claim if the full HellaSwag validation run holds:

> A source-private, fixed-byte public-basis packet can improve non-science
> commonsense continuation selection over target-only, destructive packet
> controls, and same-byte text relay at `2B` payload / `5B` framed record.

Unsafe claims until more evidence exists:

- general cross-model latent reasoning
- native NVIDIA/vLLM serving acceleration
- superiority to C2C/KVComm/TurboQuant on their own cache-transfer or
  quantization objectives
