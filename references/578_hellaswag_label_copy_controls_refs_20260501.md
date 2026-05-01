# References: HellaSwag Label-Copy Controls

Date: 2026-05-01

## Benchmark And Threat Model

- HellaSwag: https://arxiv.org/abs/1905.07830
  - Relevance: the frozen validation slice is drawn from this adversarial
    four-choice commonsense continuation benchmark.
  - Boundary: current evidence is first-1024 validation only, not full
    validation.

- HellaSwag dataset card: https://huggingface.co/datasets/Rowan/hellaswag
  - Relevance: local bridge rematerializes train/validation metadata for
    activity and split-type controls.
  - Boundary: `label`, `activity_label`, `source_id`, `split`, `split_type`,
    and `ind` remain forbidden source-packet inputs.

- LLMLingua: https://arxiv.org/abs/2310.05736
  - Relevance: prompt-compression work motivates stronger text relay baselines
    and warns against weak text controls.
  - Boundary: the new source-label-copy control is a stricter diagnostic than
    same-byte choice-prefix text, but it is not a full prompt-compression
    baseline.

## Communication And Systems Comparators

- Cache-to-Cache (C2C): https://arxiv.org/abs/2510.03215
  - Relevance: direct beyond-text model communication using projected/fused
    source KV cache.
  - Boundary: C2C sends dense internal cache state; the current LatentWire
    branch sends a few-byte task packet and currently fails the top-label-copy
    threat test.

- KVComm: https://arxiv.org/abs/2510.03346
  - Relevance: selective KV sharing is a strong systems comparator for
    inter-LLM communication.
  - Boundary: KVComm communicates selected KV pairs; LatentWire must not claim
    superiority without native matched systems rows.

- KVCOMM: https://arxiv.org/abs/2510.12872
  - Relevance: cross-context KV-cache communication/reuse frames the native
    TTFT and cache-transfer comparison we still need.
  - Boundary: it is a cache-reuse serving method, not an extreme-rate
    source-private decision packet.

- QJL: https://arxiv.org/abs/2406.03482
  - Relevance: JL projection plus sign-bit quantization motivates future
    source-score/vector repair sketches.
  - Boundary: QJL is a KV/vector quantization method; the current score-margin
    packet is a failed HellaSwag diagnostic, not a QJL result.

- TurboQuant: https://arxiv.org/abs/2504.19874
  - Relevance: residual correction after vector quantization suggests a future
    "source top label plus residual repair sketch" packet family.
  - Boundary: TurboQuant targets vector/KV compression; this memo only uses it
    as systems/method inspiration.

- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  - Relevance: native systems rows should report TTFT/TPOT/goodput, KV memory,
    and cache-transfer measurements in a serving stack.
  - Boundary: all current HellaSwag results are Mac-local CPU/Python evidence.

## Decision Boundary

Safe claim after this control:

> The HellaSwag packet result is not explained by metadata/activity leakage, but
> it currently does not beat a source-label-copy text baseline.

Unsafe claim:

> The current HellaSwag packet proves non-label-copy latent communication.
