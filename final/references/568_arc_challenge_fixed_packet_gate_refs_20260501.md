# ARC-Challenge Fixed-Packet Gate References, 2026-05-01

## Primary Sources

- Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2
  Reasoning Challenge." arXiv:1803.05457.
  https://arxiv.org/abs/1803.05457
  - ARC-Challenge is an established public science QA benchmark. The Challenge
    subset was designed from questions missed by retrieval and co-occurrence
    baselines, making it a useful first public transfer gate rather than a toy
    slice.

- Hugging Face `allenai/ai2_arc`, `ARC-Challenge`.
  https://huggingface.co/datasets/allenai/ai2_arc
  - Local artifact materializes the official train/validation/test split into
    `1119/299/1172` canonical rows.

- C2C / Cache-to-Cache, OpenReview ICLR 2026.
  https://openreview.net/forum?id=LeatkxrBCi
  - Closest direct communication competitor. It sends/fuses KV-cache state,
    while this gate sends a fixed `12B` source-private candidate residual
    packet. Do not claim native superiority over C2C until native systems rows
    exist.

- vLLM benchmark serve metrics.
  https://docs.vllm.ai/en/v0.9.1/api/vllm/benchmarks/serve.html
  - Defines the serving metrics to collect next: TTFT, TPOT, ITL, E2E latency,
    and goodput under SLO.

- TurboQuant, "Online Vector Quantization with Near-optimal Distortion Rate."
  arXiv:2504.19874.
  https://arxiv.org/abs/2504.19874
  - Relevant systems-side comparison for quantized vector/KV communication.
    TurboQuant operates on high-dimensional vectors/KV cache; the ARC gate uses
    a fixed byte-scale packet with candidate-side information.

## Local Result Boundary

- Validation:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/`
- Test:
  `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/`

The result is positive but narrow. The source scorer is local
Qwen2.5-0.5B choice-text log-likelihood; the packet is a BGE candidate-residual
random-projection sketch. The claim should be framed as public ARC transfer for
a fixed-byte packet protocol, not as proof of universal latent reasoning or
native cache-transfer superiority.
