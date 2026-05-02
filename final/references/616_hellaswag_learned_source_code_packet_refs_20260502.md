# References: HellaSwag Learned Source-Code Packet Gate

## Claim Boundary

This memo supports the failed learned source-code packet gate. The safe claim
is that source-score-derived one-byte discrete codes were evaluated under
official-train calibration and did not beat compact packet-only. It does not
claim solved learned source coding, cross-model latent language, prefix-token
conditioning, or KV/cache compression.

## Primary Sources And Why They Matter

1. Wyner-Ziv coding with decoder side information
   - https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
   - Boundary: motivates source coding with receiver-side information; this
     artifact does not introduce new information theory.

2. Neural Distributed Source Coding
   - https://arxiv.org/abs/2106.02797
   - Boundary: learned distributed source coding is prior work. Our artifact is
     a constrained LLM task-packet gate, not a first learned DSC method.

3. VQ-VAE / Neural Discrete Representation Learning
   - https://arxiv.org/abs/1711.00937
   - Boundary: discrete latent codebooks are not novel. The tested novelty is
     whether a tiny source-private task code helps cross-model decoding.

4. Relative Representations
   - https://arxiv.org/abs/2209.15430
   - Boundary: relative/anchor bases are prior latent-communication tools. This
     gate does not claim a shared latent basis.

5. Sparse autoencoders and crosscoders
   - https://arxiv.org/abs/2410.06981
   - https://arxiv.org/abs/2502.03714
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Boundary: SAE/crosscoder common-basis methods remain future comparators.
     The present code is source-score-derived, not an SAE feature language.

6. Prefix-Tuning and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2110.07602
   - Boundary: prefix/prompt tuning learns persistent continuous virtual
     tokens. The learned source-code packet is a per-example discrete byte.

7. C2C and KV communication
   - https://arxiv.org/abs/2510.03215
   - https://openreview.net/forum?id=LeatkxrBCi
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - https://arxiv.org/abs/2512.17914
   - Boundary: these methods communicate through KV/cache states. This artifact
     sends no source KV/cache state and cannot claim native systems superiority.

8. QJL, TurboQuant, KIVI, and KVQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Boundary: these are vector/KV compression methods. The learned source code
     is a task packet, not a vector-fidelity codec.

9. Latent diffusion and DiT
   - https://arxiv.org/abs/2112.10752
   - https://arxiv.org/abs/2212.09748
   - Boundary: iterative denoising remains a possible future receiver idea. No
     diffusion-style denoiser is implemented in this gate.

10. vLLM / PagedAttention and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md
    - Boundary: native serving and HBM claims remain disabled until NVIDIA rows
      are collected.

## Reviewer-Facing Framing

Safe:

- We tested a learned one-byte discrete source-code packet family on full
  HellaSwag validation.
- The best diagnostic learned code improves by only `+0.000697` over
  packet-only and has CI crossing zero.
- The train-dev-selected learned code is worse than packet-only.
- This rules out source-score quantile/k-means subcodes as the missing ICLR
  method on this surface.

Unsafe:

- Claiming learned source-code communication works.
- Claiming a shared latent language or SAE/crosscoder basis.
- Claiming equivalence to prefix tokens.
- Claiming lower TTFT/TPOT/HBM traffic or superiority over C2C/KVComm/KV
  quantization.
