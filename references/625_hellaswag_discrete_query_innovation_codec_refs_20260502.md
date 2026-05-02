# References: HellaSwag Discrete Query Innovation Codec Gate

This memo supports
`results/source_private_hellaswag_discrete_query_innovation_codec_gate_20260502/`.
The artifact is a negative gate for a decoder-conditioned fixed-query
discrete source-code branch.

## Claim Boundary

Safe claim: a train-only source encoder can be calibrated against target-side
residuals while inference transmits only a one-byte discrete task code, but
this particular cached TinyLlama-score implementation does not improve
HellaSwag validation accuracy over packet-only.

Unsafe claim: this rules out cross-model latent communication, true hidden
connectors, C2C/KVComm-style cache communication, shared SAE/crosscoder
dictionaries, or native systems wins.

## Primary Sources

1. HellaSwag
   - https://arxiv.org/abs/1905.07830
   - Relevance: benchmark under study. Boundary: this is not a new HellaSwag
     state-of-the-art claim.

2. BLIP-2 / Q-Former, Perceiver IO, and Flamingo
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - Relevance: learned query bottlenecks and resamplers are the closest
     architectural inspiration. Boundary: LatentWire does not transmit query
     embeddings or continuous resampler latents in this gate.

3. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Relevance: continuous virtual tokens are a mandatory comparison if a
     future branch injects learned vectors into the target. Boundary: this gate
     emits a discrete task code, not a continuous prompt.

4. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: closest direct semantic communication competitor; C2C projects
     and fuses source KV cache with target KV cache. Boundary: this gate sends
     no source KV/cache and makes no native latency claim.

5. KVComm and KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: required inter-LLM systems competitors. Boundary: they share,
     reuse, or align KV/cache state; this gate transmits a one-byte task code.

6. Sparse autoencoders and crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: shared sparse dictionaries are a plausible common-basis route.
     Boundary: transmitting sparse activations or atom weights would be a
     different latent-vector communication method.

7. QJL, TurboQuant, KIVI, and KVQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Relevance: rate-distortion and systems byte-floor baselines for KV/vector
     state. Boundary: this gate optimizes conditional task decision utility,
     not vector reconstruction or attention inner-product fidelity.

8. Vector quantization and product quantization
   - https://arxiv.org/abs/1711.00937
   - https://ieeexplore.ieee.org/document/5432202
   - Relevance: the codebook step is VQ-style. Boundary: the gate is
     decoder-conditioned and task-level, not an image/token reconstruction VQ
     objective.

9. Distributed source coding / side information
   - https://ieeexplore.ieee.org/document/1055039
   - https://ieeexplore.ieee.org/document/1055037
   - Relevance: the target model's candidate scores act like decoder-side side
     information. Boundary: the theory is prior art; LatentWire's contribution
     would need to be the LLM endpoint protocol plus empirical systems/eval
     evidence.

10. vLLM / PagedAttention and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://github.com/sgl-project/sglang
    - Relevance: native serving baselines for future TTFT, TPOT, goodput,
      memory, and traffic measurements. Boundary: this artifact reports only
      cached Mac microbenchmarks and byte floors.

## Reviewer-Safe Comparison

- Unlike C2C or KVComm, the gate does not transmit source KV/cache or raw
  hidden state.
- Unlike prefix tuning, Q-Former, Perceiver, or Flamingo-style connectors, the
  transmitted object is not a continuous prompt/query latent.
- Unlike SAE/crosscoder or QJL/TurboQuant/KV-quantization methods, the code is
  trained for downstream conditional decision utility rather than latent-vector
  reconstruction.

## Outcome

The branch fails. The default row is `-0.014041` below packet-only and the best
validation scout is only `+0.000199` with negative paired CI low. This weakens
decoder-conditioned source-score codebooks and promotes either a true hidden
connector/PQ branch or a cut of HellaSwag receiver-improvement claims.
