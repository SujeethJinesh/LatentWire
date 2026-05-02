# References: HellaSwag PQ Hidden Innovation Codec Gate

This memo supports
`results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/`.
The artifact is a negative gate for a product-quantized source-hidden code
under the one-byte source-private packet contract.

## Claim Boundary

Safe claim: product-quantized TinyLlama hidden residuals can be evaluated under
the same fixed-byte source-private protocol, and this cached HellaSwag slice
does not show a stable improvement over compact packet-only.

Unsafe claim: this rules out cross-model latent communication, continuous
query/cache connectors, SAE/crosscoder dictionaries, or native systems wins.

## Primary Sources

1. HellaSwag
   - https://arxiv.org/abs/1905.07830
   - Relevance: benchmark under study. Boundary: this gate is not a new
     HellaSwag state-of-the-art claim.

2. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: closest direct semantic communication competitor; C2C projects
     and fuses source KV cache into the receiver. Boundary: this gate sends no
     source KV/cache and reports no native latency win.

3. KVComm and KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: required inter-LLM systems baselines for selective sharing or
     reuse/alignment of KV cache. Boundary: this gate transmits a task-level
     one-byte code, not KV pairs or reused cache blocks.

4. TurboQuant, QJL, KIVI, and KVQuant
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Relevance: vector/KV quantization baselines and systems byte floors.
     Boundary: this gate optimizes downstream candidate decision utility, not
     MSE, inner-product fidelity, or cache reconstruction.

5. Product quantization and vector-quantized representation learning
   - https://ieeexplore.ieee.org/document/5432202
   - https://arxiv.org/abs/1711.00937
   - Relevance: the factorized codebook mechanism is PQ/VQ-inspired. Boundary:
     this gate is decoder-conditioned and task-level, not image/token
     reconstruction.

6. Sparse autoencoders and crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: shared sparse dictionaries remain a plausible common-basis
     route. Boundary: the current gate does not train a sparse dictionary or
     transmit sparse activations.

7. BLIP-2 / Q-Former, Perceiver IO, and Flamingo
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - Relevance: learned query bottlenecks and resamplers are the next connector
     family if GPU access is available. Boundary: this gate sends a discrete
     code, not continuous query embeddings.

8. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Relevance: continuous virtual tokens are mandatory baselines for future
     injection methods. Boundary: this artifact has no virtual prompt vector.

9. Distributed source coding with decoder side information
   - https://ieeexplore.ieee.org/document/1055039
   - https://ieeexplore.ieee.org/document/1055037
   - Relevance: the receiver's Qwen scores are decoder-side information.
     Boundary: the theory is prior art; LatentWire must contribute the LLM
     endpoint protocol plus empirical evidence.

10. vLLM / PagedAttention and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://github.com/sgl-project/sglang
    - Relevance: native serving baselines for future TTFT, TPOT, goodput,
      memory, HBM traffic, and cache-transfer measurements. Boundary: this
      artifact is cached Mac-local evaluation only.

## Reviewer-Safe Comparison

- Unlike C2C or KVComm/KVCOMM, the communication object is a one-byte task
  code, not source KV/cache.
- Unlike TurboQuant/QJL/KIVI/KVQuant, the objective is candidate accuracy under
  decoder side information, not vector/cache reconstruction fidelity.
- Unlike prefix or Q-Former-style methods, the receiver never consumes
  continuous prompt/query embeddings.
- Unlike SAE/crosscoder work, the codebook is a shallow product quantizer, not
  an interpretable shared dictionary.

## Outcome

The PQ hidden-code branch fails. The predeclared default is `-0.004883` versus
packet-only with CI95 low `-0.017578`; the best diagnostic scout is
`+0.006836` with CI95 low `0.000000`, below the `+0.010` promotion bar and
below the prior hidden-code near-miss. This should stop Mac-local HellaSwag
hidden-code widening and force either a true learned connector or a new
benchmark surface.
