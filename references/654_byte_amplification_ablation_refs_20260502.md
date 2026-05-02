# Byte-Amplification Ablation References

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper
  remains blocked by method-side matched-source evidence.
- Current story: systems contribution is a fixed-byte source-private packet
  interface, not a compressed KV/cache transfer claim.
- Exact gap: native NVIDIA serving rows are still pending; byte floors are not
  native quality or throughput baselines.

## Primary Sources

1. Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
   Information at the Decoder." <https://ieeexplore.ieee.org/document/1055039>
   - Boundary: LatentWire packets are best described as task-level side
     information decoded with target-side public context, not as raw state
     transport.

2. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead. <https://arxiv.org/abs/2406.03482>
   - Boundary: QJL is a KV-cache/state sketch. The ablation uses a one-token
     1-bit K+V floor of `6144 / 8 = 768B` from the local Qwen2.5-0.5B source
     config. This is a byte floor, not a native quality result.

3. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://arxiv.org/abs/2504.19874>
   - Boundary: TurboQuant is a vector/KV quantization comparator. The ablation
     uses a `3.5` bits/element one-token K+V floor, but does not claim native
     TurboQuant latency or quality.

4. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   <https://arxiv.org/abs/2402.02750>
   - Boundary: KIVI is a 2-bit KV-cache compression method. It remains
     KV-size-linear and source-state exposing.

5. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization. <https://arxiv.org/abs/2401.18079>
   - Boundary: KVQuant is a sub-4-bit cached-activation compression baseline.
     It should be compared as state compression, not as a source-private
     packet.

6. Efficient Memory Management for Large Language Model Serving with
   PagedAttention. <https://arxiv.org/abs/2309.06180>
   - Boundary: vLLM/PagedAttention is the serving-memory substrate for native
     TTFT/TPOT/goodput rows; Mac-local byte accounting cannot close this gate.

7. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models. <https://arxiv.org/abs/2510.03215>
   - Boundary: C2C projects and fuses source KV cache. LatentWire must
     distinguish fixed task packets from transported/fused state.

8. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Boundary: KVComm shares selected KV layers/pairs. Its byte floor still
     scales with selected KV elements.

9. KVCOMM: Online Cross-context KV-cache Communication for Efficient
   LLM-based Multi-agent Systems. <https://arxiv.org/abs/2510.12872>
   - Boundary: cross-context KV reuse is a systems neighbor but not a
     fixed-byte source-private task packet.

10. Interlat: Enabling Agents to Communicate Entirely in Latent Space.
    <https://arxiv.org/abs/2511.09149>
    - Boundary: latent-state communication is adjacent, but Interlat-style
      hidden-state transfer differs from answer-key-forbidden fixed-byte
      packet exchange.

11. Sparse Autoencoders Find Highly Interpretable Features in Language Models.
    <https://arxiv.org/abs/2309.08600>
    - Boundary: SAE features may help interpret a future positive receiver,
      but they are not a systems byte-floor baseline for this ablation.

12. Relative Representations Enable Zero-Shot Latent Space Communication.
    <https://arxiv.org/abs/2209.15430>
    - Boundary: relative representations motivate common-coordinate latent
      transfer, but they are not a measured packet systems comparator here.

## Byte-Floor Formula

For one source token's K+V state, the ablation uses:

`bytes = kv_elements_per_source_token * bits_per_element / 8`

For the local Qwen2.5-0.5B source config:

- `kv_elements_per_source_token = 6144`;
- QJL 1-bit floor: `768B`;
- KIVI 2-bit floor: `1536B`;
- KVQuant 3-bit proxy floor: `2304B`;
- TurboQuant 3.5-bit floor: `2688B`;
- fp16 C2C/KV floor: `12288B`.

The source-score stress row is intentionally separate:

`4 choices * 2 fp16 bytes = 8B`

That row is byte-small, but it exposes raw source scores and is not
source-private.

## Experiment Implication

The ablation should be cited as systems/interface evidence only. It supports a
claim that fixed source-private packets live in a different byte/exposure
regime than compressed KV or hidden-state transport. It does not support a
native GPU throughput or baseline-defeat claim.
