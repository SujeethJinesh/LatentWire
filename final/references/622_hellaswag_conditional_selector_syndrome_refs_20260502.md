# References: HellaSwag Conditional Selector/Syndrome Gate

This memo supports
`results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502/`.
The artifact is a negative train-only method gate for fixed-byte
source-private conditional selector/syndrome packets.

## Claim Boundary

Safe claim: a linear benefit predictor over compact source packet codes,
source-side confidence bins, and Qwen receiver-side score features fails to
recover the measured TinyLlama/Qwen oracle headroom.

Unsafe claim: this rules out nonlinear connectors, C2C-style KV fusion, learned
query resamplers, sparse dictionaries, or future fixed-byte task syndromes.

## Primary Sources

1. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: C2C is the closest model-to-model communication prior because
     it projects and fuses source KV cache into the receiver. Boundary:
     LatentWire transmits no source KV/cache and must not claim C2C-quality or
     native latency wins without native baselines.

2. KVComm and KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: these compare against selective KV sharing and online
     cross-context KV reuse. Boundary: this gate sends only a discrete packet,
     not KV pairs, cache offsets, or hidden vectors.

3. BLIP-2 / Q-Former, Perceiver IO, and Flamingo
   - https://arxiv.org/abs/2301.12597
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - Relevance: learned query/resampler bottlenecks are the right nonlinear
     connector prior if we leave linear selectors. Boundary: continuous query
     states or target-side cross-attention are not the fixed-byte packet
     method unless quantized to a discrete syndrome and decoded without source
     state exposure.

4. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Relevance: any target-injected continuous prefix or soft prompt is prior
     art. Boundary: this gate uses no soft prompt, prompt vector, adapter, or
     continuous target injection.

5. Sparse Autoencoders and cross-model dictionaries
   - https://arxiv.org/abs/2309.08600
   - https://arxiv.org/abs/2502.03714
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: sparse/common dictionaries are the natural next basis family.
     Boundary: transmitting dictionary activations would become sparse latent
     communication; LatentWire's safer novelty is private dictionary
     computation used to emit a tiny task-level syndrome.

6. QJL and TurboQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - Relevance: these define strong KV/vector compression byte floors.
     Boundary: LatentWire's packet is a decision/syndrome code, not a vector
     fidelity codec or compressed KV cache.

## Reviewer Risk

Reviewers will reject a claim that `+0.002` eval-only selector lift is a
method contribution. The correct use is negative evidence: scalar/linear
selectors cannot recover a meaningful fraction of oracle headroom, so the
paper must either produce a stronger nonlinear fixed-byte method or cut the
receiver-improvement claim from the ICLR story.
