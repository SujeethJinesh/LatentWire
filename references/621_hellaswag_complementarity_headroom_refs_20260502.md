# References: HellaSwag Complementarity Headroom Gate

This memo supports
`results/source_private_hellaswag_complementarity_headroom_gate_20260502/`.
The artifact is a cache-only oracle/headroom diagnostic, not a new learned
communication method.

## Claim Boundary

Safe claim: TinyLlama's compact source-private packet and Qwen's local
target-side prediction have complementary HellaSwag errors, creating a stable
oracle target for a future conditional syndrome/selector packet.

Unsafe claim: the gate solves latent communication, beats C2C/KVComm, or
demonstrates a nonlinear connector.

## Primary Sources

1. Cache-to-Cache
   - https://arxiv.org/abs/2510.03215
   - Relevance: C2C motivates direct model-to-model communication and shows
     that a learned fusion policy can exploit model complementarity. Boundary:
     C2C fuses KV caches; this gate only measures oracle headroom for a
     fixed-byte packet.

2. BLIP-2 / Q-Former
   - https://arxiv.org/abs/2301.12597
   - Relevance: learned querying transformers are the natural nonlinear
     connector baseline if we move beyond byte packets. Boundary: Q-Former
     transmits continuous query states, not a fixed-byte source-private packet.

3. Perceiver IO
   - https://arxiv.org/abs/2107.14795
   - Relevance: learned latent queries over arbitrary inputs motivate
     resampler-style connectors. Boundary: LatentWire's next method must emit
     a discrete packet or be framed as a different baseline class.

4. Prefix-Tuning
   - https://aclanthology.org/2021.acl-long.353/
   - Relevance: soft prefixes are a mandatory comparator for any target-side
     injection method. Boundary: this headroom gate uses no learned prompt or
     continuous target injection.

5. KVComm / KVCOMM
   - https://arxiv.org/abs/2510.03346
   - https://arxiv.org/abs/2510.12872
   - Relevance: these are native systems/cache communication comparators.
     Boundary: the next LatentWire gate should preserve no-source-KV and
     fixed-byte packet constraints unless it is explicitly a baseline.

## Next-Method Requirement

A conditional syndrome/selector packet should be promoted only if it beats the
source packet, source-label-copy, same-byte text, target-only, and
candidate-id-only baselines with positive paired uncertainty, while
row-shuffle, wrong-example, target-derived, random same-rate, and
label-permutation controls collapse.
