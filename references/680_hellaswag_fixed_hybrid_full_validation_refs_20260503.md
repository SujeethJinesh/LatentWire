# HellaSwag Fixed Hybrid Full-Validation References

Date: 2026-05-03

## Why This Memo Exists

The fixed hybrid packet now passes over the full cached HellaSwag validation
range `0:10042`. This memo records the citation and novelty boundary: the row
is strong full-validation packet-policy evidence, not learned latent receiver
fusion.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for the full-validation packet-policy gate.
   - Boundary: HellaSwag is a multiple-choice commonsense completion benchmark;
     full-validation success here does not by itself prove open-ended
     reasoning transfer.

2. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: option-order and selector-bias warning.
   - Boundary: the paper should include option/candidate controls before
     broadening candidate-id packet claims into general reasoning claims.

3. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: closest direct inter-LLM communication baseline.
   - Boundary: C2C projects and fuses source KV/cache state. The fixed hybrid
     packet transfers only one candidate id and therefore has a much lower
     exposure/rate profile but less expressive communication.

4. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective source-KV sharing comparator.
   - Boundary: KVComm is a KV-state communication baseline, not a fixed-byte
     candidate packet. Native systems rows are still needed for fair systems
     claims.

5. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: learned continuous prefix baseline.
   - Boundary: the fixed hybrid row uses no learned target prefix and should
     not be described as prompt/prefix tuning.

6. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress visible prompt context; this gate transmits
     a source-private decision packet.

7. Relative Representations Enable Zero-Shot Latent Space Communication
   - Link: https://openreview.net/forum?id=SrC-nwieGJ
   - Role: common-basis latent communication prior.
   - Boundary: the current full-validation row does not construct relative
     coordinates or a learned common basis.

8. Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language
   Models
   - Link: https://arxiv.org/abs/2410.06981
   - Role: motivation for future common-feature communication.
   - Boundary: SAE universality is future motivation; the fixed hybrid packet
     does not expose or align SAE features.

9. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: low-bit KV/vector compression floor.
   - Boundary: QJL pressures vector/KV communication variants, not a
     one-candidate discrete packet.

10. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: quantization and systems-compression comparator.
    - Boundary: TurboQuant is relevant to vector-state transport and native
      systems baselines; it does not replace the current fixed-byte packet
      evidence.

## Decision Boundary

This gate should be cited as:

```text
full-validation evidence for a fixed-byte source-private packet policy
```

It should not be cited as:

```text
learned receiver fusion
common latent basis
soft-prefix/gist-token compression
native systems speedup
general model-to-model reasoning
```

The next method branch should use the remaining candidate/hybrid and
target-or-packet oracle gaps to motivate a target-loss score-simplex receiver
or stronger common-basis mechanism.
