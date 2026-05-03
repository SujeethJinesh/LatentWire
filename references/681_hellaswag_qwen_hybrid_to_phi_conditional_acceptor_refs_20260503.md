# HellaSwag Qwen Hybrid-To-Phi Conditional Acceptor References

Date: 2026-05-03

## Why This Memo Exists

The Qwen-hybrid-to-Phi conditional acceptor gate failed. This memo records the
novelty boundary: the attempted receiver is a selective/defer-style target-side
override under a source-private fixed-byte packet contract, not a new
selective-classification method or a learned latent communication interface.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark for the cached Qwen-to-Phi receiver-family gate.
   - Boundary: HellaSwag is a multiple-choice commonsense benchmark; success
     or failure here remains candidate-choice scoped until broader benchmarks
     and option-order controls are added.

2. Selective Classification for Deep Neural Networks
   - Link: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
   - Role: risk/coverage selective-prediction baseline.
   - Boundary: the conditional acceptor is not novel as selective prediction;
     any novelty must come from the fixed-byte source-private communication
     contract and destructive controls.

3. Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer
   - Link: https://arxiv.org/abs/1711.06664
   - Role: defer-to-expert framing for choosing between a model decision and an
     external decision source.
   - Boundary: Phi-vs-Qwen override is a special defer/router case. This gate
     fails, so it should be cited as a negative control, not a contribution.

4. Large Language Models Are Not Robust Multiple Choice Selectors
   - Link: https://arxiv.org/abs/2309.03882
   - Role: reviewer-facing warning about option-position and MCQ selector
     artifacts.
   - Boundary: HellaSwag candidate-id packets need option-order or
     candidate-permutation controls before broad reasoning claims.

5. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://openreview.net/forum?id=LeatkxrBCi
   - Role: closest high-rate inter-LLM communication baseline.
   - Boundary: C2C projects/fuses source KV-cache state. The failed acceptor
     transmits only a fixed Qwen packet and uses Phi-side scores locally.

6. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://openreview.net/forum?id=F7rUng23nw
   - Role: selective KV-sharing comparator.
   - Boundary: KVComm transfers selected KV state. This gate sends no KV state
     and cannot support native serving claims without GPU systems rows.

7. Relative Representations Enable Zero-Shot Latent Space Communication
   - Link: https://openreview.net/forum?id=SrC-nwieGJ
   - Role: common-basis latent-communication prior.
   - Boundary: the conditional acceptor does not build relative coordinates or
     align latent spaces. Future common-basis work must beat packet-only.

8. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: learned continuous-prefix baseline.
   - Boundary: this gate learns no target prefix and does not optimize
     continuous soft tokens.

9. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress visible prompt context; LatentWire's
     current HellaSwag row transmits a source-private candidate packet.

10. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: low-bit KV/vector compression floor.
    - Boundary: QJL matters for future vector/KV transport variants and native
      systems baselines, not this one-candidate packet acceptor.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: quantization and systems-compression comparator.
    - Boundary: TurboQuant pressures any future latent/vector communication
      claim; it does not explain or replace the failed score-rule acceptor.

## Decision Boundary

This gate should be cited as:

```text
negative target-score conditional acceptor evidence under a source-private
fixed-byte packet contract
```

It should not be cited as:

```text
learned common-basis communication
soft-prefix or gist-token compression
new selective classification
native systems speedup
positive target-side receiver fusion
```

The next method branch should either introduce a new receiver feature source
or move to common-basis / conditional-innovation methods that are evaluated
against the fixed hybrid packet, not just against target-only.
