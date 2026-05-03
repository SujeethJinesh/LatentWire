# HellaSwag Strict Channel-Selector References

Date: 2026-05-03

## Why This Memo Exists

The strict channel-selector gate found a narrow positive result: a fixed
hybrid vote-on-score-agreement packet policy beats the `1B` candidate-only
packet on Qwen HellaSwag validation `0:9216`. This memo records the novelty
boundary so the paper does not overclaim it as a learned latent-receiver method.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: adversarial multiple-choice commonsense benchmark used for the
     strict packet-policy surface.
   - Boundary: a candidate-policy gain on HellaSwag does not by itself prove
     open-ended reasoning or general model-to-model latent communication.

2. Adaptive Mixtures of Local Experts
   - Link: https://doi.org/10.1162/neco.1991.3.1.79
   - Role: classic routed-expert framing for choosing among specialist
     predictions.
   - Boundary: the fixed hybrid policy is not a learned MoE router; it is a
     hand-specified source-side decision rule over cached packet channels.

3. SelectiveNet: A Deep Neural Network with an Integrated Reject Option
   - Link: https://arxiv.org/abs/1901.09192
   - Role: selective prediction / reject-option reference for learning when to
     trust or abstain from a predictor.
   - Boundary: our learned selectors do not pass; the positive row is static
     and should not be claimed as a new selective-prediction architecture.

4. Selective Classification for Deep Neural Networks
   - Link: https://arxiv.org/abs/1705.08500
   - Role: statistical framing for risk/coverage and confidence-based
     acceptance.
   - Boundary: the HellaSwag hybrid policy does not abstain or optimize
     coverage. It emits exactly one candidate id for every row.

5. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: frozen-model continuous conditioning baseline.
   - Boundary: this result sends a discrete candidate id chosen by a
     source-side policy. It is not learned continuous prefix conditioning.

6. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt-compression baseline.
   - Boundary: gist tokens compress prompt context for a model; the hybrid
     packet policy compresses source-side task evidence into one candidate id.

7. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Role: direct KV-cache communication baseline.
   - Boundary: C2C sends/fuses source KV state. The hybrid packet policy sends
     only a final candidate id to the receiver, with much lower exposure but
     far less expressivity.

8. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective KV-sharing systems baseline for inter-LLM communication.
   - Boundary: KVComm is a state-transfer method. Our row needs byte/privacy
     and native serving comparisons, not a claim that candidate packets replace
     selective KV sharing in general.

9. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: low-bit KV/vector compression baseline.
   - Boundary: QJL preserves approximate inner products/attention geometry;
     our hybrid policy is symbolic task-evidence selection.

10. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: modern systems/quantization pressure on vector or KV-transfer
      claims.
    - Boundary: TurboQuant does not preempt a source-private discrete packet,
      but any future vector-state row must compare against it.

## Decision Boundary

The strict channel-selector gate should be cited as a narrow positive
packet-policy row:

```text
fixed hybrid packet policy > candidate-only packet >> score-only controls
```

It should not be cited as a learned receiver, prefix-token method, MoE router,
or universal latent basis. The next ICLR-grade step is to make this policy
survive cross-family pressure or beat it with a genuine learned receiver /
common-basis method.
