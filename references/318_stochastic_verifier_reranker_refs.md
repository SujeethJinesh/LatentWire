# Stochastic Routing, Verifier, and Reranker References

Primary-source memo for turning the current random-route oracle gap into a
positive method. The immediate LatentWire evidence is that stochastic route
generation contains useful answers, but naive majority vote does not select
them. This points to verifier-guided selection, confidence-aware sampling, and
process-style reranking as the next method family.

| Source | Primary links | Transferable mechanism | Concrete LatentWire ablation |
|---|---|---|---|
| **Self-Consistency Improves Chain of Thought Reasoning in Language Models** | [paper](https://arxiv.org/abs/2203.11171) | Multiple reasoning samples can expose correct answers that greedy decoding misses, but the aggregation rule matters. | Treat multiple stochastic route/value masks as candidate generators; compare majority vote, target tie-break, logprob confidence, answer-format confidence, and verifier-selected candidates at fixed sample count. |
| **Large Language Models are Better Reasoners with Self-Verification** | [paper](https://arxiv.org/abs/2212.09561) | Backward verification scores can select among candidate answers and provide interpretable validation evidence. | Add a cheap target-model verifier pass that scores whether each candidate answer satisfies the original GSM prompt; log verifier score, selected seed, and whether selected seed was oracle-correct. |
| **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** | [paper](https://arxiv.org/abs/2408.03314) | Best-of-N is not always compute-optimal; adaptive compute allocation and revisions can dominate raw sampling. | Route only hard examples through multi-mask sampling using target-alone confidence or entropy; keep easy examples at target-alone to hold latency/bytes fixed. |
| **Generative Verifiers: Reward Modeling as Next-Token Prediction** | [paper](https://arxiv.org/abs/2408.15240) | A verifier can be trained/guided as a generative next-token predictor instead of a separate classifier. | Prototype a text-only generative verifier prompt first, then a learned lightweight verifier over candidate telemetry: route entropy, vote margin, normalized-answer agreement, and source/target logprob features. |
| **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** | [paper](https://arxiv.org/abs/2410.08146) | Process-level progress signals can guide search more efficiently than outcome-only ranking. | For multi-step arithmetic outputs, add a process proxy that checks numeric intermediate consistency, final expression validity, and answer extraction stability before selecting a route candidate. |
| **The Lessons of Developing Process Reward Models in Mathematical Reasoning** | [paper](https://arxiv.org/abs/2501.07301) | PRM quality depends on process labels, data efficiency, and how the reward is aggregated during best-of-N. | Keep verifier telemetry decomposed into outcome score, process score, and confidence score so the paper can show which signal selects oracle-correct seeds. |
| **Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning** | [paper](https://arxiv.org/abs/2603.08999) | Sampling should be conditional on confidence, not applied uniformly. | Gate stochastic route ensembles by target-alone entropy, answer extraction confidence, or low vote margin; report compute saved versus uniform three-seed sampling. |

## LatentWire-Ready Ablations

- Candidate generation: one deterministic route, three stochastic masks, five
  stochastic masks, and mixed deterministic plus stochastic masks.
- Selection: majority vote, target tie-break, source/target logprob, verifier
  prompt, numeric consistency, and oracle upper bound.
- Efficiency: always-sample versus confidence-gated sampling at equal average
  bytes and latency.
- Interpretability: selected seed id, vote margin, vote entropy, verifier
  score, candidate agreement, and method-only / baseline-only paired flips.
- Failure slicing: questions where oracle is correct but majority is wrong,
  target-alone is correct but stochastic hurts, and all candidates fail.

## Immediate Paper Claim Shape

The current evidence does not justify claiming random routing as the method.
It does justify a sharper positive-method hypothesis:

1. Stochastic cross-model route perturbations expose a useful answer candidate
   set.
2. Raw vote aggregation is insufficient.
3. A verifier/reranker that selects among route candidates is the missing
   component to test next.

