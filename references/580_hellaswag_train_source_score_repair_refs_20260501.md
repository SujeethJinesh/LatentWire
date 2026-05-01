# References: HellaSwag Train Source-Score Repair Probe

Date: 2026-05-01

## Local Artifact

- `paper/source_private_hellaswag_train_source_score_repair_probe_20260501.md`
- `results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/`

## Primary References

- HellaSwag: Can a Machine Really Finish Your Sentence?
  https://arxiv.org/abs/1905.07830
  Role: adversarial commonsense continuation benchmark and current
  source-label-copy falsification surface.

- Calibrate Before Use: Improving Few-Shot Performance of Language Models.
  https://arxiv.org/abs/2102.09690
  Role: prior evidence that answer priors and prompt-choice biases can distort
  LM scores, motivating train-only calibration controls.

- Large Language Models Are Not Robust Multiple Choice Selectors.
  https://arxiv.org/abs/2309.03882
  Role: motivates option-bias and trained label-copy controls for MCQ scoring.

- Answer-level Calibration for Free-form Multiple Choice Question Answering.
  https://aclanthology.org/2022.acl-long.49/
  Role: additional MCQ calibration evidence that answer-choice biases can be
  corrected separately from context-sensitive scoring.

- When Benchmarks are Targets: Revealing the Sensitivity of Large Language
  Model Leaderboards.
  https://arxiv.org/abs/2402.01781
  Role: motivates answer-order/scoring perturbation checks and hybrid scoring
  baselines when interpreting MCQ gains.

- On Calibration of Modern Neural Networks.
  https://arxiv.org/abs/1706.04599
  Role: baseline temperature/vector calibration reference for future
  source-score repair gates.

- Selective Classification for Deep Neural Networks.
  https://arxiv.org/abs/1705.08500
  Role: frames the repair policy as a selective trust/switch decision over a
  base model's ranked candidates.

- Language Models (Mostly) Know What They Know.
  https://arxiv.org/abs/2207.05221
  Role: supports probing whether model scores contain uncertainty signal while
  warning that calibration can fail under distribution shift.

- Energy-based Out-of-distribution Detection.
  https://arxiv.org/abs/2010.03759
  Role: motivates future source-error features beyond top-1/top-2 margin and
  entropy.

- Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
  Methods.
  https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/
  Role: rank-fusion baseline if future hidden-summary packets combine source
  and receiver candidate ranks.

## Systems / Latent Communication Boundary

- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  https://arxiv.org/abs/2510.03215
  Boundary: C2C projects and fuses KV-cache state. This probe sends only a
  few-byte rank/score-shape packet and does not yet claim semantic latent
  transfer.

- KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
  https://openreview.net/forum?id=F7rUng23nw
  Boundary: selective KV-pair sharing is the native internal-state comparator
  for a future GPU run, not the same byte-rate threat model.

- KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems.
  https://arxiv.org/abs/2510.12872
  Boundary: cross-context KV reuse is a serving-system comparator for repeated
  context processing, not an MCQ repair packet baseline.

- DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
  Serving.
  https://arxiv.org/abs/2411.02820
  Boundary: another cache/state-sharing systems comparator for related-model
  serving, not a same-byte source-private MCQ packet baseline.

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  https://arxiv.org/abs/2504.19874
  Boundary: useful inspiration for random rotations and residual sign sketches,
  but this probe shows scalar score-shape codes are insufficient.

## Claim Boundary

This is a negative diagnostic. It weakens the idea that train-source score
shape alone can fix HellaSwag source top-choice errors. The next live branch
should use source hidden summaries or a richer residual code; otherwise
HellaSwag should be demoted from headline ICLR evidence.
