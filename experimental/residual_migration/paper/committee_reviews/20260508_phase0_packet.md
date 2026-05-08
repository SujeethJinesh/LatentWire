# Residual Migration Phase 0 Committee Review

## 1. COLM Area Chair

Novelty: 4/10. Rigor: 5/10. Clarity: 7/10.

The packet is clear about its limited status: `PASS_RM_PHASE0_RETHINKING_REPLICATES` is a mechanical checker pass, not a positive-method result. The draft correctly states that baseline accuracy is 0.0, ablation accuracy is 0.0, accuracy drop is 0.0, and the bootstrap 95% CI is [0.0, 0.0]. That transparency prevents overclaiming, but it also means the result has almost no evidential force for capability preservation. With no baseline successes, the evaluation has a severe headroom limitation: clipping cannot be shown harmless when the unablated model already fails every prompt.

The preregistration is useful and the artifact packet is complete, but the current evidence is too narrow for COLM readiness: one tiny 12-trace slice, one model, no positive accuracy headroom, no oracle diagnostic, no cross-family falsification, and no scale or seed stability. As a gate report, this is acceptable. As a paper contribution, it is preliminary.

REQUIRES_FIX: no. The current draft already identifies the headroom failure and does not appear to claim capability preservation. Remaining issues are future-gate requirements.

## 2. MLSys Reviewer

The systems contribution is not yet established. The packet documents an intervention mechanism: every discovered transformer layer gets a forward pre-hook that clips residual values above the per-layer/per-token-position 95th percentile while preserving sign. That is a concrete, inspectable systems operation, but it is not yet a deployable compression, latency, memory, routing, or serving contribution.

Reproducibility is the strongest part of the packet. The artifact checker reports `artifact_complete: true`; required files include environment, model provenance, prompt manifest, command metadata, random seed, ablation config, generations, metrics, bootstrap CI, hashes, logs, and events. The draft and reviewer pack identify the model snapshot, prompt SHA, generation artifact SHA, frozen 2048-token generation length, trace count, and checker rule. This is good engineering hygiene for a Phase 0 gate.

Engineering rigor is still bounded by evaluation headroom. The checker PASS is mechanical: `ci_high=0.00000000 < 0.015`. Because `baseline_accuracy=0.0`, the result cannot show that residual clipping preserves useful computation. An MLSys claim would need larger runs, nonzero baseline success, paired uncertainty, throughput/memory/latency instrumentation, and failure-mode analysis.

REQUIRES_FIX: no. The present draft labels the result as headroom-limited; systems gaps belong to Phase 1+.

## 3. Adversarial Reviewer

The central risk is that readers may overweight the word PASS. The checker PASS is mechanical, not scientific confirmation. Since `baseline_accuracy=0.0`, `ablation_accuracy=0.0`, and all prompt-level drops are 0.0, the metric is saturated at failure. The CI [0.0, 0.0] is therefore a symptom of no headroom, not proof of robustness.

Unjustified claims would include any statement that residual clipping preserves reasoning, enables residual quantization, supports cross-model communication, or transfers across model families. The current draft mostly avoids those claims and explicitly says it is not paper-ready. Missing controls include a task/model pair with nonzero baseline accuracy, oracle or easier-prompt headroom diagnostics, seed repeats beyond prompt bootstrap, larger frozen slices, strict same-family versus cross-family separation, and latency/bytes controls.

The preregistration helps against p-hacking: threshold, prompt count, decision strings, and forbidden post-hoc changes were specified. However, Phase 0 still has scope-creep risk if a mechanical replicate pass is promoted into a positive-method result. There is also a citation risk: the motivating "Rethinking the Outlier Distribution in Large Language Models" claim should be backed by a real, checked reference before any public paper version leans on it.

REQUIRES_FIX: no. I see no current-draft overclaim requiring immediate correction; the necessary fixes are future experimental gates and citation verification before broader submission.
