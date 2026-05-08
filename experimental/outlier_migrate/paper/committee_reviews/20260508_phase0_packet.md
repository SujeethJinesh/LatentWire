# OutlierMigrate Phase 0 Committee Review

## Paper Readiness

- current paper readiness: not COLM/ICLR-ready; useful Phase 0 evidence only.
- current story: high-magnitude decode-time activation channels in Granite-4.0-H-Tiny appear strongly non-stationary under the preregistered top-1% rank-migration metric.
- blocking gap: Phase 1 must show same-family scale stability and later evidence must show cross-family behavior and a useful migration-aware intervention before this can become a positive-method submission.

## Member A: COLM Area Chair Meta-Review

Scores: novelty 6/10, rigor 7/10, clarity 8/10.

The Phase 0 update is honest and usable for the next gate. The draft makes a narrow, preregistered claim: on 12 deterministic AIME-2025 traces for `ibm-granite/granite-4.0-h-tiny`, the migration fraction is 0.8178385416666667 with bootstrap 95% CI [0.797265625, 0.8368489583333334], clearing the preregistered dynamic threshold. It does not present this as camera-ready or as a validated method, which is the right posture.

Novelty is plausible but not yet submission-strength. "Outliers move during long decode" could motivate a method, but the current artifact is still an observation and does not yet explain mechanism, task dependence, model-family dependence, or utility. Rigor is good for a Phase 0 screen because the decision rule was frozen and the checker result is directly cited. Rigor is not paper-level because the sample is small, the uncertainty is bootstrap over only 12 traces, and there is no same-family scale validation or cross-family falsification.

The biggest non-fixable missing evidence is Phase 1: larger frozen surface, model-size scaling, and controls that separate true migration from metric artifacts. Fixable now: the draft should eventually state why a rank shift of more than two channel positions is meaningful rather than merely preregistered. This can wait until Phase 1, but the current text should not imply practical impact.

## Member B: MLSys Reviewer

The systems contribution is currently prospective, not established. The result is relevant to systems because static outlier protection, routing, quantization, or cache policies often assume stable high-magnitude channels. The packet challenges that assumption on one hybrid model, but it does not yet show a systems mechanism, speedup, memory reduction, quality retention, or hardware-facing implementation.

Reproducibility is relatively strong for Phase 0. The paper and reviewer pack point to the preregistration, checker output, metrics, artifact completeness, model provenance, prompt manifest, command metadata, seed file, and activation artifact. The model snapshot commit is recorded. This is the right engineering shape for a gate result.

The engineering-rigor concern is that the current metric may be sensitive to implementation details: activation capture point, hidden-state normalization, whether ranks are computed per trace or mean magnitude, channel ties, and top-1% discretization for hidden size 1536. These are not fatal for a preregistered screen, but they become mandatory controls before making systems claims. Phase 1 should include artifact checks that verify decode positions were actually reached for every trace, exact layer/channel dimensions, deterministic prompt mapping, and no silent truncation or fallback path.

I would not accept this as an MLSys systems paper yet. I would accept it as a useful gate update that justifies the next experiment. The next packet needs a larger same-family model and an intervention or oracle diagnostic showing headroom for migration-aware handling.

## Member C: Adversarial Review

The draft is commendably restrained, but several risks remain. First, the threshold is extremely low relative to the observed value: 5% preregistered versus 81.78385416666667% observed. That is not p-hacking by itself because the threshold was frozen, but it makes the scientific interpretation depend heavily on whether the metric is meaningful. A huge pass could indicate true migration, or it could indicate that the rank metric is unstable for top-1% channels over long decode.

Second, the result should not be allowed to mutate into a positive-method claim. There is no evidence that protecting migrating channels helps quality, latency, memory, robustness, or cross-model transfer. Any language implying "OutlierMigrate works" would be overclaiming. The current draft mostly avoids that.

Third, the packet has no negative controls visible in the paper: random-channel rank drift, adjacent-position drift, same-position rerun stability, per-layer null baselines, or shuffled trace controls. Without these, reviewers cannot tell whether 0.8178 is surprising relative to ordinary rank churn. This is missing future evidence, not a fixable text issue unless the artifacts already contain such controls.

Fourth, citation risk is low because the draft cites almost no literature, but this also means novelty is unsupported. Do not add broad claims about quantization/outlier prior work without checking primary sources.

Verdict: usable as an honest Phase 0 packet. Not camera-ready. No prereg amendment is required unless the next step changes thresholds, positions, model family, or prompt surface after seeing this result.

## Fixes Required Before Next Commit

None for this committee-review-only commit. The current draft is not camera-ready, but its limitations are explicit enough for a Phase 0 gate packet. The required next work is new Phase 1 evidence, not paper-text cleanup.
