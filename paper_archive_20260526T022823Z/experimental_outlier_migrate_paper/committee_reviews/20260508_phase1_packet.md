REQUIRES_FIX: yes

# OutlierMigrate Phase 1 Committee Review

## COLM area chair

Scores: novelty 6/10, rigor 6/10, clarity 7/10.

Meta-review: The Phase 1 packet is a real improvement over Phase 0: the same preregistered dynamic-outlier rule passes on Granite-4.0-H-Small with 24 traces, migration fraction 0.843165650406504, and bootstrap 95% CI [0.8334349593495936, 0.8511432926829268], after Phase 0 passed at 0.8178385416666667 with CI [0.797265625, 0.8368489583333334]. Artifact checks are complete for both packets, model snapshots and prompt hashes are recorded, and the paper mostly preserves the correct same-family scope.

Novelty is plausible but still a candidate contribution rather than a paper-ready method. Rigor is adequate for a gate result, not for COLM acceptance: the work has no strict cross-family falsification pair, no independent seed repeat beyond bootstrap resampling, no contamination audit, and no intervention showing that migration-aware handling improves quality, latency, memory, or robustness. Those are future-gate issues, not fixable in the current draft.

Fixable current-draft issue: the abstract calls the result a "strong positive-method finding." That should be weakened to "strong same-family observational gate" or equivalent, because the current evidence supports rank migration, not a positive method. Current iteration needs this wording fix before being treated as a clean review packet.

## MLSys reviewer

The systems contribution is not established yet. The result is systems-relevant because static channel protection, routing, quantization, and cache policies can depend on stable activation outliers. However, the packet measures rank migration only; it does not include a migration-aware kernel, cache policy, compression policy, oracle headroom study, or end-to-end performance/quality tradeoff.

Reproducibility is the strongest part of the packet. Phase 0 and Phase 1 both provide checker outputs, artifact completeness records, environment/provenance/prompt/seed/command metadata, metrics, bootstrap CIs, artifact hashes, logs, and activation manifests. The reviewer pack gives exact paths and exact values, and both artifact_check.json files report artifact_complete=true.

Engineering-rigor gaps are mostly future gates. Before an MLSys-level claim, the project needs controls for decode-position reachability, activation-capture semantics, rank-tie behavior, top-1% discretization, per-layer/channel dimensionality, and silent truncation/fallback. It also needs a systems baseline: static top-channel protection versus migration-aware protection, with paired uncertainty and cost accounting. Cross-family validation on non-Granite hybrids is mandatory to avoid optimizing around one architecture family.

Fixable current-draft issue: do not frame this as a systems method yet. The paper already says no efficiency claim is established, but the abstract wording should match that restraint.

## Adversarial reviewer

The central risk is overclaiming. The observed values are far above the 0.05 preregistered threshold, but that does not by itself prove a meaningful mechanism; it could reflect ordinary rank churn under a loose metric. Missing controls include random-channel drift, adjacent-position drift, same-position rerun stability, shuffled-trace nulls, layer-stratified null baselines, and sensitivity to the rank-delta threshold/top-channel fraction. These require future gates unless already present in unreported artifacts.

The draft is mostly honest about same-family scope, but "positive-method finding" is unjustified. There is no demonstrated intervention, no quality/latency/memory/robustness improvement, no cross-model transfer, and no strict cross-family falsification pair. This is fixable now as a wording issue; the evidence gaps themselves require future validation/intervention gates.

P-hacking risk is controlled by the preregistered Phase 0/1 surfaces, but future work must not retune positions, thresholds, model choices, or prompt slices after seeing these results. The reviewer pack correctly warns against that. Statistical risk remains: bootstrap CIs over 12 and 24 deterministic traces are useful for the gate but not enough for a submission-level claim.

Citation risk is moderate. The related-work sources appear real from primary/official sources, but the Qwen3.6 and Kimi/Qwen architectural claims should remain motivational only unless the cited model cards/papers are hand-audited and the models are measured. No hallucinated empirical citation is acceptable.
