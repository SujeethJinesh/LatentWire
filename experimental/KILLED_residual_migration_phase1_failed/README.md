# Residual Migration Phase 1: KILL Manifest

Date killed: 2026-05-08

Decision: `KILL_RM_PHASE1_FAILED_AT_SCALE`

Preregistration: `experimental/residual_migration/phase1/preregister_rm_phase1.md`

Run directory: `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z`

Checker: `experimental/residual_migration/phase1/check_rm_phase1.py`

Artifact check: `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/artifact_check.json`

## Decision Summary

Residual Migration Phase 1 is killed because the full residual 95th-percentile
clipping ablation failed to replicate the Phase 0 no-drop result on
`ibm-granite/granite-4.0-h-small`.

Preregistered replicate-path rule:

- PASS if AIME-2025 accuracy drop on Granite-4.0-H-Small has bootstrap 95% CI
  upper bound `< 1.5%`.
- KILL if point drop is `>= 1.5%`.

Observed full-ablation result:

- baseline accuracy: `0.08333333333333333`
- full-ablation accuracy: `0.0`
- accuracy drop: `0.08333333333333333`
- bootstrap 95% CI: `[0.0, 0.20833333333333334]`
- checker reason: `failed at scale: point=0.08333333 >= 0.015`

The kill is mechanical under the preregistered point-drop rule.

## Headroom Diagnostic

The Phase 1 packet has low but nonzero headroom:

- baseline correct count: `2 / 24`
- baseline accuracy: `0.08333333333333333`
- extractor failure count: `3`
- lenient oracle answer mention count: `3 / 24`
- headroom status: `LOW_BASELINE_HEADROOM`

This is enough to make the Phase 1 failure meaningful: both baseline-correct
prompts became incorrect under the full residual clipping ablation.

## Layer-Stratified Attribution

Layer-stratified ablations were attribution-only and did not change the gate
decision. Each stratified ablation showed the same point drop as the full
ablation:

| Ablation set | Accuracy drop |
|---|---:|
| full_ablation | `0.08333333333333333` |
| first_half | `0.08333333333333333` |
| second_half | `0.08333333333333333` |
| attention_only | `0.08333333333333333` |
| mamba_only | `0.08333333333333333` |

The packet therefore does not support a clean attention-vs-Mamba attribution
claim. Broad residual clipping in any preregistered layer group removed the two
baseline successes.

## Artifact Completeness

`artifact_complete=true` in `artifact_check.json`.

Required packet files present:

- `environment.json`
- `model_provenance.json`
- `prompt_manifest.json`
- `command_metadata.json`
- `random_seed.json`
- `ablation_config.json`
- `generations.jsonl`
- `metrics.json`
- `bootstrap_ci.json`
- `stratified_metrics.json`
- `headroom_diagnostics.json`
- `artifact_hashes.json`
- `logs/stdout.log`
- `logs/stderr.log`
- `run_events.jsonl`

## Artifact SHAs

Selected artifact hashes:

- `artifact_check.json`: `sha256:bc5a2bfdb16437d800468e76c05f687ac567a337122b1e47e70f66116653c5d5`
- `checker_result.json`: `sha256:2289271c741d0ca43213d5f65cbe91c9229afc8cbd5779209f74ea475881d186`
- `metrics.json`: `sha256:a7252bc6f80ed907980a4f9c556685fed7d4ff9cc0f8ca2d7469d4415608bb86`
- `bootstrap_ci.json`: `sha256:46244e78cafd7bd75e3f11085d7be822ab41cbac988361508b97ade66e84c93b`
- `stratified_metrics.json`: `sha256:2f6e538f53f63358918006e575043fd3934452dfabad389f9b9b665da5a02275`
- `headroom_diagnostics.json`: `sha256:e8f0fdd724e87708156469ad4300a96496e6d2abda340a394f8b86083f2897e6`
- `generations.jsonl`: `sha256:759f0bfc03bba92445e93f666c02bfde3d3f284eefe93ff5b4e11fb26de3feea`
- `artifact_hashes.json`: `sha256:c2a7140c0f4e858b9e39d5dad048ca2cd13b41e63101ff0283f4e8435db9ea1d`

Full packet hashes are in:

- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/artifact_hashes.json`

## Disposition

No paper iteration is opened for this kill. The Phase 0 draft remains preserved
as an audit artifact, but Residual Migration does not advance to Phase 2
cross-model validation under this preregistration.

Why this is not a negative-result paper: the result is a mechanical scale-gate
failure with low headroom (`2 / 24` baseline successes) and no clean
layer-stratified attribution, not a positive method or a broadly defensible
claim about residual outlier causality in hybrid reasoners. It is useful as an
audit trail and as input to diagnostics, but it would be weak as a standalone
submission.

The diagnostic in `experimental/residual_migration/diagnostic.md` records any
fresh positive-method pivot hypotheses that would require new preregistration
before data collection.
