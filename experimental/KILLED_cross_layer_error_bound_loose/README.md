# Cross-Layer Quantization Error Compounding: KILL Manifest

Date killed: 2026-05-08

Decision: `KILL_CLE_BOUND_LOOSE`

Preregistration: `experimental/cross_layer_error/preregister_cle_theoretical.md`

Run directory: `experimental/cross_layer_error/results/cle_theoretical_20260508T191327Z`

Checker: `experimental/cross_layer_error/check_cle_bound.py`

Artifact check: `experimental/cross_layer_error/results/cle_theoretical_20260508T191327Z/artifact_check.json`

## Decision Summary

The theoretical branch is killed because the preregistered bound was too loose.
The packet is artifact-complete, and the checker reached the preregistered kill
decision mechanically.

Preregistered rule:

- PASS if predicted/measured drift is in `[1.0, 2.0]` at every depth.
- KILL if predicted/measured drift is `>5.0` at any depth, or if the bound
  predicts the wrong functional form.

Observed result:

| Depth | Mean L2 drift | 95% CI | Predicted bound | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 252.78462756756122 | [199.47787392131627, 321.07791007647427] | 1323.2819430699453 | 5.2348196795166055 |
| 5 | 402.63182934109807 | [341.61210200894214, 472.0482870268835] | 2011.7596640493757 | 4.996524162884974 |
| 10 | 466.36588408768 | [398.27372784149634, 531.0936687232418] | 2669.6676375036664 | 5.724405940898007 |
| 15 | 472.08004427928694 | [398.9710663600348, 546.8640814839874] | 3267.8287115543008 | 6.922192012041549 |

Kill trigger: ratio `>5.0` at depths 1, 10, and 15.

The functional-form check did not independently kill the branch:

- measured depth-15/depth-1 growth factor: `1.8675187997858576`
- predicted depth-15/depth-1 growth factor: `2.469487873440718`
- `functional_form_mismatch=false`

The proximate failure is scale/calibration, not missing artifacts.

## Artifact Completeness

`artifact_complete=true` in `artifact_check.json`.

Required packet files present:

- `derivation.md`
- `environment.json`
- `model_provenance.json`
- `prompt_manifest.json`
- `command_metadata.json`
- `random_seed.json`
- `quantization_config.json`
- `predicted_bounds.json`
- `trace_tokens.jsonl`
- `logits_manifest.json`
- `raw_drift_rows.jsonl`
- `drift_metrics.json`
- `bootstrap_ci.json`
- `artifact_hashes.json`
- `logs/stdout.log`
- `logs/stderr.log`
- `run_events.jsonl`

## Artifact SHAs

Selected artifact hashes:

- `artifact_check.json`: `sha256:f0ab830459a8df66d7084b533555e46d75ff1214fbafb1edea6ae28a7909f635`
- `checker_result.json`: `sha256:8736b6bd00538781c6ab5f664fb96f635b9b6d7d25360f93a541cfb253e3e52f`
- `artifact_hashes.json`: `sha256:c484fe3f6a3b071efc9efba44bc51ac10b12f0684da553997fdf03d3869fdf1a`
- `drift_metrics.json`: `sha256:9aaa01d919c681c00b901a620ab08b0c9d71000005fed6bcac57feaa86c6b965`
- `predicted_bounds.json`: `sha256:cee8fb553f08ee320317fd65762279729db8e60ebae73f9949e79b6f29f01487`
- `derivation.md`: `sha256:eaeb0d6596ec1e38d499fd77052d9d1412ac85b4316627758a70163779ad2004`
- `raw_drift_rows.jsonl`: `sha256:36d59331f33dd6e4d78577c5f77a98faba6028710c81f68e784cfa9500a9aff7`
- `bootstrap_ci.json`: `sha256:9b90c65346265e13f8af0713fcf201fddcf20bdabafd797051a3319858a04886`
- `logits_manifest.json`: `sha256:590214c6aaaf10c1f985ae1ff7f39a9a1853e1323ed3493082391165c9b386f0`
- `trace_tokens.jsonl`: `sha256:7e1c243c95b117cfb88e6efe186ef6e09294b3dceddfc4590cdc2b058ecba339`

Full packet hashes are in:

- `experimental/cross_layer_error/results/cle_theoretical_20260508T191327Z/artifact_hashes.json`

## Caveat

`derivation.md` prose refers to a 32-element block in one place, while
`quantization_config.json` and the runner use the artifacted 16-value block
configuration:

- `block_size=16`
- `quantization_format=nvfp4_e2m1_weight_sim`
- `native_kernel_claim=false`

This documentation inconsistency did not drive the kill decision. The checker
used the serialized metrics and the packet's explicit quantization
configuration.

## Disposition

No paper draft is opened for this killed branch. The diagnostic in
`experimental/cross_layer_error/diagnostic.md` records positive-method pivot
hypotheses that would require fresh preregistration before any new data is
inspected.
