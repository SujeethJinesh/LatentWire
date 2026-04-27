# SVAMP32 Source Sample Selector New Clean2

- date: `2026-04-27`
- status: `selector_fails_no_clean_recovery`
- scale rung: `smoke`
- base target set: `results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json`
- decision surface: `decision_surface.json`
- clean IDs:
  - `6e9745b37ab6fc45`
  - `de1bf4d142544e5b`

## Gate

The decision surface appends the four source-sampling rows from
`results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl` as
`source_sample_s0` through `source_sample_s3`, while retaining the prior
target/no-source candidate pool. The raw source rows still have inherited
methods `target_sample_s0` through `target_sample_s3`; the decision surface
aliases their labels to avoid target-source provenance confusion.

## Results

| Profile | Status | Matched Correct | Matched Clean Correct | Control Clean Union | Accepted Harm |
|---|---|---:|---:|---:|---:|
| `full` | `top_sidecar_selector_fails_smoke` | `6/32` | `0` | `0` | `5` |
| `answer_only` | `top_sidecar_selector_fails_smoke` | `6/32` | `0` | `0` | `5` |
| `answer_masked` | `top_sidecar_selector_fails_smoke` | `2/32` | `0` | `0` | `6` |

Per-ID matched selections on the clean IDs were wrong:

- `de1bf4d142544e5b`: matched selected `40`, `40`, or `97` depending on
  profile; gold is `57`.
- `6e9745b37ab6fc45`: matched selected fallback `600` or wrong value `661`;
  gold is `61`.

## Decision

Fail. The deterministic source-candidate score sidecar does not recover either
new clean source-sampled candidate, including in answer-masked mode. It also
introduces several target-correct harms. Do not tune this selector family on the
same two-ID surface.

## Artifacts

- `decision_surface.json`
- `decision_surface.md`
- `sidecars_full/manifest.md`
- `sidecars_answer_only/manifest.md`
- `sidecars_answer_masked/manifest.md`
- `top_select_full.md`
- `top_select_answer_only.md`
- `top_select_answer_masked.md`
