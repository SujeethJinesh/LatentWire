# Hybrid Trace Packet Runbook

This runbook defines the first real Mac/GPU-admissible trace packet for
SSQ-LR, HORN, and HBSM. Synthetic packets validate script mechanics only.

## Required Packet Shape

Every real packet must contain:

- `config.json`: model id, model revision/hash, tokenizer revision, prompt
  source, prompt ids/hash, seed list, context lengths, dtype, device, and exact
  command. It must also include `architecture_map_hash` from
  `shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`
  and `trace_plan_hash` from the project trace-plan JSONL used for capture.
- `raw_rows.jsonl`: one JSON row per layer/boundary/state measurement.
- `summary.json`: aggregate readouts, decision, claim boundary, and row count.
- `summary.md`: human-readable table and interpretation.
- `decision.md`: pass/fail/continue decision tied to the preregistered gate.
- optional `tensors/`: state or activation tensor packet written with
  `activation_dumper.py`.

Validate any packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir>
```

For real packets, use the stricter project contract:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir> \
  --mode real --project <ssq_lr|horn|hbsm>
```

The checker rejects real packets without provenance fields, `summary.md`,
matching `row_count`, project-specific row schemas, required controls, and
project-specific admissible coverage.

For real packets, `config.json` must record `prompt_ids_hash`,
`architecture_map_hash`, and, unless it is a schema rehearsal,
`trace_plan_hash` as `sha256:<64-hex-digest>` strings. Non-rehearsal real
packets must also cite the trace-plan JSONL in `trace_plan_path`; packets that
omit it are not admissible because coverage cannot be checked. For
non-rehearsal real packets, `model_id` (or a registered `model_id_aliases` entry such as an HF
repo ID) and `architecture_map_hash` must match the shared architecture map
artifact at
`experimental/shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`;
a syntactically valid but unrelated hash is rejected. Packet builders canonicalize
registered served IDs into the map `model_id` and preserve the original value as
`served_model_id` in `config.json`. The `model_revision` and
`tokenizer_revision` values must match the registered eligibility snapshot SHA
for the served/canonical model, and `trace_plan_path` rows are checked against
that exact frozen row set. A packet that records
`resource_limit_note` is admissible only as a diagnostic artifact: its
`summary.json` decision must start with
`RESOURCE_LIMITED_NOT_PROMOTABLE`, and it cannot promote a gate.
The shared packet builder enforces this automatically for SSQ-LR, HORN, and
HBSM: if the input tensor/row metadata contains `resource_limit_note`, the
written packet decision is prefixed with `RESOURCE_LIMITED_NOT_PROMOTABLE_`
even when the recomputed gate status itself would pass. This lets two-prompt or
small-context smoke traces validate hooks and schema without weakening the
promotion boundary. Non-resource-limited packet decisions must equal the
recomputed project gate status exactly.
Synthetic schema rehearsals may run the same real validators only when
`config.json` sets `schema_rehearsal: true` and `summary.json` uses a decision
beginning `SCHEMA_REHEARSAL_NOT_PROMOTABLE`. These packets are checker-path
tests, not model evidence.

Before dumping tensors or sensitivity rows, generate the deterministic capture
plan:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

The current trace-plan artifact is
`experimental/shared/results/hybrid_trace_plan_20260507/`. It enumerates
`ssq_lr_trace_plan.jsonl`, `horn_trace_plan.jsonl`, and
`hbsm_trace_plan.jsonl` rows from the frozen prompt manifest and shared
architecture maps. Its decision is `TRACE_PLAN_READY_NOT_MODEL_EVIDENCE`: it is
only an execution checklist and cannot promote a gate.

Then generate capture-manifest templates:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

The current capture-manifest artifact is
`experimental/shared/results/hybrid_capture_manifests_20260507/`. It expands
the trace plans into per-project/per-model metadata templates for SSQ-LR and
HORN tensor packets, plus row-packet templates for HBSM sensitivity tables. Its
decision is `CAPTURE_MANIFEST_READY_NOT_MODEL_EVIDENCE`: these files contain no
tensors, no model outputs, no sensitivity metrics, and no GPU evidence. Fill
every `TO_FILL_BEFORE_CAPTURE` field from a real capture before invoking the
packet builder. The shared builders reject templates marked `_template_only:
true` or containing unfilled template markers.
For HORN `permuted_direction` entries, capture only the observed boundary
tensor and leave `tensor_alias_of` pointing to it; the builder reuses that
tensor while flipping only the metadata direction.

## SSQ-LR Real S1 Packet

Minimum admissible row fields:

- `model_id`, `model_revision`, `prompt_id`, `layer`, `layer_kind`,
  `position_bucket`
- `state_tensor_kind`, `state_shape`, `max_abs`, `rms`, `std`, `kurtosis`,
  `outlier_mass`
- `control_type`, including `bf16_no_quant`

Required controls:

- BF16/no-quant state recording.
- `layer_kind` must identify an SSM/Mamba recurrent layer, and
  `state_tensor_kind` must identify recurrent SSM state rather than an arbitrary
  activation tensor.
- At least the preregistered `prefill_end`, `2k_or_end`, `8k_or_end`, and `final_minus_128` position buckets.
- Every `(prompt_id, layer)` pair must cover all four preregistered buckets.
- At least 12 fixed prompts or an explicit resource-limit note.

Required `summary.json` fields:

- `gate_name`, `gate_status`, `gate_pass`
- `prompt_count`, `position_buckets`, `ssm_layer_count`, `passing_layer_count`
- `distribution_passing_layer_count`
- `required_passing_layer_count`, `pass_fraction`, `selected_s1_ratio`
- `selected_s1_ci_low`, `holm_p_min`
- `magnitude_gate_pass`, `distribution_effect_floor_pass`,
  `distribution_gate_pass`
- `max_abs_ratio_final_minus_128_vs_prefill_end`
- `std_ratio_final_minus_128_vs_prefill_end`
- `kurtosis_ratio_final_minus_128_vs_prefill_end`

The checker recomputes these fields with
`experimental.shared.hybrid_gate_evaluators.evaluate_ssq_lr_s1`; stale or
fabricated summaries are rejected. The S1 lower bound is computed from
prompt-level bucket ratios, and the distribution path uses Holm-corrected
two-sample tests between `prefill_end` and `final_minus_128`, so reduced rows
must preserve prompt IDs rather than only global layer means.
Distribution-only promotion also requires the selected S1 ratio to clear the
1.25x effect-size floor; tiny but statistically significant shifts remain a
failed S1 packet.

## HORN Real H1a/H1 Packet

The first live hybrid packet is H1a: a single-model screen that validates the
measurement path and decides whether to continue. H1 promotion requires at least
two hybrid models with the same selected direction and clean H3 controls.

Minimum admissible row fields:

- `model_id`, `layer_left`, `layer_right`, `direction`,
  `matched_boundary_direction`, `boundary_index`
- `prompt_id`, `pre_norm_position`, `post_norm_position`, `max_abs`, `rms`,
  `kurtosis`
- `control_type` in `boundary`, `non_boundary`, or `permuted_direction`

Required controls:

- non-boundary adjacent pairs;
- direction-label permutation matched to an observed boundary tuple, including
  `prompt_id`, boundary index, layer IDs, and matched normalization positions;
- at least one `attention->ssm` boundary and one `ssm->attention` boundary;
- non-boundary and permuted controls must each match both boundary directions;
  non-boundary rows may retain their true architecture direction, such as
  `ssm->ssm`, while `matched_boundary_direction` records the boundary
  direction they control for, and both matched non-boundary directions are
  required for every prompt;
- the non-boundary and permuted controls must erase the selected high-magnitude
  direction label; a faithful label flip may preserve unsigned max/min
  asymmetry while moving the signal to the opposite label;
- permuted controls must reuse the observed boundary metrics and flip only the
  actual `direction` label; for permuted rows `matched_boundary_direction`
  must be absent or equal to that flipped label;
- non-boundary controls must stay below the selected H1 threshold, not merely
  below the measured boundary ratio;
- matched normalization placement.
- at least 12 fixed prompts or an explicit resource-limit note.

Required `summary.json` fields:

- `gate_name`, `gate_status`, `gate_pass`
- `prompt_count`
- `boundary_directions`
- `selected_h1_metric`, `selected_h1_direction`, `selected_h1_threshold`
- `selected_h1_ratio`
- `selected_h1_ci_low`
- `max_abs_direction_ratio`, `kurtosis_direction_ratio`
- `non_boundary_control_ratio`, `permuted_direction_ratio`
- `non_boundary_direction_count`, `permuted_direction_count`
- `support_fraction`

The checker recomputes these fields with
`experimental.shared.hybrid_gate_evaluators.evaluate_horn_h1`; the H1a decision
is therefore coupled to the non-boundary and permuted-direction control ratios,
not only boundary rows.

## HBSM Real B1 Packet

Minimum admissible row fields:

- `model_id`, `layer`, `boundary_flag`, `precision_perturbation`
- `prompt_id`
- `kl_or_nll_drift`, `cheap_predictor`, `parameter_count`, `weight_norm`
- `top_decile_flag`, `random_top_decile`, `train_test_split`
- `control_type` in `perturbation_off`, `random_flags`, `layer_index`,
  `parameter_count_norm`, `boundary_only`, `kl_lens_rank`, or
  `activation_outlier`

Required controls:

- perturbation-off/no-op row with near-zero drift;
- random top-decile flags;
- layer-index baseline;
- parameter-count/norm baseline;
- boundary-only baseline;
- KL-style sensitivity ranking baseline;
- activation/outlier ranking baseline;
- `boundary_only` rows with both `boundary_flag=true` and `boundary_flag=false`;
- every `boundary_only` prompt must include boundary and non-boundary layers;
- B1 scoring aggregates `boundary_only` prompt rows to one row per
  `(model_id, layer)` before computing top-decile enrichment;
- measured `top_decile_flag=true` and `random_top_decile=true` counts must
  each equal `ceil(0.10 * scoring_layers)`;
- supplied `top_decile_flag` values must match the top scoring layers derived
  from aggregated `kl_or_nll_drift` for every `boundary_only` prompt row;
- random top-decile flags must not reproduce the boundary enrichment;
- both `train` and `test` split rows, unless a resource-limit note is present;
- train/test layer split if layer count permits.

Required `summary.json` fields:

- `gate_name`, `gate_status`, `gate_pass`
- `primary_row_count`, `scoring_layer_count`, `prompt_count`
- `expected_top_decile_count`
- `top_decile_count`
- `random_top_decile_count`
- `train_count`
- `test_count`
- `split_counts`, `control_types`
- `boundary_top_decile_count`, `non_boundary_top_decile_count`
- `boundary_top_decile_rate`, `non_boundary_top_decile_rate`
- `boundary_top_decile_enrichment`
- `fisher_p_boundary_top_decile`
- `cheap_predictor_spearman`, `baseline_spearman`,
  `cheap_predictor_margin_vs_best_baseline`

The checker recomputes these fields with
`experimental.shared.hybrid_gate_evaluators.evaluate_hbsm_b1`; stale
sensitivity summaries are rejected.

## Promotion Boundary

A real packet can promote only the next gate, never the full paper. Native GPU
speed, HBM, latency, and production-packing claims require separate NVIDIA
validation.
