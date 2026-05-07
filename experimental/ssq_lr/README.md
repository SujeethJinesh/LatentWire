# SSQ-LR

SSQ-LR tests whether recurrent SSM state in hybrid reasoners can be quantized
below FP16 without quality loss during long reasoning.

## Current Readiness

Status: **NEW / Mac gates pending**.

Estimated completion:

- **15%** as a positive-method paper: hypothesis, gates, packet checker, and
  trace-plan handoff are scaffolded.
- **0%** as a systems-result paper: no native GPU state-cache integration or
  benchmark exists.

## Paper Story

Published systems often leave recurrent SSM state at FP16/FP32 while quantizing
weights, activations, or KV cache. SSQ-LR asks whether a sub-FP16 state recipe
can preserve reasoning quality and reduce state memory/bandwidth in hybrid
Mamba-Transformer models.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_ssq_lr_20260506.md`
- `phase1/competitor_matrix.md`

Gate S1 tests state distribution heterogeneity. Gate S2 tests simulated state
quantization sensitivity. Gate S3 tests cross-model transfer without retuning.

## Current Mac Packet

Synthetic real-schema rehearsal packet:

- `phase2/results/ssq_lr_synthetic_s1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`

This uses the real SSQ-LR S1 row schema, provenance fields, recomputed summary
fields, and real checker path. It is synthetic CPU data only and cannot promote
S1.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1 \
  --mode real --project ssq_lr \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/` for packet provenance.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.
The exact S1 capture checklist is
`../shared/results/hybrid_trace_plan_20260507/ssq_lr_trace_plan.jsonl`;
regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

This trace plan is not model evidence. It only enumerates the rows that must be
captured before `hybrid_trace_packet_builder` can produce a real S1 packet.

Validate the first real S1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<YYYYMMDD>_<model_slug> \
  --mode real --project ssq_lr
```

The real checker requires `prefill_end`, `2k_or_end`, `8k_or_end`, and
`final_minus_128` buckets for every `(prompt_id, layer)` pair plus at least 12
prompt IDs, unless `config.json` records a resource-limit note. Rows must
identify an SSM/Mamba `layer_kind` and a recurrent `state_tensor_kind`, so
arbitrary activation tensors cannot promote S1. Any
resource-limited packet must set a decision beginning
`RESOURCE_LIMITED_NOT_PROMOTABLE`; it may document local limits but cannot
promote S1. Schema-rehearsal packets must set `schema_rehearsal: true` in
`config.json`, use a decision beginning `SCHEMA_REHEARSAL_NOT_PROMOTABLE`, and
remain non-evidence even when their synthetic rows make the evaluator pass.

Real `config.json` provenance must include `prompt_ids_hash`,
`architecture_map_hash`, and `trace_plan_hash` as `sha256:<64-hex-digest>`
strings. Real `summary.json` must
include the recomputed S1 evaluator fields: `gate_status`, `gate_pass`,
`prompt_count`, `position_buckets`, `ssm_layer_count`, `passing_layer_count`,
`distribution_passing_layer_count`, `required_passing_layer_count`,
`pass_fraction`, `selected_s1_ratio`, `selected_s1_ci_low`, `holm_p_min`,
`magnitude_gate_pass`, `distribution_effect_floor_pass`,
`distribution_gate_pass`, and the final-minus-128 versus
prefill-end max-abs/std/kurtosis ratios. The checker recomputes these values
from prompt-level bucket ratios plus Holm-corrected distribution tests and
rejects stale summaries. A distribution-only pass must also clear the
preregistered 1.25x effect-size floor; tiny statistically significant shifts do
not promote S1.

## Output Paths

Use:

```text
experimental/ssq_lr/phase2/results/ssq_lr_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

Reproduce the current synthetic packet:

```bash
./venv_arm64/bin/python -m experimental.ssq_lr.phase2.ssq_lr_synthetic_s1_gate
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1 \
  --mode real --project ssq_lr \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
jq '.decision, .row_count, .gate_status, .selected_s1_ratio' \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1/summary.json
```

Expected decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`.

## GPU Rule

No 5090 work until S1--S3 pass and the exact state quantization recipe is
frozen.
