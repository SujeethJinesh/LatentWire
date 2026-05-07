# SSQ-LR

SSQ-LR tests whether recurrent SSM state in hybrid reasoners can be quantized
below FP16 without quality loss during long reasoning.

## Current Readiness

Status: **NEW / layer-selective S1 signal alive but non-promoting**.

Estimated completion:

- **30%** as a positive-method paper: hypothesis, gates, packet checker,
  trace-plan handoff, one corrected one-layer S1 smoke pass, one four-layer S1
  smoke failure, one all-recurrent-layer metrics-only S1 failure, and one
  checker-passing selected-layer prompt-repeat packet are scaffolded.
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

Current executable scope: S1 has the strict real trace packet builder/checker
path. S2/S3 now have follow-up contract checks in
`../shared/followup_gate_contracts.py`, but no S2/S3 model packets exist and no
quantization-quality, byte-savings, or cross-model-transfer claim is allowed
until real S1 promotes and those follow-up contracts are filled from the frozen
real S1 replay surface.

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
Local capture readiness is recorded in
`../shared/results/hybrid_local_capture_preflight_20260507/`; the current
decision is `LOCAL_CAPTURE_READY_NOT_EVIDENCE` for Granite Tiny because its
weights are cached locally and its native `transformers` hybrid class is
available. Granite Small and Qwen3-Next remain GPU-sized or uncached.
`mamba_ssm` and `vllm` are recorded as optional runtime packages here, not hard
blockers for a local `transformers` capture. This packet is preflight-only and
cannot promote S1. Rerun it before any real capture attempt:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_local_capture_preflight
```

The current local execution smoke is
`../shared/results/hybrid_transformers_smoke_probe_20260507/`, decision
`RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`. It loaded Granite Tiny, ran
one 8-token CPU forward, and observed 36 recurrent-state cache layers plus 4
attention-cache layers. This proves the local execution path reaches recurrent
state, but it is too short and resource-limited to promote S1.

The current manifest-driven local capture packet is
`../shared/results/hybrid_manifest_local_capture_20260507/ssq_lr_gate_packet/`,
decision `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`. It has
4 rows for one prompt, one layer, and all four S1 buckets, selected ratio `1.0`, and
passes `check_gate_packet --mode real --project ssq_lr`. It proves saved-tensor
provenance and checker reload, not S1 heterogeneity. The next runner change must
capture true intermediate recurrent states before prompt/layer scaling, because
the current short forward only exposes the final recurrent cache state.

The corrected bucket-truncated local capture packet is:

- `../shared/results/ssq_lr_local_bucket_capture_20260507/ssq_lr_gate_packet/`
- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`
- checker: passes `check_gate_packet --mode real --project ssq_lr`
- rows: `4` (one prompt, one layer, four short-prefix bucket replays at
  2/4/6/8 tokens)
- selected S1 ratio: `3.293847`

This fixes the known duplicate-final-cache smoke issue and proves the local
runner can capture non-identical recurrent states across S1 buckets. It is
still explicitly non-promoting: one prompt/layer and short-prefix bucket labels
are not enough for S1.

The current multilayer local capture packet is:

- `../shared/results/ssq_lr_local_multilayer_capture_20260507/ssq_lr_gate_packet/`
- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`
- checker: passes `check_gate_packet --mode real --project ssq_lr`
- rows: `16` (one prompt, four layers, four short-prefix bucket replays)
- S1 readout: `passing_layer_count=1`, `required_passing_layer_count=3`
- per-layer max-abs ratios: layer `0` = `3.2938`, layer `1` = `1.3736`,
  layer `2` = `0.8811`, layer `3` = `0.8680`

This weakens SSQ-LR: the one-layer pass appears isolated on this short local
surface. Do not move SSQ-LR to GPU until either an all-layer/limited-prompt
scout or a prompt repeat shows broader S1 support. Only the compact readout is
tracked; regenerate the full local tensor packet before rerunning the checker.

The current all-recurrent-layer metrics scout is:

- `../shared/results/ssq_lr_all_layer_scout_20260507/`
- decision: `RESOURCE_LIMITED_ALL_LAYER_SCOUT_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`
- rows: `144` (one prompt, 36 recurrent layers, four short-prefix bucket replays)
- S1 readout: `passing_layer_count=4`, `required_passing_layer_count=9`
- passing local layers: `0`, `12`, `18`, `30`
- selected global S1 ratio: `0.806153`

This further weakens SSQ-LR on the current Mac surface. The branch should not
move to GPU unless a prompt-repeat scout shows those four layers reproduce
across prompts, or a full S1 packet changes the conclusion.

The selected-layer prompt-repeat tensor packet is:

- `../shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/ssq_lr_gate_packet/`
- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`
- checker: passes `check_gate_packet --mode real --project ssq_lr`
- rows: `192` (12 prompts, selected layers `0`, `12`, `18`, `30`, four buckets)
- S1 readout: `passing_layer_count=3`, `required_passing_layer_count=3`
- selected S1 ratio: `2.561113`
- selected S1 CI low: `2.014131`
- distribution passing layers: `4`
- Holm minimum p-value: `2.775512e-05`

This keeps SSQ-LR alive as a layer-selective hypothesis, but it is still
explicitly non-promoting because the layer subset was selected after the
all-layer scout. Treat layers `0`, `12`, and `30` as the current frozen primary
set and layer `18` as a near-miss/control. The next local gate must be either a
fresh/held-out layer-selective S1b or an S2 state-quantization sensitivity gate;
do not move to GPU from this packet alone.

Regenerate it with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --max-input-tokens 8 \
  --output-dir experimental/shared/results/ssq_lr_local_bucket_capture_20260507
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_local_bucket_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

Regenerate the multilayer smoke with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --max-input-tokens 8 --ssq-layer-limit 4 \
  --output-dir experimental/shared/results/ssq_lr_local_multilayer_capture_20260507
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_local_multilayer_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

Regenerate the all-layer metrics scout with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.ssq_lr_all_layer_scout \
  --max-input-tokens 8 \
  --output-dir experimental/shared/results/ssq_lr_all_layer_scout_20260507
```

Regenerate the selected-layer prompt-repeat tensor packet with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --ssq-prompt-limit 12 --ssq-layers 0,12,18,30 \
  --max-input-tokens 8 \
  --output-dir experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

The exact S1 capture checklist is
`../shared/results/hybrid_trace_plan_20260507/ssq_lr_trace_plan.jsonl`;
regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

This trace plan is not model evidence. It only enumerates the rows that must be
captured before `hybrid_trace_packet_builder` can produce a real S1 packet.
Generate the fill-in metadata templates with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

For SSQ-LR, the current templates live under
`../shared/results/hybrid_capture_manifests_20260507/` as
`ssq_lr__<model_slug>__metadata_template.json`. They are not model evidence:
fill every `TO_FILL_BEFORE_CAPTURE` field from a real SSM-state capture before
using `hybrid_trace_packet_builder`. The builder rejects `_template_only: true`
templates and unfilled markers. If the capture records a served HF model ID
such as `ibm-granite/granite-4.0-h-tiny`, the builder preserves it as
`served_model_id` and canonicalizes row `model_id` to the shared architecture
map slug before validation.

Validate the first real S1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<YYYYMMDD>_<model_slug> \
  --mode real --project ssq_lr
```

Validate later S2/S3 follow-up packets only after real S1 promotes:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s2_<YYYYMMDD>_<model_slug> \
  --gate ssq_lr_s2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s3_<YYYYMMDD>_<model_slug> \
  --gate ssq_lr_s3
```

The S2 contract requires frozen recipe IDs, effective bits, state/scale/metadata
bytes, BF16 no-op drift, same-byte controls, explicit INT8/FP8/MXFP4 state
baselines, random same-L2 noise controls, shuffled-scale controls, paired
uncertainty, and a recipe that clears both the quality and 4x state-memory
gates. The S3 contract requires one frozen recipe hash, one source S2 packet
hash, no retuning rows, and transfer quality within the preregistered tolerance
on at least two validation models.

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
preregistered 1.25x effect-size floor per passing layer/metric test; a large
global mean from one layer cannot promote tiny statistically significant shifts
elsewhere.

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

Resource-limited Granite Tiny execution smoke:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_transformers_smoke_probe \
  --max-input-tokens 8
```

Resource-limited manifest capture packet:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --max-input-tokens 8
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/hybrid_manifest_local_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

## GPU Rule

No 5090 work until S1--S3 pass and the exact state quantization recipe is
frozen.
