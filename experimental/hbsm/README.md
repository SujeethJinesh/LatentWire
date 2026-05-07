# HBSM

HBSM tests whether layer sensitivity in current hybrid reasoners can be
explained and predicted more cheaply than full forward-pass KL sweeps.

## Current Readiness

Status: **NEW / wounded novelty / weak resource-limited Mac smoke complete**.

Estimated completion:

- **20%** as a narrow mechanism paper: hypothesis, gates, packet checker,
  trace-plan handoff, and one resource-limited real-model sensitivity packet
  are scaffolded.
- **0%** as a broad sensitivity-discovery paper because recent KL Lens-style
  work narrows that novelty.

## Paper Story

HBSM is not a generic forward-only sensitivity paper. Its remaining wedge is a
mechanistic account for frontier hybrid reasoners, FP4-specific sensitivity,
and cheaper predictors based on weight/statistics rather than repeated
quantized forward passes.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_hbsm_20260506.md`
- `phase1/competitor_matrix.md`

B1 replicates sensitivity heterogeneity on current hybrids. B2 tests
no-forward-pass predictors. B3 tests the softmax-amplification mechanism.

Current executable scope: B1 has the strict real trace packet builder/checker
path. B2/B3 now have follow-up contract checks in
`../shared/followup_gate_contracts.py`, but no B2/B3 model packets exist and no
cheap-predictor, no-forward-pass, or mechanism claim is allowed until real B1
promotes and those contracts are filled from frozen predictor splits and
matched-noise mechanism rows.

## Current Mac Packet

Synthetic-only real-schema rehearsal packet:

- `phase2/results/hbsm_synthetic_b1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`
- rows: `720` (`480` primary prompt rows plus `240` layer-aligned control rows)

This validates the real B1 row schema, prompt-to-layer aggregation, required
controls, provenance fields, and recomputed evaluator summary. It is not model
evidence and cannot promote B1.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1 \
  --mode real --project hbsm \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit boundary IDs and architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/` for boundary-flagged
layer definitions.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.
Local capture readiness is recorded in
`../shared/results/hybrid_local_capture_preflight_20260507/`; the current
decision is `LOCAL_CAPTURE_READY_NOT_EVIDENCE` for Granite Tiny because its
weights are cached locally and its native `transformers` hybrid class is
available. Granite Small and Qwen3-Next remain GPU-sized or uncached.
`mamba_ssm` and `vllm` are recorded as optional runtime packages here, not hard
blockers for a local `transformers` capture. This packet is preflight-only and
cannot promote B1. Rerun it before any real capture attempt:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_local_capture_preflight
```

The current local execution smoke is
`../shared/results/hybrid_transformers_smoke_probe_20260507/`, decision
`RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`. It loaded Granite Tiny and
ran one 8-token CPU forward. This proves the local model path is viable for
developing the B1 sensitivity runner, but it has no sensitivity rows and cannot
promote B1.

The shared manifest local capture runner currently covers SSQ-LR and HORN only.
HBSM now has its own perturbation/sensitivity replay because it emits row
packets, not raw tensor packets; do not treat the SSQ-LR/HORN
resource-limited packets as HBSM evidence.

Resource-limited real-model smoke packet:

- `../shared/results/hbsm_local_sensitivity_20260507/`
- gate packet: `../shared/results/hbsm_local_sensitivity_20260507/hbsm_gate_packet/`
- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- checker: passes `check_gate_packet --mode real --project hbsm`
- rows: `56` (`8` primary Granite Tiny layers plus six layer-aligned control
  families)
- top observed drift: layer `5`, symmetric KL `0.027301367`
- B1 readout: `fisher_p_boundary_top_decile=0.375`,
  `cheap_predictor_spearman=-0.476`, so the smoke packet is weak and
  non-promoting.

Regenerate it with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hbsm_local_sensitivity_runner \
  --max-input-tokens 8 --layer-limit 8 --block-size 32
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/hbsm_local_sensitivity_20260507/hbsm_gate_packet \
  --mode real --project hbsm
```

The exact B1 sensitivity-row checklist is
`../shared/results/hybrid_trace_plan_20260507/hbsm_trace_plan.jsonl`;
regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

This trace plan is not model evidence. It only enumerates the layer/prompt rows
and comparator controls to fill before building a real HBSM packet.
Generate the fill-in row-packet templates with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

For HBSM, the current templates live under
`../shared/results/hybrid_capture_manifests_20260507/` as
`hbsm__<model_slug>__row_packet_template.json`. They are not model evidence:
fill every `TO_FILL_BEFORE_CAPTURE` field from a real sensitivity capture
before using `hybrid_trace_packet_builder --project hbsm --row-packet ...`.
The builder rejects `_template_only: true` templates and unfilled markers. If
the capture records a served HF model ID, the builder preserves it as
`served_model_id` and canonicalizes row `model_id` to the shared architecture
map slug before validation.
Required real controls are `perturbation_off`, `random_flags`, `layer_index`,
`parameter_count_norm`, `boundary_only`, `kl_lens_rank`, and
`activation_outlier`. Real rows must also include
`prompt_id`, `top_decile_flag`, `random_top_decile`, and `train_test_split`.
Only `boundary_only` rows are scored for B1 enrichment. The evaluator aggregates
prompt rows to one score per `(model_id, layer)` before computing top-decile
enrichment, so prompt count cannot inflate the effective layer count. Primary
rows still need prompt-level boundary/non-boundary layer coverage, true
top-decile cardinality `ceil(0.10 * scoring_layers)`, a same-count random
baseline that is not enriched, and both train/test split rows unless the packet
records a resource-limit note.
The B1 evaluator derives measured top-decile membership from aggregated
`kl_or_nll_drift`; the checker rejects packets whose supplied
`top_decile_flag` values disagree with that measured ranking, including any
individual prompt row inside a measured top layer.
Convert saved B1 sensitivity rows with
`experimental.shared.hybrid_trace_packet_builder --project hbsm --row-packet
...` before validation.

Any resource-limited packet must set a decision beginning
`RESOURCE_LIMITED_NOT_PROMOTABLE`; it may document local limits but cannot
promote B1. Real `config.json` provenance must include `prompt_ids_hash`,
`architecture_map_hash`, and `trace_plan_hash` as `sha256:<64-hex-digest>`
strings. Real `summary.json` must
include the recomputed B1 evaluator fields: `gate_status`, `gate_pass`,
primary row, scoring-layer, and prompt counts, expected/top/random/train/test counts,
split/control summaries, boundary/non-boundary top-decile counts and rates,
random-baseline counts/rates, `boundary_top_decile_enrichment`,
`random_boundary_top_decile_enrichment`, Fisher p-values, and
`cheap_predictor_spearman`, `baseline_spearman`, and
`cheap_predictor_margin_vs_best_baseline`. The checker recomputes these values from rows and
rejects stale summaries.

Validate the first real B1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_gate_b1_<YYYYMMDD>_<model_slug> \
  --mode real --project hbsm
```

Validate later B2/B3 follow-up packets only after real B1 promotes:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/hbsm/phase2/results/hbsm_gate_b2_<YYYYMMDD>_<model_slug> \
  --gate hbsm_b2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/hbsm/phase2/results/hbsm_gate_b3_<YYYYMMDD>_<model_slug> \
  --gate hbsm_b3
```

The B2 contract requires one preregistered predictor registry hash, fixed
hyperparameter hashes, train-only predictor selection, held-out Spearman and
baseline-margin fields, and explicit baseline-predictor rows. The B3 contract
requires matched attention/SSM noise rows, HORN-alignment sign, confidence
intervals around the mechanism effect, noise-off rows near zero drift, and
direction-flip controls.

## Output Paths

```text
experimental/hbsm/phase2/results/hbsm_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

Reproduce the current synthetic packet:

```bash
./venv_arm64/bin/python -m experimental.hbsm.phase2.hbsm_synthetic_b1_gate
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1 \
  --mode real --project hbsm \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
jq '.decision, .row_count, .gate_status, .primary_row_count, .scoring_layer_count' \
  experimental/hbsm/phase2/results/hbsm_synthetic_b1/summary.json
```

Expected decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`.

Resource-limited Granite Tiny execution smoke:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_transformers_smoke_probe \
  --max-input-tokens 8
```

Resource-limited Granite Tiny HBSM B1 sensitivity smoke:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hbsm_local_sensitivity_runner \
  --max-input-tokens 8 --layer-limit 8 --block-size 32
```

## GPU Rule

No GPU validation until B1--B3 pass. If HORN passes and HBSM is redundant, fold
HBSM into HORN instead of keeping a separate paper.
