# HORN

HORN tests whether activation outliers propagate asymmetrically through hybrid
attention/SSM boundaries.

## Current Readiness

Status: **NEW / Mac gates pending**.

Estimated completion:

- **10%** as a positive-method paper: hypothesis and gates are scaffolded.
- **0%** as a systems-result paper: no precision allocation or native GPU
  validation exists.

## Paper Story

If attention-to-SSM and SSM-to-attention boundaries have different outlier and
noise-propagation behavior, then uniform activation precision is wasteful.
HORN asks whether one boundary direction needs higher precision than the other.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_horn_20260506.md`
- `phase1/competitor_matrix.md`

H1 measures boundary activation magnitude and kurtosis. H2 injects
FP4-equivalent noise around each boundary direction. H3 checks cross-model and
pure-architecture controls.

## Current Mac Packet

Synthetic-only packet:

- `phase2/results/horn_synthetic_h1/`
- decision: `SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT`

This validates artifact mechanics only. It is not model evidence.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_synthetic_h1 \
  --expected-decision-prefix SYNTHETIC
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit boundary IDs and architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/`; do not rely on
substring-only module classification for real H1 rows.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.

Validate the first real H1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_gate_h1_<YYYYMMDD>_<model_slug> \
  --mode real --project horn
```

The real checker requires at least 12 prompt IDs unless resource-limited, both
boundary directions, both-direction non-boundary controls, paired flipped
`permuted_direction` controls, and finite numeric rows. The flipped controls
must match an observed boundary by `prompt_id`, boundary index, layer IDs, and
normalization positions, reuse the observed boundary metrics, then invert only
the direction label. H1 cannot pass if
the permuted controls preserve the selected high-magnitude direction; a faithful
label flip may preserve unsigned max/min asymmetry while moving the signal to
the opposite label, which is an acceptable null. Non-boundary controls must stay
below the selected H1 threshold rather than merely below the boundary ratio. Any
resource-limited packet must set a decision
beginning `RESOURCE_LIMITED_NOT_PROMOTABLE`; it may document local limits but
cannot promote H1.

Real `config.json` provenance must include `prompt_ids_hash` and
`architecture_map_hash` as `sha256:<64-hex-digest>` strings. Real `summary.json` must
include the recomputed H1 evaluator fields: `gate_status`, `gate_pass`,
`prompt_count`, `boundary_directions`, `selected_h1_metric`,
`selected_h1_direction`, `selected_h1_ratio`, `selected_h1_threshold`,
`selected_h1_ci_low`, direction ratios, control ratios, and
direction-count/support fields. The checker recomputes these values from rows
and rejects stale summaries.

## Output Paths

```text
experimental/horn/phase2/results/horn_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

Reproduce the current synthetic packet:

```bash
./venv_arm64/bin/python -m experimental.horn.phase2.horn_synthetic_h1_gate
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_synthetic_h1 \
  --expected-decision-prefix SYNTHETIC
jq '.decision, .ssm_to_attention_over_attention_to_ssm_max_ratio, .ssm_to_attention_over_attention_to_ssm_kurtosis_ratio' \
  experimental/horn/phase2/results/horn_synthetic_h1/summary.json
```

Expected decision: `SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT`.

## GPU Rule

No native performance or precision-allocation claim until H1--H3 pass and a
directional recipe is frozen. GPU execution may still be used to collect the
same H1--H3 evidence if local hybrid model loading is the only blocker.
