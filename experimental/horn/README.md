# HORN

HORN tests whether activation outliers propagate asymmetrically through hybrid
attention/SSM boundaries.

## Current Readiness

Status: **DEMOTED CONTROL / H1a and H2 scouts failed**.

Estimated completion:

- **0% active positive-method readiness under the current demoted scope**:
  H1a/H2 failed and no GPU handoff is admissible.
- **12% reusable scaffold if the branch is deliberately preregistered again**: hypothesis, gates,
  packet checker, trace-plan handoff, one checker-passing resource-limited real
  H1a packet, and one H2 follow-up scout are scaffolded, but both observed
  directional signals are weak and the branch is demoted unless reopened with a
  new preregistered H2/H3 scope.
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

Current executable scope: H1a/H1 boundary statistics have the strict real trace
packet builder/checker path. H2/H3 now have follow-up contract checks in
`../shared/followup_gate_contracts.py`. A resource-limited H2 scout exists and
fails; it is not current evidence for a positive H2/H3 claim. No
noise-propagation, precision-allocation, or cross-architecture claim is allowed
unless HORN is deliberately reopened with a full preregistered H2/H3 scope on
more prompts/models.

## Current Mac Packet

Synthetic-only real-schema rehearsal packet:

- `phase2/results/horn_synthetic_h1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A`
- rows: `72`

This validates the real H1a row schema, paired controls, summary
recomputation, and non-promoting schema-rehearsal path. It is not model
evidence.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_synthetic_h1 \
  --mode real --project horn \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit boundary IDs and architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/`; do not rely on
substring-only module classification for real H1 rows.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.
Local capture readiness is recorded in
`../shared/results/hybrid_local_capture_preflight_20260507/`; the current
decision is `LOCAL_CAPTURE_READY_NOT_EVIDENCE` for Granite Tiny because its
weights are cached locally and its native `transformers` hybrid class is
available. Granite Small and Qwen3-Next remain GPU-sized or uncached.
`mamba_ssm` and `vllm` are recorded as optional runtime packages here, not hard
blockers for a local `transformers` capture. This packet is preflight-only and
cannot promote H1a/H1. Rerun it before any real capture attempt:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_local_capture_preflight
```

The current local execution smoke is
`../shared/results/hybrid_transformers_smoke_probe_20260507/`, decision
`RESOURCE_LIMITED_EXECUTION_SMOKE_NOT_PROMOTABLE`. It loaded Granite Tiny, ran
one 8-token CPU forward, and observed the expected hybrid cache split. This
proves the local execution path is alive, but it is too short and lacks
boundary activation hooks, so it cannot promote H1a/H1.

The current manifest-driven local capture packet is
`../shared/results/hybrid_manifest_local_capture_20260507/horn_gate_packet/`,
decision `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`.
It has 288 rows for 12 prompts across all 8 planned Granite Tiny boundaries,
both boundary directions, matched non-boundary controls, and permuted-direction
controls, selected ratio `1.06` with cluster-bootstrap low `1.06`, and passes
`check_gate_packet --mode real --project horn`. Its tensors are captured from
right-layer forward pre-hooks, so the packet now checks real boundary-input
plumbing rather than hidden-state proxies. It is still resource-limited
evidence and does not promote H1a/H1. The weak magnitude-screen effect means
HORN should remain a control branch unless a future, preregistered H2/H3
reopening gives a concrete reason this near-null screen should reverse.

The current H2 noisy-continuation scout is
`../shared/results/horn_h2_noise_replay_scout_20260507/`, decision
`RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
with raw `gate_status`
`FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`. It is a contract-valid
resource-limited follow-up packet, not H2 promotion: 20 rows, 2 prompts, 3
seeds, paired units `6/6`, hook-off max delta `0.0`, H1-selected direction
preserved in the aggregate, directional drift ratio `1.037`, signed
selected-direction lower bound `0.324`, selected-direction support fraction
`0.5`, and demotion recommendation `DEMOTE_HORN_STANDALONE_WEAK_H2`. This weak
H2 result demotes HORN as a standalone branch.
The exact reproduction command, artifact hashes, model revision, prompt source
hash, source H1a packet hash, and checker output are recorded in
`phase2/h2_noise_replay_repro_manifest_20260507.md`.

The exact H1a/H1 capture checklist is
`../shared/results/hybrid_trace_plan_20260507/horn_trace_plan.jsonl`;
regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

This trace plan is not model evidence. It only enumerates observed-boundary,
permuted-direction, and matched non-boundary rows to capture before building a
real HORN packet. Every planned HORN row, including non-boundary controls,
preserves `prompt_cluster_id` for cluster-bootstrap and paired-control checks.
Generate the fill-in metadata templates with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

For HORN, the current templates live under
`../shared/results/hybrid_capture_manifests_20260507/` as
`horn__<model_slug>__metadata_template.json`. They are not model evidence:
fill every `TO_FILL_BEFORE_CAPTURE` field from a real boundary-activation
capture before using `hybrid_trace_packet_builder`. The builder rejects
`_template_only: true` templates and unfilled markers. Permuted-direction rows
use `tensor_alias_of` to reuse the observed boundary tensor; do not capture a
second tensor for the flipped-label control. If the capture records a served HF
model ID, the builder preserves it as `served_model_id` and canonicalizes row
`model_id` to the shared architecture map slug before validation.

Validate the first real H1a screen packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/horn/phase2/results/horn_gate_h1_<YYYYMMDD>_<model_slug> \
  --mode real --project horn
```

The current Granite Tiny resource-limited H1a packet is expected to report
`RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`;
any packet beginning `PASS_REAL_H1A` must be a new preregistered full H1a/H1
capture, not a relabeling of the current scout.

Validate promotable H2/H3 follow-up packets only after real H1a/H1 promotes.
Under the current demoted scope, H2/H3 work is limited to reproducing existing
stop packets or to a new preregistered reopening. Historical resource-limited
H2/H3 demotion scouts remain explicitly non-promoting evidence:

```bash
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/horn/phase2/results/horn_gate_h2_<YYYYMMDD>_<model_slug> \
  --gate horn_h2
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/horn/phase2/results/horn_gate_h3_<YYYYMMDD>_<model_slug> \
  --gate horn_h3
```

The H2 contract requires a fixed H1-selected direction, exact noise side and
noise standard-deviation basis, at least three seeds, both boundary directions
paired for every `(prompt_cluster_id, seed, noise_side)` unit, paired clean/noisy
NLL rows, hook-off controls, and a directional drift ratio with paired lower
bound. The H3 contract requires at least two passing hybrid validation models
plus pure-attention and pure-Mamba controls that fold below the preregistered
1.2 null threshold or have CI overlap with 1.0 under the same directional test.

H1a is a single-model screen only. It can justify running H2 and adding more
models, but H1 promotion requires the same selected direction on at least two
hybrid models plus the H3 pure-architecture controls.

The real checker requires at least 12 prompt IDs unless resource-limited, both
boundary directions, both-direction non-boundary controls, paired flipped
`permuted_direction` controls, and finite numeric rows. Non-boundary controls
may keep their true architecture direction, such as `ssm->ssm`, but must carry
`matched_boundary_direction` so they can be paired against the boundary
direction they control for, and every prompt must include both matched
non-boundary directions. The flipped controls
must match an observed boundary by `prompt_id`, boundary index, layer IDs, and
normalization positions, reuse the observed boundary metrics, then invert only
the actual `direction` label; `matched_boundary_direction` must agree with
that flipped label for permuted rows. H1 cannot pass if
the permuted controls preserve the selected high-magnitude direction; a faithful
label flip may preserve unsigned max/min asymmetry while moving the signal to
the opposite label, which is an acceptable null. Non-boundary controls must stay
below the selected H1 threshold rather than merely below the boundary ratio. Any
resource-limited packet must set a decision
beginning `RESOURCE_LIMITED_NOT_PROMOTABLE`; it may document local limits but
cannot promote H1a or H1.

Real `config.json` provenance must include `prompt_ids_hash`,
`architecture_map_hash`, and `trace_plan_hash` as `sha256:<64-hex-digest>`
strings. Real `summary.json` must
include the recomputed H1a evaluator fields: `gate_status`, `gate_pass`,
`prompt_count`, `boundary_directions`, `selected_h1_metric`,
`selected_h1_direction`, `selected_h1_ratio`, `selected_h1_threshold`,
`selected_h1_ci_low`, `selected_h1_cluster_bootstrap_low`, direction ratios,
control ratios, and direction-count/support fields. The checker recomputes
these values from rows and rejects stale summaries. The lower bound is a
deterministic prompt-cluster bootstrap over cluster-level directional ratios, so
`prompt_cluster_id` must be preserved through reduction.

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
  --mode real --project horn \
  --expected-decision-prefix SCHEMA_REHEARSAL_NOT_PROMOTABLE
jq '.decision, .row_count, .gate_status, .selected_h1_ratio, .non_boundary_control_ratio, .permuted_direction_ratio' \
  experimental/horn/phase2/results/horn_synthetic_h1/summary.json
```

Expected decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A`.

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
  --project horn --max-input-tokens 8
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/hybrid_manifest_local_capture_20260507/horn_gate_packet \
  --mode real --project horn
```

Resource-limited H2 noisy-continuation scout:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.horn_h2_noise_replay_scout \
  --prompt-limit 2 --max-input-tokens 2 --seeds 1,2,3 --noise-scale 0.05
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/horn_h2_noise_replay_scout_20260507 \
  --gate horn_h2
```

Expected decision:
`RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`
with raw `gate_status` `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`.

## GPU Rule

No native performance or precision-allocation claim until H1--H3 pass and a
directional recipe is frozen. The current real H1a packet fails with selected
ratio `1.06`, and the H2 scout also stays near null with directional drift
ratio `1.037` and selected-direction support `0.5`. HORN must not be promoted
to GPU as a standalone branch. Keep it as negative/control evidence for
the active project ledger unless a future full H2/H3 reopening has a new
preregistered reason.
Allowed Mac-local work is limited to revalidating the existing H1a/H2 stop
packets, docs/tests/runbook hygiene, or writing a new preregistered reopening.
Do not add exploratory H2 rows under the current demoted scope.
