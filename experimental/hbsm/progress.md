# HBSM Progress

## 2026-05-06

Status: **NEW / wounded novelty / Mac gates pending**.

Added and ran a deterministic synthetic B1 packet. It has now been upgraded
to a real-schema rehearsal:

- script: `phase2/hbsm_synthetic_b1_gate.py`
- packet: `phase2/results/hbsm_synthetic_b1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`
- rows: `720` (`480` primary prompt rows plus `240` layer-aligned control rows)
- scoring layers after prompt aggregation: `40`
- real checker: passes `--mode real --project hbsm`

Interpretation: synthetic-only schema validation. It exercises the real B1 row
schema, prompt-to-layer aggregation, controls, provenance fields, and recomputed
evaluator summary, but does not promote the branch or replace real hybrid
layer-sensitivity measurements.

Next exact gate: B1 sensitivity heterogeneity replication on current hybrid
models. Kill or fold into HORN if the mechanism wedge does not differentiate.

## 2026-05-06 Architecture Provenance Update

Added shared config-derived architecture maps at
`../shared/results/hybrid_architecture_maps_20260506/`. Real B1 packets must use
these explicit boundary IDs for boundary-flagged layers and include
`architecture_map_hash`; cheap predictors will be compared against these fixed
flags plus random/layer-index/size/norm baselines.

## 2026-05-06 Model Eligibility Update

Added metadata-only model eligibility at
`../shared/results/hybrid_model_eligibility_20260506/`. HBSM can now name the
frontier-hybrid targets and their approximate weight sizes, but it cannot run B1
without loaded model weights and forward sensitivity measurements.

## 2026-05-06 Packet Builder Update

`../shared/hybrid_trace_packet_builder.py` now supports `--project hbsm` from a
JSON row packet. Real B1 packets must include the perturbation-off no-op plus
random, layer-index, parameter-count/norm, and boundary-only controls before the
checker will accept them.

The checker also requires both `boundary_flag=true` and `boundary_flag=false`,
finite sensitivity/predictor/size/norm fields, and near-zero drift for
`perturbation_off` rows. This keeps HBSM's first real packet tied to a genuine
sensitivity table rather than a control-only schema artifact.
It now also requires train/test split coverage and matched counts for
`top_decile_flag=true` and `random_top_decile=true`.

## 2026-05-06 Reviewer Pack Update

Added `paper/reviewer_pack.md` and wired the stricter B1 packet blocker into the
COLM shell. The paper now states that HBSM is not camera-ready as a standalone
paper until B1--B3 separate the mechanism from existing sensitivity tools.

## 2026-05-06 Decision-Grade Packet Hardening

Tightened the real B1 contract after COLM-style review. A real HBSM packet now
must include hash-shaped prompt and architecture provenance, aggregate
sensitivity/enrichment fields in `summary.json`, matched top-decile/random
counts, train/test counts, and the existing no-op plus baseline controls.
Resource-limited runs are diagnostic only and must use
`RESOURCE_LIMITED_NOT_PROMOTABLE`.

Decision: **B1 PROMOTION NOW REQUIRES A COMPLETE REAL SENSITIVITY TABLE**. The
next exact gate remains a live layer-sensitivity sweep on current hybrid
reasoners.

## 2026-05-06 Recomputed Gate Evaluator Update

Added `../shared/hybrid_gate_evaluators.py` and wired the HBSM real-packet
checker to recompute B1 top-decile enrichment, Fisher p-value, split counts,
control coverage, and cheap-predictor Spearman correlation from
`raw_rows.jsonl`. The builder now rejects non-boolean
`boundary_flag`, `top_decile_flag`, and `random_top_decile` values instead of
coercing strings. The shared Mac smoke prompt manifest is
`../shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl` with SHA-256
`48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

Decision: **NEXT B1 MUST BE GENERATED FROM RAW SENSITIVITY ROWS**. The blocker
is still live model weights and forward-sensitivity measurements.

## 2026-05-06 Reviewer Novelty Note

The latest COLM-style review confirmed HBSM is wounded by KL Lens unless the
first real packet includes a same-table comparison against a KL-style ranking
and activation/outlier baselines. Synthetic B1 remains plumbing only. A real
packet that lacks those comparators should be treated as mechanism scouting for
HORN/SSQ-LR rather than as a standalone paper result.

The real-packet checker now enforces those comparator rows explicitly through
`kl_lens_rank` and `activation_outlier` controls, and it rejects inflated
"top-decile" sets by requiring exactly `ceil(10% of aggregated scoring layers)`
true flags for both measured and random top-decile rows.

Decision: **HBSM NEEDS REAL B1 PLUS KL-LENS/OUTLIER BASELINES TO STAY
STANDALONE**. The blocker remains a live layer-sensitivity sweep.

## 2026-05-06 B1 Schema-Rehearsal Upgrade

After COLM-style review, the HBSM synthetic packet now validates as a
non-promoting real-schema rehearsal with `schema_rehearsal: true` and decision
`SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1`. The shared checker now
counts top-decile and random-top-decile cardinality on aggregated
`(model_id, layer)` scoring rows, matching the B1 evaluator and paper contract,
rather than on raw prompt rows.

The evaluator now derives B1 measured top-decile membership from aggregated
`kl_or_nll_drift` rather than trusting caller-supplied labels. The real checker
still requires supplied `top_decile_flag` fields for auditability, but rejects a
packet if those fields disagree with the measured drift ranking on any
individual `boundary_only` prompt row.

Decision: **HBSM PACKET PLUMBING IS READY FOR A REAL SENSITIVITY TABLE**. The
blocker is still a live hybrid forward-sensitivity sweep with the same controls.

## 2026-05-07 Resource-Limited Builder Guard

Updated the shared trace packet builder so any HBSM row packet whose metadata
contains `resource_limit_note` writes a `RESOURCE_LIMITED_NOT_PROMOTABLE_...`
decision automatically. This keeps partial Mac sensitivity tables useful for
schema and comparator debugging while preventing a small or incomplete table
from promoting B1.

Decision: **RESOURCE-LIMITED B1 TABLES CAN DEBUG THE PIPELINE BUT CANNOT PROMOTE
B1**. The blocker remains a complete real hybrid forward-sensitivity table with
KL-style and activation/outlier comparators.

## 2026-05-07 Architecture Hash Provenance Guard

The shared real-packet checker now verifies that a non-rehearsal HBSM packet's
`model_id` and `architecture_map_hash` match the shared architecture map
artifact. A B1 sensitivity table can no longer pass packet validation by using a
syntactically valid but unrelated architecture hash.

Decision: **B1 SENSITIVITY ROWS MUST BE TIED TO A KNOWN HYBRID ARCHITECTURE
MAP**. The blocker remains the same real forward-sensitivity table.

## 2026-05-07 B1/B2 Wording Separation

Cleaned the HBSM paper/reviewer-pack wording so the current synthetic artifact
is described as a B1 real-schema rehearsal only. It can carry B2-related
predictor fields for future checks, but it does not rehearse or promote the B2
cheap-predictor gate.

Decision: **CURRENT HBSM EVIDENCE IS B1 PACKET PLUMBING ONLY**. B2 remains a
future real-packet gate after sensitivity heterogeneity is established.

## 2026-05-07 Trace-Plan Artifact

Added `../shared/hybrid_trace_plan.py` and generated
`../shared/results/hybrid_trace_plan_20260507/`. For HBSM, the plan enumerates
2,304 B1 sensitivity rows across frozen prompts, all mapped hybrid layers,
map-derived boundary flags, train/test splits, and layer-aligned
comparator/control rows. The real-packet checker now requires a
`trace_plan_hash` for
non-rehearsal packets, so future B1 rows must cite the exact plan JSONL used
during sensitivity capture.

Decision: **B1 SENSITIVITY CAPTURE IS NOW OPERATIONALLY SPECIFIED BUT STILL NOT
RUN**. The next exact gate remains a real forward-sensitivity row packet built
from those planned rows and checked with `check_gate_packet --mode real --project
hbsm`.

## 2026-05-07 Capture-Manifest Templates

Added `../shared/hybrid_trace_capture_manifest.py` and generated
`../shared/results/hybrid_capture_manifests_20260507/`. For HBSM, the artifact
provides per-model row-packet templates with B1 boundary-only rows and required
comparator/control rows. Granite templates contain 720 planned entries each,
while the Qwen3-Next template contains 864 entries because its architecture map
exposes more scored layers.

Decision: **B1 CAPTURE NOW HAS A FILL-IN TEMPLATE BUT STILL NO MODEL
EVIDENCE**. The next exact gate is to fill one HBSM row-packet template from a
real forward-sensitivity capture, build the packet, and validate it with
`check_gate_packet --mode real --project hbsm`.

## 2026-05-07 Model-Alias and Template-Sentinel Guard

The shared architecture maps now carry canonical model IDs and registered
served/HF aliases, so real B1 packets can preserve a served ID as
`served_model_id` while validating rows against the canonical map ID. The HBSM
builder now rejects top-level capture templates and recursively rejects
unfilled `TO_FILL_BEFORE_CAPTURE` markers before attempting to coerce metric
fields.

Decision: **B1 ROW PACKETS NOW FAIL EARLY ON UNFILLED TEMPLATES OR UNKNOWN
MODEL IDS**. The blocker remains a real forward-sensitivity table.

## 2026-05-07 Trace-Plan Path Guard

The shared real-packet checker now rejects non-rehearsal HBSM packets that omit
`trace_plan_path`. B1 sensitivity rows, KL-style comparators, activation/outlier
comparators, random baselines, and no-op perturbation controls must all be
checkable against a cited frozen row plan.

Decision: **B1 REAL SENSITIVITY ROWS MUST BE TRACE-PLAN-CHECKABLE**. The
blocker remains a real forward-sensitivity table.

## 2026-05-07 Promotable Trace-Plan Hash Guard

After reviewer audit, B1 real packets cannot promote by pointing
`trace_plan_path` at a caller-created sensitivity table plan. Non-resource-
limited packets must cite trace-plan rows whose file SHA-256 equals the
registered shared HBSM `trace_plan_hash`. Resource-limited subset tables remain
allowed only as `RESOURCE_LIMITED_NOT_PROMOTABLE` diagnostics.

Decision: **B1 PROMOTION MUST USE THE REGISTERED FROZEN PLAN CONTENT**. The
blocker remains a real forward-sensitivity table with stronger comparator
controls.

## 2026-05-07 Source Sensitivity Artifact Guard

After COLM-style artifact review, the HBSM packet builder now copies the source
forward-sensitivity row packet into `evidence/hbsm_row_packet.json`, records
`source_row_packet_sha256` in `config.json`, and writes
`evidence/source_manifest.json`. The real checker verifies those hashes before
interpreting B1 rows, so a sensitivity table cannot promote from orphaned row
JSON without a reviewable source artifact.

Decision: **B1 ROWS MUST BE HASHED BACK TO THEIR SOURCE SENSITIVITY PACKET**.
The blocker remains a real forward-sensitivity table with stronger comparator
controls.

## 2026-05-07 B1 Scout Decision

Both Granite Tiny resource-limited B1 scouts fail the current sensitivity
heterogeneity story. The one-prompt smoke has 56 checker-passing rows from one
prompt and 8 layers, top drift layer `5`, Fisher p `0.375`, and cheap-predictor
Spearman `-0.476`. The two-prompt prompt-repeat scout has 64 checker-passing
rows, Fisher p `1.0`, boundary top-decile count `0`, non-boundary top-decile
count `1`, and cheap-predictor Spearman `-0.667`.
The 8-layer smoke is too small to formally satisfy the full Fisher gate; the
branch is weakened because the observed ranking and cheap-predictor direction
are wrong, not because this tiny subset alone formally falsifies B1.

Decision: **HBSM IS WEAKENED AND SHOULD NOT SCALE B1 AS-IS**. Do not run a
larger B1 table or GPU validation until a narrower mechanism hypothesis is
pre-registered. If the mechanism remains indistinguishable from HORN's H2
question, fold HBSM into the HORN control appendix rather than keeping a
standalone branch.
