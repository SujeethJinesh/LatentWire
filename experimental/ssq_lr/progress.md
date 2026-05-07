# SSQ-LR Progress

## 2026-05-06

Status: **NEW / Mac gates pending**.

Added and ran a deterministic synthetic S1 packet. This has now been upgraded to
a real-schema rehearsal:

- script: `phase2/ssq_lr_synthetic_s1_gate.py`
- packet: `phase2/results/ssq_lr_synthetic_s1/`
- decision: `SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1`
- rows: `288` (`12` prompts x `6` recurrent layers x `4` buckets)
- real checker: passes `--mode real --project ssq_lr`

Interpretation: synthetic-only schema validation. It exercises the real S1 row
schema, provenance fields, recomputed evaluator summary, and checker path, but
does not promote the branch or replace real hybrid SSM state dumps.

Next exact gate: S1 state distribution heterogeneity on the smallest available
hybrid model traces using shared activation/state dump utilities.

## 2026-05-06 Architecture Provenance Update

Added shared config-derived architecture maps at
`../shared/results/hybrid_architecture_maps_20260506/`. Real S1 packets must
include the corresponding `architecture_map_hash` in `config.json`; this keeps
state rows tied to an explicit hybrid layer map even before GPU validation.

## 2026-05-06 Model Eligibility Update

Added metadata-only model eligibility at
`../shared/results/hybrid_model_eligibility_20260506/`. The smallest live target
found is `ibm-granite/granite-4.0-h-tiny` at 12.93 GB of safetensors, but it is
not cached repo-locally. SSQ-LR therefore cannot produce a real S1 state packet
on the current Mac without first downloading/loading a live hybrid model.

## 2026-05-06 Real Packet Admissibility Update

The shared checker now rejects real SSQ-LR packets unless rows cover `prefill_end`, `2k_or_end`, `8k_or_end`, and `final_minus_128` buckets and include at least 12 distinct prompt IDs,
unless `config.json` explicitly records a resource-limit note. This makes S1 a
real state-distribution gate instead of a one-row schema check.

## 2026-05-06 Reviewer Pack Update

Added `paper/reviewer_pack.md` and wired the stricter real-packet blocker into
the COLM shell. The paper now states that SSQ-LR is not camera-ready as a
method paper until real S1--S3 evidence exists.

## 2026-05-06 Decision-Grade Packet Hardening

Tightened the real S1 contract after COLM-style review. A real SSQ-LR packet now
must include `prompt_ids_hash` and `architecture_map_hash` provenance,
project-specific aggregate fields in `summary.json`, and complete coverage of
all preregistered buckets for every `(prompt_id, layer)` pair. Resource-limited
runs are still admissible for diagnosis, but their decision must start with
`RESOURCE_LIMITED_NOT_PROMOTABLE` and they cannot promote S1.

Decision: **S1 PROMOTION NOW REQUIRES A COMPLETE REAL STATE MATRIX**. The next
exact gate is unchanged: live hybrid SSM-state dumps on the smallest available
hybrid model.

## 2026-05-06 Recomputed Gate Evaluator Update

Added `../shared/hybrid_gate_evaluators.py` and wired the SSQ-LR real-packet
checker to recompute S1 summary fields from `raw_rows.jsonl`. A real packet now
cannot promote by hand-filling late/early ratios or passing-layer counts in
`summary.json`; stale or inconsistent S1 summaries are rejected. The shared
Mac smoke prompt manifest is
`../shared/prompts/hybrid_reasoning_smoke_12_20260506.jsonl` with SHA-256
`48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

Decision: **NEXT S1 MUST BE GENERATED FROM RAW STATE ROWS**. The blocker is
still live hybrid SSM-state dumps.

## 2026-05-06 S1 Distribution-Test Alignment

After COLM-style review, the shared S1 evaluator now implements the
preregistered second pass path instead of hard-coding it off. It computes
per-layer two-sample KS p-values between `prefill_end` and `final_minus_128`
state metrics, applies Holm correction over layer/metric tests, and exposes
`distribution_passing_layer_count`, `magnitude_gate_pass`, and
`distribution_gate_pass` in the recomputed summary contract.

Follow-up hardening now requires the distribution-only path to clear a 1.25x
per-layer effect-size floor on the Holm-significant layer/metric tests counted
by `distribution_passing_layer_count`; a global selected-ratio spike in one
layer cannot rescue near-zero but statistically significant shifts elsewhere.

Decision: **S1 CAN PASS BY MAGNITUDE OR HOLM-CORRECTED DISTRIBUTION SHIFT, BUT
ONLY FROM RAW REAL STATE ROWS**. The blocker remains the same live hybrid
SSM-state dump.

## 2026-05-07 Resource-Limited Builder Guard

Updated the shared trace packet builder so any SSQ-LR tensor packet whose
metadata contains `resource_limit_note` writes a
`RESOURCE_LIMITED_NOT_PROMOTABLE_...` decision automatically. This closes a
pre-GPU failure mode where a two-prompt Mac hook smoke test could inherit a
passing evaluator status in `summary.json` even though the checker policy says
resource-limited packets are diagnostic only.

Decision: **RESOURCE-LIMITED S1 SMOKE PACKETS CAN TEST HOOKS BUT CANNOT PROMOTE
S1**. The blocker remains the full real hybrid SSM-state dump with at least 12
fixed prompts.

## 2026-05-07 Architecture Hash Provenance Guard

The shared real-packet checker now verifies that a non-rehearsal SSQ-LR packet's
`model_id` and `architecture_map_hash` match the shared architecture map
artifact, not just the `sha256:<64-hex>` format. A forged or unrelated
architecture hash is rejected before S1 interpretation.

Decision: **S1 STATE ROWS MUST BE TIED TO A KNOWN HYBRID ARCHITECTURE MAP**.
The blocker remains the same real SSM-state dump.

## 2026-05-07 Trace-Plan Artifact

Added `../shared/hybrid_trace_plan.py` and generated
`../shared/results/hybrid_trace_plan_20260507/`. For SSQ-LR, the plan enumerates
5,184 required S1 capture rows across the frozen 12-prompt smoke manifest,
shared architecture-map models, recurrent SSM layers, and the four
preregistered buckets (`prefill_end`, `2k_or_end`, `8k_or_end`, and
`final_minus_128`). The real-packet checker now requires a `trace_plan_hash`
for non-rehearsal packets, so future S1 rows must cite the exact plan JSONL
used during capture.

Decision: **S1 TRACE CAPTURE IS NOW OPERATIONALLY SPECIFIED BUT STILL NOT RUN**.
The next exact gate remains a real tensor packet built from those planned rows
and checked with `check_gate_packet --mode real --project ssq_lr`.

## 2026-05-07 Capture-Manifest Templates

Added `../shared/hybrid_trace_capture_manifest.py` and generated
`../shared/results/hybrid_capture_manifests_20260507/`. For SSQ-LR, the
artifact provides one per-model fill-in metadata template with 1,728 planned S1
entries per model. These templates are derived from the frozen trace plan and
carry `trace_plan_hash`, prompt provenance, architecture hashes, bucket names,
SSM layer IDs, and tensor filename placeholders.

Decision: **S1 CAPTURE NOW HAS A FILL-IN TEMPLATE BUT STILL NO MODEL
EVIDENCE**. The next exact gate is to fill one SSQ-LR template from a real
hybrid SSM-state capture, build the packet, and validate it with
`check_gate_packet --mode real --project ssq_lr`.

## 2026-05-07 Model-Alias Guard

The shared architecture maps now carry canonical model IDs and registered
served/HF aliases. The packet builder canonicalizes a served ID such as
`ibm-granite/granite-4.0-h-tiny` to the architecture-map slug while preserving
the served value as `served_model_id` in `config.json`; the checker accepts only
registered aliases with the matching architecture hash.

Decision: **S1 PACKETS CAN USE SERVED HF IDS WITHOUT BREAKING PROVENANCE**. The
blocker remains a real hybrid SSM-state capture.

## 2026-05-07 Trace-Plan Path Guard

The shared real-packet checker now rejects non-rehearsal SSQ-LR packets that
omit `trace_plan_path`. A correct `trace_plan_hash` alone is no longer enough:
the checker must be able to load the cited row plan and verify that observed
state rows are neither off-plan nor duplicated.

Decision: **S1 REAL ROWS MUST BE TRACE-PLAN-CHECKABLE, NOT JUST HASH-SHAPED**.
The blocker remains a real tensor packet generated from the frozen S1 plan.

## 2026-05-07 Tensor Provenance Guard

`activation_dumper.py` now writes `tensor_manifest.json` with original state
hook names, packet-safe storage names, SHA-256 hashes, dtypes, shapes, and
element counts. SSQ-LR builder rows copy that provenance into every S1 row and
the checker requires `state_shape` to match `tensor_shape`.

Decision: **S1 STATE METRICS MUST BE HASHED BACK TO THEIR SAVED TENSORS**. The
blocker remains a real tensor packet generated from the frozen S1 plan.

## 2026-05-07 Promotable Trace-Plan Hash Guard

After reviewer audit, the real-packet checker no longer treats an arbitrary
`trace_plan_path` as sufficient for a promotable S1 packet. A non-resource-
limited packet must cite trace-plan rows whose file SHA-256 equals the
registered shared `trace_plan_hash`. Small caller-created subset plans are only
accepted when the packet is explicitly marked
`RESOURCE_LIMITED_NOT_PROMOTABLE`.

Decision: **S1 PROMOTION CANNOT SELF-CERTIFY WITH A CALLER-SUPPLIED PLAN**. The
blocker remains a real tensor packet generated from the frozen S1 plan.

## 2026-05-07 Saved Tensor Metric Guard

After COLM-style artifact review, the shared packet builder now copies the
saved SSQ-LR tensor manifest and `.pt` files into every built real packet. The
real checker reloads each cited tensor and recomputes `max_abs`, `rms`, `std`,
`kurtosis`, and `outlier_mass`; a row whose metrics do not match the saved
tensor bytes is rejected even if its SHA-256 provenance fields are well formed.

Decision: **S1 ROW METRICS MUST BE RECOMPUTABLE FROM SAVED STATE TENSORS**. The
blocker remains a real tensor packet generated from the frozen S1 plan.
