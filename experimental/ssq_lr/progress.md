# SSQ-LR Progress

## Current Supersession Note

The current SSQ-LR recipe is stopped by the 2026-05-07 no-retuning S3 transfer
failure. Older "next gate" entries below are historical. Admissible local work
is limited to reproducing existing stop packets, documentation/test hardening,
or writing a new preregistration for a genuinely new rescue rule before any new
rows are generated.

## 2026-05-06

Status: **HISTORICAL / superseded Mac-gate setup**.

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
method paper until non-resource-limited, passing S1--S3 evidence exists.

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

## 2026-05-07 Held-Out S1b / S2 Scout Decision

The Granite Tiny held-out S1b packet is the current positive Mac signal:
192 saved-tensor rows over 12 held-out prompts and layers `0`, `12`, `18`,
and `30`, with primary layers `0`, `12`, and `30` passing at selected ratio
`2.459`, CI low `1.861`, and Holm p-min `2.78e-05`.

The current S2 recipes do not promote. MXFP4 preserves BF16 argmax on the short
continuation replay but reaches only `3.765x`--`3.938x` once scale bytes are
counted, missing the preregistered `>=4x` state-memory gate. INT3 can clear the
byte gate in the 4-prompt block-256 scout (`5.224x`, zero argmax delta), but
the 12-prompt held-out block-64/block-256 scouts lose argmax fidelity.

Decision: **SSQ-LR IS WEAKENED, NOT GPU-READY**. S1b heterogeneity stays alive,
but the next gate must be a new frozen sub-4-bit/native-packed state recipe
that clears S2 with paired quality bounds. Do not promote the current MXFP4 or
INT3 scouts to GPU.

## 2026-05-07 S2b Mixed-Block Recipe Patch

The highest-value Mac-side S2 follow-up is now a bounded recipe test rather
than a gate change. The S2 scout supports two new candidate rows:
`mixed_int3_mxfp4_low_error_10pct` and
`mixed_int3_mxfp4_low_error_25pct`. For each recurrent state tensor, the
allocator quantizes blocks with INT3 and MXFP4, stores the lowest INT3-error
blocks as INT3, stores the remaining blocks as MXFP4, and counts both FP16
scale bytes and a one-bit-per-block precision mask in `metadata_bytes`.

This keeps the selector deployable: it uses only block-local quantization error
against the current state tensor, not labels or downstream logits. The first
gate question is whether a small INT3 fraction can push the MXFP4-like row over
the preregistered `>=4x` byte threshold without reproducing the 12-prompt INT3
argmax failure. If both mixed-block rows fail, the next bounded diagnostic is a
single-layer S2 localization scout before freezing any layer-selective recipe.

Scratch-only capability check: a one-prompt local run selected
`mixed_int3_mxfp4_low_error_10pct` with `4.034x` counted state-memory reduction,
zero BF16-argmax delta, and `0.00443` NLL-delta CI high. This only proves the
new scout path is executable and byte-feasible on the current Mac cache; it is
not promotable evidence. The actual S2b decision requires the frozen 12-prompt
held-out replay packet.

## 2026-05-07 S2b Held-Out Replay Result

Packet:
`experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_20260507/`.
The executable follow-up contract accepts the packet with
`PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`, but the packet remains explicitly
resource-limited and not promotable to GPU.

Selected recipe: `mixed_int3_mxfp4_low_error_25pct`.

- counted state-memory reduction: `4.192x`
- selected accuracy CI high: `0.000000`
- selected NLL-delta CI high: `0.03956`
- prompts: `12`
- rows: `156`

Decision after this short-window packet: **S2 WAS REVIVED AS A MAC CANDIDATE,
NOT GPU-READY**. The next packet must test the same recipe on a longer
continuation window before any GPU discussion.

## 2026-05-07 S2b Longer-Window Replay Result

Packet:
`experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_20260507/`.
The executable follow-up contract rejects the packet with
`FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY`.

Selected recipe: `mixed_int3_mxfp4_low_error_25pct`.

- counted state-memory reduction: `4.192x`
- selected accuracy CI high: `0.066667`
- selected NLL-delta CI high: `0.07641`
- prompts: `12`
- rows: `156`
- replay shape: `--max-input-tokens 24 --prefix-tokens 8`

Decision after this packet: **THE ALL-PRIMARY MIXED RECIPE IS STOPPED BEFORE
GPU**. S1b heterogeneity remains alive, but the three-layer `0,12,30` recipe is
not viable. The only bounded Mac continuation is single-layer S2 localization;
do not lower the `>=4x` byte threshold or send the current mixed recipe to GPU.

## 2026-05-07 S2b Layer-Localization Replay Result

Packets:

- `experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer0_20260507/`
- `experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer12_20260507/`
- `experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layer30_20260507/`
- `experimental/shared/results/ssq_lr_s2_state_replay_scout_mixed_block256_12p_ctx24_layers0_30_20260507/`

Longer-window localization results:

| Layers | Decision | Selected recipe | Memory | Accuracy CI high | NLL CI high |
|---|---|---|---:|---:|---:|
| `0` | pass | `int3_primary_state_block_scaled` | `5.224x` | `0.000000` | `0.04294` |
| `12` | fail | `fp8_e4m3_primary_state` | `2.000x` | `0.066667` | `0.06903` |
| `30` | pass | `int3_primary_state_block_scaled` | `5.224x` | `0.000000` | `0.04505` |
| `0,30` | pass | `int3_primary_state_block_scaled` | `5.224x` | `0.000000` | `0.04294` |

Decision: **SSQ-LR IS ALIVE ONLY AS A LAYER-SELECTIVE MAC CANDIDATE**. Exclude
layer `12`. Pure INT3 on layers `0,30` became the immediate candidate, but the
stricter S3 prefilter below supersedes it with a mixed INT3/MXFP4 recipe. Still
no GPU promotion.

## 2026-05-07 S3 Prefilter Replay

Packet:

- `experimental/shared/results/ssq_lr_s3_prefilter_granite_tiny_layers0_30_20260507/`

Readout:

| Recipe | Memory | Accuracy CI high | NLL CI high | Decision |
|---|---:|---:|---:|---|
| `int3_primary_state_block_scaled` | `5.224x` | `0.105263` | `0.13503` | weakened |
| `mixed_int3_mxfp4_low_error_25pct` | `4.192x` | `0.000000` | `0.05044` | selected |

Decision: **PURE INT3 WAS WEAKENED; MIXED 25% INT3/MXFP4 WAS THE PRE-TRANSFER
CANDIDATE, NOW SUPERSEDED**. The stricter Granite Tiny replay kept layers
`0,30`, but the Granite 350M no-retuning transfer replay below stopped this
candidate before GPU.

## 2026-05-07 S3 Transfer Prefilter

Packet:

- `experimental/shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507/`

Readout:

| Field | Value |
|---|---:|
| Decision | `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER` |
| Validator status | clean under `followup_gate_contracts --gate ssq_lr_s3` |
| Frozen recipe | `mixed_int3_mxfp4_low_error_25pct`, layers `0,30`, block size `256` |
| Frozen recipe hash | `sha256:df4c3f234306c6cc98c07073ab21a88e67a186ecca62436d49507a95d62bdbc1` |
| Transfer model count | `1` |
| Passing model count | `1` |
| Retuned row count | `0` |
| Max NLL delta reported by S3 checker | `0.050439` |

Decision: **CACHE-ONLY S3 WAS LOCALLY BLOCKED**. This prefilter had one
complete local hybrid model (`ibm-granite/granite-4.0-h-tiny`) and recorded
Granite Small, Granite Small FP8, and Qwen3-Next as config-only caches. It is
now superseded by the Granite 350M transfer replay below, which removes the
cache-only blocker and exposes a quality/transfer failure.

## 2026-05-07 Granite 350M Transfer Replay

Packets:

- `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507/`
- `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer0_20260507/`
- `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layer30_20260507/`
- `experimental/shared/results/ssq_lr_s3_source_granite_tiny_12p_layer0_ctx32_20260507/`
- `experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507/`
- `experimental/shared/results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507/`

Readout:

| Packet | Decision | Selected/live recipe | Memory | Accuracy CI high | NLL CI high |
|---|---|---|---:|---:|---:|
| Granite 350M layers `0,30` | `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | fallback `int8_primary_state_block64` | `1.984x` | `0.000000` | `0.008906` |
| Granite 350M layer `0` | `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | `int3_primary_state_block_scaled` | `5.224x` | `0.000000` | `0.015687` |
| Granite 350M layer `30` | `FAIL_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | fallback `int8_primary_state_block64` | `1.984x` | `0.000000` | `0.008893` |
| Granite Tiny layer `0`, matched context | `PASS_REAL_SSQ_LR_S2_QUANTIZATION_SENSITIVITY` | `mixed_int3_mxfp4_low_error_25pct` | `4.192x` | `0.000000` | `0.041124` |
| Local S3, layer `0`, frozen mixed25 | `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER` | source-passing mixed25 | -- | `0.066667` | `0.041124` |
| Local S3, layer `0`, frozen INT3 diagnostic | `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER` | transfer-passing INT3 | -- | `0.052632` | `0.123213` |

Decision: **SSQ-LR S3 FAILS ON THE CURRENT MAC TRANSFER SURFACE**. The
second complete local hybrid model (`ibm-granite/granite-4.0-h-350m`) removes
the earlier cache-only blocker and exposes a real transfer failure. The frozen
`0,30` mixed recipe does not preserve the 12-prompt transfer surface. The
layer-0 diagnostic is also not a rescue: Granite Tiny and Granite 350M prefer
different low-bit recipes, so no single frozen recipe passes both models under
the current S3 contract. Do not GPU-promote SSQ-LR without a new
preregistered recipe/layer rule.

Current recipe status: **DEMOTED / STOPPED FOR GPU HANDOFF**. The stopped
recipe is `mixed_int3_mxfp4_low_error_25pct` on layers `0,30`; the layer-0
mixed25 and INT3 follow-ups are post-hoc diagnostics and cannot be converted
into promotion rows.

No additional rows under the current recipe are admissible. Only one bounded
Mac rescue is admissible before revival or GPU reconsideration, and it must be
a new preregistered branch: write a new preregistration before running it,
freeze a fresh held-out prompt file, freeze a single layer-selection rule and
recipe-selection rule before transfer rows are inspected, keep the `>=4x`
counted state-memory threshold and the S3 no-retuning transfer contract,
include Granite Tiny and Granite 350M with at least 12 prompts each, and add
verbosity/length drift before any GPU handoff. If that rescue fails, SSQ-LR
should remain diagnostic evidence rather than an active positive-method paper.
