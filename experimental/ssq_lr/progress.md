# SSQ-LR Progress

## 2026-05-06

Status: **NEW / Mac gates pending**.

Added and ran a deterministic synthetic S1 packet:

- script: `phase2/ssq_lr_synthetic_s1_gate.py`
- packet: `phase2/results/ssq_lr_synthetic_s1/`
- decision: `SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT`
- late/early max-abs ratio: `8.461`
- late/early std ratio: `3.640`
- late/early kurtosis ratio: `3.141`

Interpretation: synthetic-only artifact validation. It fixes the S1 readout
format but does not promote the branch or replace real hybrid SSM state dumps.

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
