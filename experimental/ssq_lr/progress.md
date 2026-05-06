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
