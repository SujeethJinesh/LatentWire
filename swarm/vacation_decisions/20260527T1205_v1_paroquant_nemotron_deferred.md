# V1 ParoQuant-on-Nemotron Throughput Deferral

## Decision

`DEFERRED_V1_THROUGHPUT_CAP_PROJECTED`

V1 did not produce a valid ParoQuant-on-Nemotron recovery estimate. The run was
stopped after the observed throughput made it clear that the full 12-trace
packet could not complete within the preregistered 5 GPU-hour cap.

## Run Directory

`experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260527T1125Z`

## Observed Progress

- Run started: `2026-05-27T10:09:58Z`
- ParoQuant transform plus trace 0 completed: `2026-05-27T11:08:32Z`
- Trace 1 completed: `2026-05-27T12:00:52Z`
- Run stopped: `2026-05-27T12:04:14Z`
- Completed ParoQuant score traces: 2 / 12
- Observed trace-1 scoring time: 52.3 minutes
- GPU hours consumed: about 1.90

## Rationale

The local algorithmic ParoQuant reproduction completed the expensive Nemotron
weight transform, then scored trace 1 at roughly 52 minutes per 10K-token
teacher-forced scoring pass. At that rate, the remaining 10 traces would
require roughly 8.7 more GPU hours after the already-spent setup, exceeding the
V1 cap before any valid 12-trace bootstrap CI could be computed.

This is not evidence that ParoQuant succeeds or fails on Nemotron. No paper
claim is authorized from this partial packet.

## Queue Impact

Move to V2: M11b top-10 on DeepSeek-R1-Distill-Qwen-1.5B and Falcon-H1-0.5B.
V2 remains higher-value under the current budget because the two models are
small enough to plausibly complete within the 10 GPU-hour cap and directly test
whether M11b transfers beyond Nemotron.
