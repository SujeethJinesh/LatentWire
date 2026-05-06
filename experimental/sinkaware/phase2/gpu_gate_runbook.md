# SinkAware Native GPU Gate Runbook

- status: future native benchmark plan
- scope: compare exact attention, exact fixed-sink decomposition, and rank-2
  approximate sink-logit prediction

## Why This Gate Exists

The Mac-local evidence revived SinkAware only as an approximate low-rank branch:
rank-2 reduces held-out output relative-L2 versus position-only while staying
below the estimated multiply-add cost of exact four-sink QK. Exact static sink
reuse remains killed. The latest layer-head paired readout is mixed, so a GPU
run must preserve the per-head drift table instead of reporting only aggregate
means. A three-seed randomized token split repeat keeps all-rank2 positive, but
does not solve per-head fragility. A bounded Mac-local length/sink sweep
(`max_length={64,96}`, `sink_tokens={2,4}`) also keeps all-rank2 positive, so
does a trace-level frozen split repeat on 24 traces (`+0.0398 +/- 0.0014`
output rel-L2 improvement). Native work should preserve both aggregate and
per-head quality readouts because the trace-level head win rate remains low
(`0.287 +/- 0.018`).

The GPU gate must answer whether the approximation is useful after real kernel
costs, memory movement, and output drift are measured together.

## Rows To Run

| Row | Computes `QK_sink`? | Approximate? | Purpose |
|---|---:|---:|---|
| exact attention | yes | no | quality and speed reference |
| exact fixed-sink decomposition | yes | no | checks whether separating sink path helps layout/fusion without changing outputs |
| rank-2 sink-logit predictor | no, predicted | yes | live approximate branch |
| position-only predictor | no, predicted | yes | cheap baseline the rank-2 method must beat |

## Promotion Criteria

Promote only if all are true:

1. rank-2 output drift remains below the paper threshold selected from the
   Mac-local probe;
2. rank-2 improves speed or memory traffic over exact attention by at least 3%
   on repeated native runs;
3. rank-2 beats position-only on output drift and any quality proxy;
4. exact decomposition does not already capture the systems win without
   approximation.

Kill if rank-2 is slower than exact attention, if output drift is unbounded, or
if position-only is indistinguishable.

## Required Artifacts

- `metadata.json`: GPU, driver, CUDA, PyTorch, Triton, model, dtype, sequence
  shapes.
- `quality_drift.csv`: per-layer and mean output relative-L2, sink-mass MAE,
  attention L1.
- `quality_drift_by_head.csv`: layer/head paired output relative-L2,
  sink-mass MAE, and attention L1 versus position-only.
- `latency.csv`: paired timing for each row, batch, and sequence length.
- `ncu_summary.csv`: memory bytes, achieved occupancy, register pressure, and
  tensor/core utilization where available.
- `decision.md`: promote/kill decision with the exact threshold.

## Current Mac Inputs

- `real_qk_sink_softmax_output_probe.md`
- `rank2_split_stability_gate.md`
- `rank2_length_sink_sweep_gate.md`
- `rank2_trace_frozen_split_gate.md`
- `qk_sink_cost_model.md`
- `decomposition_decision.md`
