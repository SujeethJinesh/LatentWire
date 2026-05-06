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
(`max_length={64,96}`, `sink_tokens={2,4}`) also keeps all-rank2 positive, as
does a 48-trace trace-level frozen split repeat. The strongest downstream Mac
controls now run 48 traces, lengths 64/96, sink counts 2/4, split seeds 0/1/2,
and separately fit distilgpt2/OPT-125M predictors. Exact sink-logit replacement
remains a no-op, and rank-2 is closer than position-only in loss drift and KL in
every model/config row, but top-1 disagreement remains non-negligible. Native
work should preserve aggregate, per-head, and downstream-control readouts
because the trace-level head win rate remains a risk and the downstream patch is
still only a quality-control diagnostic. A 48-trace downstream rank frontier at
length 96/sink4 shows ranks 1/2/4/8 monotonically reduce drift, but rank4/rank8
lose the simple multiply-add wedge against exact four-sink QK.

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

1. rank-2 preserves the Mac-local quality-control envelope: mean output
   relative-L2 must be no worse than 0.15, rank-2 must beat position-only in
   every matched model/shape group on output relative-L2, downstream loss
   drift, and KL-to-exact, and top-1 disagreement aggregated over all measured
   rows in the packet must stay at or below 0.15; model/shape subgroup values
   must also be reported so the global aggregate cannot hide a collapsed row;
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

## Packet Validator

Before citing any native numbers in the paper, run the Mac-local packet
validator on the returned directory:

```bash
./venv_arm64/bin/python experimental/sinkaware/phase2/check_native_gpu_packet.py \
  --run-dir "$SINKAWARE_GPU_PACKET" \
  | tee "$SINKAWARE_GPU_PACKET/artifact_check.json"
```

A `PASS` only means the packet is complete enough for review. It is not a speed,
HBM, quality, or promotion claim.

The checker rejects:

- missing or empty required artifacts;
- placeholder/TODO-filled packet files;
- `metadata.json` that does not identify a native CUDA/NVIDIA environment;
- missing rows for any runbook row in `quality_drift.csv`,
  `quality_drift_by_head.csv`, `latency.csv`, or `ncu_summary.csv`;
- `metadata.json` model values that do not match the measured CSV model groups;
- `metadata.json` sequence shapes that do not match the measured CSV
  sequence_length/batch_size groups;
- missing required comparison rows within any measured
  model/sequence_length/batch_size group;
- fewer than three distinct `run_id` values for any
  row/model/sequence_length/batch_size group in `latency.csv`;
- mismatched row/model/sequence_length/batch_size groups across quality,
  per-head quality, latency, and NCU CSV files;
- non-numeric quality, latency, or NCU metric cells;
- `decision.md` without an explicit promote/kill decision, rank-2 quality
  threshold, and speed/memory/HBM evidence discussion.

Use these canonical row ids in CSV files:

- `exact_attention`
- `exact_fixed_sink_decomposition`
- `rank2_sink_logit_predictor`
- `position_only_predictor`

## Current Mac Inputs

- `real_qk_sink_softmax_output_probe.md`
- `rank2_split_stability_gate.md`
- `rank2_length_sink_sweep_gate.md`
- `rank2_trace_frozen_split_gate.md`
- `rank2_cross_model_length_stability_gate.md`
- `qk_sink_cost_model.md`
- `decomposition_decision.md`
- `downstream_quality_control_gate_traces48_len64_sink2.md`
- `downstream_quality_control_gate_traces48_len96_sink2.md`
- `downstream_quality_control_gate_traces48_len64_sink4.md`
- `downstream_quality_control_gate_traces48_len96_sink4.md`
- `downstream_rank_frontier_traces48_len96_sink4.md`
