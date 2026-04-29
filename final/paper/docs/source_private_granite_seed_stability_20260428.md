# Granite Seed Stability And Trace Ablation

- date: `2026-04-28`
- gate: `granite33_2b_seed_stability_and_trace_ablation_20260428`
- status: passed strict-prompt seed repeat; raw-log/no-trace collapses as intended
- scale rung: medium cross-family prompt-contract confirmation

## Readiness

This gate strengthens the scoped full-paper case by hardening the weakest
positive non-Qwen source emitter. It does not remove the MoE/FP8 gap or prove
universal prompt-invariant packet emission.

## Result

`ibm-granite/granite-3.3-2b-instruct` repeats its strict `trace_no_hint` n160
boundary result on seed31:

- matched model packet: `101/160 = 0.631`
- target-only: `40/160 = 0.250`
- best source-destroying control: `40/160 = 0.250`
- packet valid rate: `0.537`
- exact-ID parity: `true`
- matched-minus-best-control: `+0.381`
- p50 packet latency: `3691 ms`

The paired `raw_log_no_trace` seed31 source-signal ablation collapses:

- matched model packet: `40/160 = 0.250`
- target-only: `40/160 = 0.250`
- best source-destroying control: `40/160 = 0.250`
- packet valid rate: `0.000`
- exact-ID parity: `true`
- p50 packet latency: `2857 ms`

## Interpretation

Granite should be framed as a stable but weaker prompt-contract source emitter.
It is useful precisely because it is not an easy perfect row: it shows the
protocol can transfer source-private information outside Qwen/Gemma, but also
that instruction-following quality controls packet validity and accuracy.

The raw-log/no-trace collapse rules out a wrapper-only explanation for the
Granite gain. The private diagnostic trace is necessary for successful packet
emission.

## Artifacts

- `results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed31/summary.json`
- `results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed31/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/granite33_2b_trace_no_hint_n160_cpu_seed31/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/granite33_2b_raw_log_no_trace_n160_cpu_seed31/summary.json`
- `results/source_private_latest_model_matrix_20260428/granite33_2b_raw_log_no_trace_n160_cpu_seed31/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/granite33_2b_raw_log_no_trace_n160_cpu_seed31/predictions.jsonl`

## Next Gate

The next highest-value local gate is target-decoder n160 on core and held-out
surfaces. That attacks the remaining hand-coded decoder objection more directly
than another source-emitter repeat.
