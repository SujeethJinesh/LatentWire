# Source-Private Hidden-Repair Holdout-Families Gate

- date: `2026-04-29`
- status: promoted held-out family confirmation, not yet ICLR-ready
- live branch: explicit source-private tool-trace packet handoff
- scale rung: held-out family confirmation

## Question

Does the medium-confirmed private tool-trace packet method survive on repair
families not used by the core benchmark templates?

## Setup

The held-out gate adds eight new repair families:

- `clamp_negative_to_zero`
- `last_value_default`
- `parse_int_default`
- `average_all_values`
- `strip_and_lower`
- `nested_key_default`
- `wrapped_index_lookup`
- `strict_positive_filter`

The gate freezes `500` held-out examples with the same target view, candidate
pool structure, `trace_no_hint` source prompt, source-destroying controls, and
Qwen3/Phi-3 source emitters used in the medium gate. Qwen3
`raw_log_no_trace` is the aligned source-signal destruction row.

## Results

Deterministic packet sweep:

| Budget bytes | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 |
| 4 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 8 | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 |
| 16 | 1.000 | 0.250 | 0.252 | 0.250 | 1.000 |
| 32 | 1.000 | 0.250 | 0.254 | 0.250 | 1.000 |

Model-produced packets with paired bootstrap intervals:

| Run | Model | Mode | Matched | Target | Best control | Valid | Mean bytes | p95 latency ms | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | 0.922 | 0.250 | 0.258 | 0.864 | 1.73 | 429.15 | [0.632, 0.712] | [0.622, 0.706] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.258 | 1.000 | 2.00 | 602.46 | [0.710, 0.788] | [0.702, 0.778] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.258 | 0.000 | 0.00 | 620.12 | [0.000, 0.000] | [-0.016, -0.002] |

## Interpretation

The result survives held-out repair families. This weakens the largest reviewer
objection to the medium gate: that the method only works because the eight core
families are template-specific. The method still depends on an explicit private
tool-trace field, but that interface generalizes across repair families,
source-model families, and 500-example frozen slices.

The remaining claim boundary is unchanged. This is not raw-log inference:
removing `REPAIR_DIAG` returns Qwen3 to target-only with zero valid packets.

## Decision

Promote the branch to held-out family confirmation. The live paper story should
now be: **explicit source-private tool-trace packets as a compact
agent-to-agent communication interface**.

## Next Gate

`source_private_hidden_repair_packet_seed_repeat_20260429`:

- repeat core-medium and held-out-family gates over at least two additional
  seeds where deterministic surfaces are frozen separately
- run Qwen3 and Phi-3 `trace_no_hint`
- keep Qwen3 `raw_log_no_trace` as the destruction row
- report paired bootstrap intervals per seed and aggregated fixed-effect
  deltas
- pass only if the positive margin and flat controls are stable across seeds
