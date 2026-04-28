# Source-Private Hidden-Repair Cross-Model Gate

- date: `2026-04-28`
- status: promoted cross-model smoke, not yet ICLR-ready
- live branch: source-private hidden-repair packet handoff
- scale rung: cross-model smoke

## Question

Does the hidden-repair packet protocol survive when the source emitter changes
across small instruction-tuned model families?

## Setup

The gate reuses the frozen `64` example benchmark from
`results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl`.
Each source model sees the private hidden execution log and helper-line
`REPAIR_DIAG` field, then emits a two-character repair packet. The target
decoder, candidate pool, exact IDs, and source-destroying controls are
unchanged across models.

## Commands

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_cross_model_20260428/qwen25_0_5b_helper \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 64 \
  --seed 28 \
  --max-new-tokens 8 \
  --no-enable-thinking
```

The same command shape was run for:

- `Qwen/Qwen3-0.6B`
- `microsoft/Phi-3-mini-4k-instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Results

| Run | Model | Family | Pass | Matched | Target-only | Best control | Valid packets | Mean bytes | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | `true` | 0.984 | 0.250 | 0.250 | 0.984 | 1.97 | 330.85 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 312.52 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 2.00 | 511.35 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 0.00 | 561.73 |

Aggregate pass: `true`. Three of four source models pass, including one
non-Qwen model family. TinyLlama fails cleanly as a negative capability row: it
mostly repeats the instruction phrase rather than emitting the diagnostic code.

## Interpretation

This removes the simplest one-model prompt-artifact objection to the hidden
repair packet smoke. Capable instruction-tuned source models can copy the
private diagnostic from actual hidden execution logs into a compact packet, and
the target-side gains still disappear under source-destroying controls.

The remaining limitation is still central: the task is helper-line and
metadata-assisted. The result supports a protocol-assisted private hidden-log
packet claim, not a broad claim that arbitrary models infer repair facts from
raw logs without a shared protocol.

## Decision

Promote the branch to cross-model smoke on capable source emitters. Keep
TinyLlama as a capability boundary and do not hide the failure.

## Next Gate

`source_private_hidden_repair_packet_weakened_helper_20260428`:

- use the same frozen hidden-repair benchmark
- remove the copied helper line from the prompt first, while keeping the
  `REPAIR_DIAG` line inside the private log
- then test a harder no-helper variant where the packet must be inferred from
  masked expected/actual or exception fields
- pass only if at least one capable model remains above target-only by `>=0.15`
  and zero/shuffled/random/target-derived controls remain flat
