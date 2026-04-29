# Source-Private Tool-Trace Target Decoder Progress Gate

- date: `2026-04-29`
- artifact: `results/source_private_tool_trace_target_decoder_progress_gate_20260429/`
- script: `scripts/run_source_private_tool_trace_target_decoder_smoke.py`
- test: `tests/test_run_source_private_tool_trace_target_decoder_smoke.py`
- scale rung: smoke / receiver harness hardening

## Purpose

This gate addresses the hand-coded-decoder objection. The target-side decoder is
a frozen local `Qwen/Qwen3-0.6B` model that sees the public candidate metadata
and a source packet. The script now supports condition subsets plus append-only
progress JSONL so larger CPU receiver runs are auditable instead of silent.

## Harness Change

The receiver script now supports:

- `--conditions`, for cheap discriminative receiver subsets before full
  source-destroying controls.
- `--progress-jsonl`, for append-only progress after each configurable number
  of examples.
- subset-aware summaries that preserve exact-ID parity over the active
  conditions.

This was necessary because long CPU runs previously wrote only at completion.

## Smoke Evidence

Both frozen `n=16` receiver subsets pass with target-only and shuffled-packet
controls:

| Surface | Conditions | Matched | Target-only | Shuffled | Valid matched | p50 matched ms | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| core seed29 | target, matched, shuffled | 0.688 | 0.250 | 0.250 | 1.000 | 4190.55 | true |
| holdout seed30 | target, matched, shuffled | 0.750 | 0.250 | 0.250 | 1.000 | 4059.46 | true |

Exact commands:

```bash
/opt/homebrew/bin/timeout 240s env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n16_subset_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 16 --seed 29 \
  --max-new-tokens 24 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet \
  --progress-jsonl .debug/source_private_target_decoder_n256_20260429/core_seed29_qwen3_n16_subset_progress.jsonl \
  --progress-every 1
```

```bash
/opt/homebrew/bin/timeout 240s env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n16_subset_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 16 --seed 30 \
  --max-new-tokens 24 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet \
  --progress-jsonl .debug/source_private_target_decoder_n256_20260429/holdout_seed30_qwen3_n16_subset_progress.jsonl \
  --progress-every 1
```

## Interpretation

This is not yet a main receiver-scale result. It is a positive smoke that shows
the frozen target model reads the 2-byte packet contract beyond target prior and
shuffled-source controls on both core and holdout surfaces. The result supports
the claim that the receiver need not be purely hand-coded, but the paper still
needs a larger full-control receiver row before treating this as a primary
defense.

## Next Gate

Run the same progress-enabled harness at `n=64` or `n=160` with all six
conditions. The pass rule should require matched packet accuracy above target
and every source-destroying/text control within `target_only + 0.05`, plus valid
prediction rate near `1.0`.
