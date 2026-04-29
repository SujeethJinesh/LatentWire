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

## All-Control Evidence

The all-control receiver gate now passes at `n=32` on both frozen surfaces when
the model is allowed enough tokens to emit complete candidate labels:

| Surface | Conditions | Matched | Target-only | Best control | Valid matched | p50 matched ms | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| core seed29 | all six | 0.688 | 0.250 | 0.250 | 1.000 | 2117.03 | true |
| holdout seed30 | all six | 0.750 | 0.250 | 0.281 | 1.000 | 2123.86 | true |

The n16 all-control rerun also passes on core (`0.688` matched versus `0.250`
target/best control). Holdout n16 is strongly positive (`0.750` matched), but
it narrowly misses the strict control threshold because one random same-byte row
lands on an extra correct candidate (`0.312` best control versus cutoff
`0.300`). The n32 row resolves this small-slice variance while preserving the
same matched accuracy.

Exact n32 commands:

```bash
/opt/homebrew/bin/timeout 1500s env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_progress_gate_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24 \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 32 --seed 29 \
  --max-new-tokens 24 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte \
  --progress-jsonl .debug/source_private_target_decoder_progress_20260429/core_seed29_qwen3_n32_all_controls_cpu_max24_progress.jsonl \
  --progress-every 1
```

```bash
/opt/homebrew/bin/timeout 1500s env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_progress_gate_20260429/holdout_seed30_qwen3_n32_all_controls_cpu_max24 \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 32 --seed 30 \
  --max-new-tokens 24 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte \
  --progress-jsonl .debug/source_private_target_decoder_progress_20260429/holdout_seed30_qwen3_n32_all_controls_cpu_max24_progress.jsonl \
  --progress-every 1
```

One failed diagnostic row is intentionally kept: with `max_new_tokens=8`, the
model emits only a shared candidate-label prefix and valid prediction rate is
`0.000`. The valid receiver rows therefore use `max_new_tokens=24`, which emits
complete 13-token labels. A direct MPS probe on this Mac still fails before
prediction with the known Apple MPS matmul shape error, so these receiver rows
are CPU-only.

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

This is now more than a three-condition smoke. The frozen target model reads
the 2-byte packet contract beyond target prior, shuffled-source, random
same-byte, and matched-byte structured-text controls on both core and holdout
surfaces at `n=32`. It supports the claim that the receiver need not be purely
hand-coded. It is still not a final receiver-scale result: the paper should keep
it as a strong reviewer-defense row until `n=160` or `n=256` all-control rows
are available.

## Next Gate

Run the same progress-enabled harness at `n=160` with all six conditions, or
switch to an endpoint TTFT/E2E frontier if serving telemetry becomes the more
urgent systems gap. The pass rule should require matched packet accuracy above
target and every source-destroying/text control within `target_only + 0.05`,
plus valid prediction rate near `1.0`.
