# Source-Private Tool-Trace Artifact Release Manifest

- date: `2026-04-28`
- gate: `source_private_tool_trace_artifact_release_choice_20260428`
- status: release archive built and integrity-checked
- archive: `paper/artifacts/source_private_tool_trace_artifacts_20260428.zip`
- archive bytes: `7548312`
- archive SHA256: `64153e44dd5b41a30e54ffa5cdb0d95ca5498c2345c13da015a9f9f076c0121f`
- archive file count: `265`
- uncompressed bytes: `113013097`

## Contents

The archive is rooted at `source_private_tool_trace_artifacts_20260428/` and
contains:

- `README.md` with artifact map, headline metrics, and reproduction commands.
- `paper/iclr2026/source_private_tool_trace.pdf`.
- `paper/iclr2026/source_private_tool_trace_iclr_source_20260428.zip`, the
  compile-tested manuscript source bundle.
- paper readout memos for reviewer-risk rows, target-decoder smoke, final
  review, submission decision, and human PDF read.
- decisive raw JSON/JSONL result roots:
  - `results/source_private_tool_trace_reviewer_risk_rows_20260429/`
  - `results/source_private_tool_trace_target_decoder_smoke_20260429/`
  - `results/source_private_tool_trace_baseline_pack_20260429/`
  - `results/source_private_tool_trace_latex_or_figures_20260430/`
  - `results/source_private_hidden_repair_packet_medium_20260429/`
  - `results/source_private_hidden_repair_packet_holdout_families_20260429/`
  - `results/source_private_hidden_repair_packet_seed_repeat_20260429/`
  - `results/source_private_hidden_repair_packet_medium_llm_20260429/`
  - `results/source_private_hidden_repair_packet_holdout_families_llm_20260429/`
- target-decoder input surfaces copied from scratch space under
  `debug_inputs/source_private_tool_trace_target_decoder_smoke_20260429/`.

## Verification

Archive integrity:

```bash
unzip -t paper/artifacts/source_private_tool_trace_artifacts_20260428.zip
```

Result: all files tested `OK`; no compressed-data errors detected.

Local-path hygiene scan: the release tree, archive, and baseline-pack readout
were scanned for absolute user/home path markers. Result: no matches.

## Reproduction Commands

Deterministic packet surfaces:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_gate.py \
  --n 500 --seed 29 --out-dir results/source_private_hidden_repair_packet_medium_20260429

./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_gate.py \
  --n 500 --seed 30 --family-mode holdout \
  --out-dir results/source_private_hidden_repair_packet_holdout_families_20260429
```

Model-packet surfaces require locally cached source models because the recorded
runs used `local_files_only=True`:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_model_gate.py \
  --benchmark results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --model Qwen/Qwen3-0.6B --out-dir results/source_private_hidden_repair_packet_medium_llm_20260429/qwen3_trace_no_hint

./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_model_gate.py \
  --benchmark results/source_private_hidden_repair_packet_holdout_families_20260429/benchmark.jsonl \
  --model Qwen/Qwen3-0.6B --out-dir results/source_private_hidden_repair_packet_holdout_families_llm_20260429/qwen3_trace_no_hint
```

Reviewer-risk rows:

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_reviewer_risk_rows.py \
  --out-dir results/source_private_tool_trace_reviewer_risk_rows_20260429
```

Target-decoder smoke:

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark .debug/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_benchmark/benchmark.jsonl \
  --out-dir results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n16
```

## Interpretation

This package is for artifact review and result audit. It preserves the raw
decisive JSON/JSONL rows behind the manuscript tables and controls. It does not
expand the scientific claim beyond the scoped source-private diagnostic-packet
protocol stated in the paper.
