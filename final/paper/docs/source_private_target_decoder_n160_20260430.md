# Source-Private Target-Decoder N160 Gate

Date: 2026-04-30

## Cycle Start

1. Current ICLR readiness and distance: materially improved but still not
   comfortably full-paper ready. The direct frozen target-decoder row now
   clears `n=160` on Mac CPU, reducing the hand-coded receiver objection; the
   remaining distance is held-out/seed repetition and native systems telemetry.
2. Current paper story: source-private packets carry hidden evidence from a
   source to a target with public candidate side information. A frozen target
   LLM can consume the packet beyond target priors and same-byte/source-destroyed
   controls.
3. Exact blocker before this gate: direct Qwen target-decoder evidence was only
   `n=64`; reviewers could still treat model-mediated decoding as a smoke row.
4. Current live branch: product-codebook/semantic-anchor packets plus direct
   model-mediated receiver hardening.
5. Highest-priority gate: `Qwen/Qwen3-0.6B` target-decoder `n=160` all-control
   core surface on local CPU.
6. Scale-up rung: medium confirmation on Mac-only hardware.

## Command

```bash
env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl \
    results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl \
  --output-dir \
    results/source_private_tool_trace_target_decoder_n160_20260430/core_seed29_qwen3_n160_all_controls_cpu \
  --model Qwen/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --limit 160 \
  --seed 29 \
  --max-new-tokens 24 \
  --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet random_same_byte \
    structured_json_2byte structured_free_text_2byte \
  --progress-jsonl \
    .debug/source_private_target_decoder_n160_20260430/core_seed29_qwen3_n160_all_controls_cpu_progress.jsonl \
  --progress-every 8
```

Paired uncertainty:

```bash
./venv_arm64/bin/python \
  scripts/summarize_source_private_target_decoder_uncertainty.py \
  --result-dirs \
    results/source_private_tool_trace_target_decoder_n160_20260430/core_seed29_qwen3_n160_all_controls_cpu \
  --output-dir \
    results/source_private_tool_trace_target_decoder_n160_20260430/paired_uncertainty_core \
  --bootstrap-samples 5000 \
  --seed 20260430
```

## Result

Artifacts:

- `results/source_private_tool_trace_target_decoder_n160_20260430/core_seed29_qwen3_n160_all_controls_cpu/`
- `results/source_private_tool_trace_target_decoder_n160_20260430/paired_uncertainty_core/`

Point gate:

- pass gate: `True`
- examples: `160`
- exact ID parity: `True`
- matched packet: `111/160 = 0.694`
- target-only: `40/160 = 0.250`
- shuffled packet: `40/160 = 0.250`
- random same-byte: `40/160 = 0.250`
- structured JSON 2-byte: `40/160 = 0.250`
- structured free-text 2-byte: `40/160 = 0.250`
- matched minus target: `+0.444`
- matched minus best control: `+0.444`
- matched valid prediction rate: `1.000`
- matched p50 latency: `2670.3 ms`

Paired uncertainty:

- pass gate: `True`
- min CI95 low vs target: `+0.369`
- min CI95 low vs best control: `+0.369`
- min valid prediction rate: `1.000`

## Interpretation

This is the strongest local direct target-decoder evidence so far. It does not
make the method a fast model-decoder system: the CPU Qwen target call costs
about `2.7 s` p50 per condition. It does show that a frozen target LLM can use
the compact packet as information, while same-byte structured text and
source-destroying controls remain at the target floor.

The row therefore narrows the hand-coded-decoder critique. The deterministic
decoder remains the cleanest method for systems rate-frontier claims, but the
paper no longer relies only on a hand-coded target lookup for packet
consumption evidence.

## What Passed Expectations

- The matched packet row stayed strongly positive at `n=160`, not just `n=64`.
- All controls stayed exactly at target-only accuracy.
- Valid prediction rate stayed `1.000`.
- Paired bootstrap lower bounds cleared the reviewer-facing `+0.10` bar by a
  wide margin.

## What Still Needs Work

- Held-out `n=160` direct target-decoder replication is still missing.
- Product-codebook-specific model-mediated decoding remains future work; this
  row tests protocol packets, not the PQ receiver itself.
- Native serving systems metrics remain missing; this is receiver efficacy, not
  throughput.

## Decision

Promote direct Qwen target decoding from smoke to medium Mac-local supporting
evidence. The next exact gate is held-out `n=160` direct target decoding if Mac
time is available; otherwise the next method gate is a packet-consistency
denoiser over product-codebook/semantic-anchor packets.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py \
  tests/test_summarize_source_private_target_decoder_uncertainty.py
```

Outcome: `8 passed in 0.10s`.

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  scripts/summarize_source_private_target_decoder_uncertainty.py
```

Outcome: passed.
