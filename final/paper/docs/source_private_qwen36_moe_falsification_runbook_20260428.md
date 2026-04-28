# Qwen3.6 MoE Falsification Runbook

- date: `2026-04-28`
- status: ready-to-run off-machine gate
- benchmark: `results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl`

## Purpose

The current latest-model evidence covers Qwen3.5 small dense/hybrid rows and a
non-Qwen Granite row. MoE generalization remains unproven until sparse-MoE and
FP8 sparse-MoE source emitters pass the same source-destroying controls.

## CUDA Runner

Run these from the repository root on a CUDA host with the same repo and
`./venv_arm64` or equivalent repo-local venv.

```bash
HF_HOME=.hf_home ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_qwen36_moe_falsification_20260428/qwen36_35b_a3b_n32_seed29 \
  --model Qwen/Qwen3.6-35B-A3B \
  --device cuda \
  --dtype bfloat16 \
  --limit 32 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

```bash
HF_HOME=.hf_home ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_qwen36_moe_falsification_20260428/qwen36_35b_a3b_fp8_n32_seed29 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --device cuda \
  --dtype float16 \
  --limit 32 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

## Endpoint Wrapper Contract

If using a vLLM/OpenAI-compatible endpoint instead of local HF loading, use the
repo runner below. It replaces only source packet generation and reuses the same
benchmark rows, prompt construction, extraction regex, evaluator, control names,
and pass rule as the local HF runner.

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_endpoint.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_qwen36_moe_falsification_20260428/qwen36_35b_a3b_endpoint_n32_seed29 \
  --model Qwen/Qwen3.6-35B-A3B \
  --api-base http://127.0.0.1:8000/v1 \
  --limit 32 \
  --seed 29 \
  --max-tokens 8 \
  --prompt-mode trace_no_hint
```

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_endpoint.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_qwen36_moe_falsification_20260428/qwen36_35b_a3b_fp8_endpoint_n32_seed29 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --api-base http://127.0.0.1:8000/v1 \
  --limit 32 \
  --seed 29 \
  --max-tokens 8 \
  --prompt-mode trace_no_hint
```

Required packet row fields:

- `example_id`
- `generated_text`
- `packet`
- `packet_bytes`
- `packet_tokens`
- `latency_ms`
- `valid_packet`
- `prompt_mode`

After writing `model_packets.jsonl`, run the existing evaluator logic from
`scripts/run_source_private_hidden_repair_packet_llm.py` to create:

- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
- `endpoint_trace.jsonl`
- `endpoint_metadata.json`

`endpoint_trace.jsonl` must contain raw request JSON and raw response JSON for
each example. `endpoint_metadata.json` should include `/v1/models` output when
available plus the served model ID/revision, serving engine, launch command,
dtype, quantization mode, tensor-parallel size, and max model length. The
runner exposes these as CLI flags and `SERVED_MODEL_*` environment variables.

## Pass Rule

Each MoE row passes only if all hold:

- `summary.pass_gate == true`
- `summary.n == 32`
- `summary.exact_id_parity == true`
- matched model packet accuracy beats target-only by at least `0.15`
- best source-destroying control is at most target-only plus `0.02`
- packet valid rate is reported

With the current constructional floor, target-only is expected to be `8/32 =
0.250`; matched must reach at least `13/32 = 0.406`, and controls should remain
at or below about `0.270`.

## Required Artifacts

Save per row:

- `model_packets.jsonl`
- `predictions.jsonl`
- `summary.json`
- `summary.md`
- `manifest.json`
- `manifest.md`
- stdout/stderr log
- `endpoint_trace.jsonl` and `endpoint_metadata.json` for endpoint runs
- exact model revision or served model ID
- serving command or endpoint configuration
- benchmark SHA256
- runner or wrapper SHA256

## Safe Wording If Both n32 Rows Pass

Safe: "On an off-machine n32 falsification slice, the same source-private
packet protocol also works for Qwen3.6 sparse-MoE and FP8 sparse-MoE source
emitters under unchanged target-only and source-destroying controls."

Unsafe before n500 and seed repeats: "MoE generalization is proven",
"architecture-independent", or "robust across MoE deployments."
