# Latest-Model Source-Packet Matrix

- generated: `2026-04-28`
- benchmark: `results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl`

## Recommendation

Treat MoE generalization as plausible but unproven. Run Qwen3.5-0.8B and Qwen3.5-2B first after upgrading Transformers to a qwen3_5-capable version; use Qwen3.6-35B-A3B and FP8 as off-machine MoE falsification rows.

## Model Matrix

| Priority | Model | Family | Params | Active | Status | Rung |
|---|---|---|---:|---:|---|---|
| P0 | `Qwen/Qwen3.5-0.8B` | Qwen3.5 small hybrid | 0.8B | - | blocked locally until Transformers supports qwen3_5 | compatibility smoke then n16 |
| P0 | `Qwen/Qwen3.5-2B` | Qwen3.5 small hybrid | 2B | - | planned after qwen3_5 compatibility update | n16 then n64 if cached |
| P1 | `Qwen/Qwen3.5-4B` | Qwen3.5 small hybrid | 4B | - | planned after qwen3_5 compatibility update | n16 if memory permits |
| P1 | `Qwen/Qwen3.6-35B-A3B` | Qwen3.6 MoE | 35B total | 3B activated | off-machine candidate | remote/API n32 then n500 if pass |
| P1 | `Qwen/Qwen3.6-35B-A3B-FP8` | Qwen3.6 MoE FP8 | 35B total | 3B activated | off-machine candidate | remote/API n32 then n500 if pass |
| P2 | `Qwen/Qwen3.6-27B` | Qwen3.6 dense | 27B | - | off-machine dense latest-model comparator | remote/API n32 |
| already-tested reference | `Qwen/Qwen3-0.6B` | Qwen3 small dense | 0.6B | - | positive reference row | n500 done in final evidence |
| P2 | `Qwen/Qwen3-1.7B` | Qwen3 small dense | 1.7B | - | optional small dense bridge row | n16 if cached/downloaded |
| P2 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek distilled Qwen | 1.5B | - | optional non-Qwen-org small source emitter | n16 if cached |

## Commands

### Qwen/Qwen3.5-0.8B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_0_8b --model Qwen/Qwen3.5-0.8B --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.5-2B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_2b --model Qwen/Qwen3.5-2B --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.5-4B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_4b --model Qwen/Qwen3.5-4B --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.6-35B-A3B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_6_35b_a3b --model Qwen/Qwen3.6-35B-A3B --device cuda --dtype bfloat16 --limit 32 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.6-35B-A3B-FP8

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_6_35b_a3b_fp8 --model Qwen/Qwen3.6-35B-A3B-FP8 --device cuda --dtype float16 --limit 32 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

## Compatibility Note

A local 2026-04-28 Qwen/Qwen3.5-0.8B smoke failed before generation with transformers 4.51.0 because AutoConfig did not recognize model_type qwen3_5. The model config in the local cache declares transformers_version 4.57.0.dev0, so this is a harness dependency blocker rather than source-packet failure.
