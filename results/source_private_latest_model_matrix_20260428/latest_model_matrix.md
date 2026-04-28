# Latest-Model Source-Packet Matrix

- generated: `2026-04-28`
- benchmark: `results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl`

## Recommendation

Treat MoE generalization as plausible but unproven. Qwen3.5-0.8B now has CPU n160 seed-stable passes after upgrading Transformers to 5.7.0; Qwen3.5-2B now also passes CPU n160; Qwen3.5-4B passes CPU n16; Granite copied-helper has a non-Qwen n160 pass. Next run Qwen3.5-4B n64 only if local CPU time is acceptable, or use Qwen3.6-35B-A3B and FP8 as off-machine MoE falsification rows.

## Model Matrix

| Priority | Model | Family | Params | Active | Status | Rung |
|---|---|---|---:|---:|---|---|
| P0 | `Qwen/Qwen3.5-0.8B` | Qwen3.5 small hybrid | 0.8B | - | CPU n160 seed repeat passed after Transformers 5.7.0 upgrade; MPS backend fails before generation | CPU n160 passed on seeds 29/31 |
| P0 | `Qwen/Qwen3.5-2B` | Qwen3.5 small hybrid | 2B | - | CPU n16/n64/n160 passed after local download; MPS skipped due Qwen3.5 backend risk | CPU n16/n64/n160 passed |
| P1 | `Qwen/Qwen3.5-4B` | Qwen3.5 small hybrid | 4B | - | CPU n16 passed; local CPU latency is high, so widen only if worth the time cost | CPU n16 passed; n64 next if time permits |
| P2 | `Qwen/Qwen3.5-35B-A3B` | Qwen3.5 MoE | 35B total | 3B activated | off-machine Qwen3.5 MoE candidate | remote/API n32 after Qwen3.6 MoE |
| P2 | `Qwen/Qwen3.5-35B-A3B-FP8` | Qwen3.5 MoE FP8 | 35B total | 3B activated | off-machine Qwen3.5 MoE FP8 candidate | remote/API n32 after Qwen3.6 FP8 |
| P1 | `Qwen/Qwen3.6-35B-A3B` | Qwen3.6 MoE | 35B total | 3B activated | off-machine candidate | remote/API n32 then n500 if pass |
| P1 | `Qwen/Qwen3.6-35B-A3B-FP8` | Qwen3.6 MoE FP8 | 35B total | 3B activated | off-machine candidate | remote/API n32 then n500 if pass |
| P2 | `Qwen/Qwen3.6-27B` | Qwen3.6 dense | 27B | - | off-machine dense latest-model comparator | remote/API n32 |
| already-tested reference | `Qwen/Qwen3-0.6B` | Qwen3 small dense | 0.6B | - | positive reference row | n500 done in final evidence |
| P2 | `Qwen/Qwen3-1.7B` | Qwen3 small dense | 1.7B | - | optional small dense bridge row | n16 if cached/downloaded |
| P2 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | DeepSeek distilled Qwen | 1.5B | - | optional non-Qwen-org small source emitter | n16 if cached |
| P1 | `allenai/OLMo-2-0425-1B-Instruct` | OLMo 2 open instruct | 1B | - | n16 MPS failed behaviorally with 0 valid packets; pruned unless prompt contract changes | cross-family local n16 |
| P1 | `google/gemma-3-1b-it` | Gemma 3 small instruct | 1B | - | planned cross-family small-model falsification row | cross-family local n16 if access/cache permits |
| P1 | `ibm-granite/granite-3.3-2b-instruct` | Granite 3.3 instruct | 2B | - | CPU copied-helper n16/n64/n160 passed; trace-no-hint n64 passes weaker; MPS backend fails | CPU copied-helper n160 passed; trace-no-hint n64 weaker |
| P2 | `HuggingFaceTB/SmolLM3-3B` | SmolLM3 | 3B | - | planned cross-family small-model falsification row | cross-family local n16 if memory permits |
| P2 | `microsoft/Phi-4-mini-instruct` | Phi-4 mini instruct | 3.8B | - | planned Phi successor row | successor-family local n16 if memory permits |
| P2 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | Ministral 3 instruct | 3B-class | - | planned recent Mistral-family falsification row | cross-family local/off-machine n16 |
| P2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | Nemotron Nano | 9B | - | planned off-machine architecture-diversity row | off-machine architecture-diversity n16 |
| P2 | `moonshotai/Kimi-K2-Thinking` | Kimi K2 MoE | 1T total | 32B activated | planned off-machine non-Qwen MoE stress row | off-machine non-Qwen MoE stress row |

## Commands

### Qwen/Qwen3.5-0.8B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_0_8b --model Qwen/Qwen3.5-0.8B --device cpu --dtype float32 --limit 64 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.5-2B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_2b --model Qwen/Qwen3.5-2B --device cpu --dtype float32 --limit 160 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.5-4B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_5_4b --model Qwen/Qwen3.5-4B --device cpu --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.6-35B-A3B

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_6_35b_a3b --model Qwen/Qwen3.6-35B-A3B --device cuda --dtype bfloat16 --limit 32 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### Qwen/Qwen3.6-35B-A3B-FP8

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/qwen__qwen3_6_35b_a3b_fp8 --model Qwen/Qwen3.6-35B-A3B-FP8 --device cuda --dtype float16 --limit 32 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### allenai/OLMo-2-0425-1B-Instruct

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/allenai__olmo_2_0425_1b_instruct --model allenai/OLMo-2-0425-1B-Instruct --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### google/gemma-3-1b-it

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/google__gemma_3_1b_it --model google/gemma-3-1b-it --device mps --dtype float32 --limit 16 --seed 29 --max-new-tokens 8 --prompt-mode trace_no_hint --no-enable-thinking
```

### ibm-granite/granite-3.3-2b-instruct

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl --output-dir results/source_private_latest_model_matrix_20260428/ibm_granite__granite_3_3_2b_instruct --model ibm-granite/granite-3.3-2b-instruct --device cpu --dtype float32 --limit 64 --seed 29 --max-new-tokens 8 --prompt-mode copied_helper --no-enable-thinking
```

## Compatibility Note

A local 2026-04-28 Qwen/Qwen3.5-0.8B smoke first failed before generation with transformers 4.51.0 because AutoConfig did not recognize model_type qwen3_5. After upgrading the repo-local environment to transformers 5.7.0, tokenizers 0.22.2, and huggingface_hub 1.12.0, Qwen3.5-0.8B CPU n16, n64, and n160 source-packet rows passed, with n160 repeated on seeds 29 and 31. Qwen3.5-2B CPU n16, n64, and n160 rows also passed on seed 29. Qwen3.5-4B CPU n16 passed on seed 29 after downloading the 8.7G snapshot. The same 0.8B row still fails on Apple MPS before generation with an incompatible-dimensions matmul in the hybrid attention path, so MPS failure is logged as a backend compatibility issue rather than source-packet evidence. OLMo-2-0425-1B-Instruct is a behavioral negative at n16 with zero valid packets; Granite-3.3-2B-Instruct is a non-Qwen positive under copied-helper CPU n160 and a weaker trace-no-hint CPU n64 positive, while its MPS row is backend-blocked.
