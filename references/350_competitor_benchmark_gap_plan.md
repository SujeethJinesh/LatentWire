# Competitor Benchmark Gap Plan

Date: 2026-04-21

Scope: benchmark gaps for direct cross-model communication, same-model KV/cache controls, and prompt-compression controls. This note is intentionally conservative about fairness: do not mix task families, tokenizer assumptions, or latency/accounting rules.

## Normalization Rules

- Direct communication peers must use the same source/target pair, prompt template, answer parser, and decode budget.
- Same-model cache controls must use the same target model, prompt family, context length, and output-token budget.
- Prompt compression controls must compare against the same target-only prompt budget, not against cross-model transport claims.
- Always report bytes moved, latency, tokens/sec, and task accuracy together.

| Competitor | Repo / paper link | What it measures | Fair LatentWire comparator | Required normalization | Next runnable command / artifact |
| --- | --- | --- | --- | --- | --- |
| C2C | [GitHub](https://github.com/thu-nics/C2C) / [paper](https://arxiv.org/abs/2510.03215) | Direct semantic transfer through KV-cache projection and fusion. | LatentWire bridge on the same heterogeneous source/target pair. | Same pair, same prompt template, same output budget, same parser, same byte/latency accounting. | `./venv_arm64/bin/python scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --max-new-tokens 64 --limit 5 --prediction-output results/competitor_gap_20260421/c2c_gsm70.jsonl` |
| KVComm | [GitHub](https://github.com/Zephyroam/KVComm) / [paper](https://arxiv.org/abs/2510.03346) | Selective KV sharing with layer-importance calibration. | LatentWire bridge with the same source/target pair and the same layer budget. | Same calibration split, same top-layer fraction, same parser, same transfer-budget accounting. | `./venv_arm64/bin/python scripts/run_kvcomm_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --calibration-file data/gsm8k_100.jsonl --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --top-layers-grid 0.25,0.5,0.75,1.0 --prediction-output results/competitor_gap_20260421/kvcomm_gsm70.jsonl` |
| LatentMAS | [GitHub](https://github.com/Gen-Verse/LatentMAS) / [paper](https://arxiv.org/abs/2511.20639) | Latent-space multi-agent collaboration, token savings, and wall-clock reduction. | LatentWire latent-collaboration control, not a direct cache-fusion peer. | Same task family, same agent count, same max output tokens, same stop rules, same tokenizer. | `cd references/repos/LatentMAS && ../../venv_arm64/bin/python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples 5 --max_new_tokens 256` |
| KVPress | [GitHub](https://github.com/NVIDIA/kvpress) / [paper](https://arxiv.org/abs/2510.00636) | Policy-level KV compression and framework portability. | LatentWire target-model-only compression control. | Same target model, same compression ratio, same prompt length, same eval split, same decode budget. | `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_gate_search_30.jsonl --device mps --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --prediction-output results/competitor_gap_20260421/kvpress_expected_attention_gsm30.jsonl` |
| KVzip | [GitHub](https://github.com/snu-mllab/KVzip) / [paper](https://arxiv.org/abs/2505.23416) | Query-agnostic KV eviction plus context reconstruction quality. | LatentWire target-model-only compression under a fixed byte budget. | Same model family, same retrieval/QA family, same compression ratio, same reconstruction path, same output budget. | `cd references/repos/KVzip && ../../venv_arm64/bin/python eval.py -m qwen2.5-7b -d squad --kv_type retain --num 100` |
| Quest | [GitHub](https://github.com/mit-han-lab/Quest) / [paper](https://arxiv.org/abs/2406.10774) | Query-aware page loading / sparse KV access and throughput. | LatentWire same-family long-context control. | Same supported model family, same context length, same page budget, same benchmark family, same CUDA/FlashAttention assumptions. | `cd references/repos/Quest && bash scripts/passkey.sh` |
| H2O | [GitHub](https://github.com/FMInference/H2O) / [paper](https://arxiv.org/abs/2306.14048) | Heavy-hitter plus recency eviction tradeoffs. | LatentWire same-model long-context eviction control. | Same prompt length, same output length, same heavy/recent ratios, same GPU precision, same metric. | `cd references/repos/H2O/h2o_hf && ../../venv_arm64/bin/python run_summarization.py --input_path data/summarization_data/xsum_0shot.jsonl --output_path results/competitor_gap_20260421/h2o_xsum_0shot.jsonl --model_name lmsys/vicuna-13b-v1.3 --enable_h2o_cache` |
| SnapKV | [GitHub](https://github.com/FasterDecoding/SnapKV) / [paper](https://arxiv.org/abs/2404.14469) | Prompt-observation-based KV position selection. | LatentWire target-model prompt-side selection control. | Same observation window, same compression ratio, same context family, same benchmark split. | `cd references/repos/SnapKV/experiments/LongBench && bash scripts/run_longbench.py` |
| LLMLingua / LongLLMLingua | [GitHub](https://github.com/microsoft/LLMLingua) / [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/) / [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) | Prompt compression and token-budget preservation. | LatentWire target-only prompt compression or truncation control. | Same target model, same prompt budget, same instruction/question split, same answer budget, same tokenizer. | `cd references/repos/LLMLingua && ../../venv_arm64/bin/python -c "from llmlingua import PromptCompressor; print(PromptCompressor().compress_prompt('Question: 1+1?', instruction='', question='', target_token=20)['ratio'])"` |

## Adjacent Cross-Model Baselines To Watch

- DroidSpeak: [paper](https://arxiv.org/abs/2411.02820). No safe public repo clone was verified in this pass.
- Interlat: [paper](https://arxiv.org/abs/2511.09149). No safe public repo clone was verified in this pass.

## Top Benchmark Gaps

- Direct communication baselines still need one strict table with the exact same source/target pair, parser, and decode budget.
- Same-model cache controls still need a unified long-context table with bytes, latency, and accuracy reported together.
- Prompt compression is still missing a direct LatentWire control that fixes the target prompt budget and answer budget.
- The paper-only latent-communication watchlist is still incomplete until a safe public repo or a clean bootstrap path appears.
