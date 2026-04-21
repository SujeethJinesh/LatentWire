# 378 Latest Competitor Benchmark Contract References

Date: 2026-04-21

Scope: recent primary sources plus local harness-doc readouts for fair matched benchmarking of LatentWire against direct communication peers. The main conclusion is to use separate benchmark lanes, not one mixed leaderboard.

## Sources

- `C2C` (`Cache-to-Cache`, ICLR 2026 poster): [OpenReview](https://openreview.net/forum?id=LeatkxrBCi), [arXiv](https://arxiv.org/abs/2510.03215), [repo](https://github.com/thu-nics/C2C), [official evaluator](https://github.com/thu-nics/C2C/blob/main/script/evaluation/unified_evaluator.py), local clone [README](repos/C2C/README.md). The official evaluator covers `mmlu-redux`, `gpqa`, `gsm8k`, `ai2-arc`, `mmlu-pro`, and `LongBench` subsets including `qasper`, `hotpotqa`, `2wikimqa`, and `musique`.
- `KVComm` (`Enabling Efficient LLM Communication through Selective KV Sharing`, ICLR 2026 poster): [OpenReview](https://openreview.net/forum?id=F7rUng23nw), [arXiv](https://arxiv.org/abs/2510.03346), [repo](https://github.com/Zephyroam/KVComm), local clone [README](repos/KVComm/README.md) and [eval.py](repos/KVComm/eval.py). The repo exposes `baseline`, `skyline`, `KVComm`, `Natural Language Debate`, and `CIPHER`, on `hotpotqa`, `qasper`, `musique`, `multifieldqa_en`, `twowikimqa`, and `tmath`.
- `LatentMAS` (`Latent Collaboration in Multi-Agent Systems`, arXiv 2025): [arXiv](https://arxiv.org/abs/2511.20639), [repo](https://github.com/Gen-Verse/LatentMAS), local clone [README](repos/LatentMAS/README.md) and [run.py](repos/LatentMAS/run.py). The repo exposes `baseline`, `text_mas`, and `latent_mas` on `gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, and `medqa`.
- `Q-KVComm` (`Efficient Multi-Agent Communication Via Adaptive KV Cache Compression`, arXiv 2025): [arXiv](https://arxiv.org/abs/2512.17914), [DOI](https://doi.org/10.48550/arXiv.2512.17914). Inference from the current source pass: I did not find an official repo or runnable harness, so this is strongest today as a rate-distortion reference, not yet a mandatory reproduction baseline.
- `KVCOMM` (`Online Cross-context KV-cache Communication`, NeurIPS 2025 poster, distinct from `KVComm` above): [OpenReview](https://openreview.net/forum?id=yGOytgjurF), [repo](https://github.com/FastMAS/KVCOMM). This is a systems-style reuse baseline for TTFT and prefill savings on `MMLU`, `GSM8K`, and `HumanEval`, not a direct semantic cross-model communication baseline.
- `lm-evaluation-harness`: [README](https://github.com/EleutherAI/lm-evaluation-harness), [interface docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [decontamination docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md), local clone [interface](repos/lm-evaluation-harness/docs/interface.md) and [decontamination](repos/lm-evaluation-harness/docs/decontamination.md). Useful features for 2025-2026 benchmarking: `--log_samples`, `--output_path`, explicit seeds, `--apply_chat_template`, `think_end_token`, and documented contamination handling.
- `OpenCompass`: [README](https://github.com/open-compass/opencompass), [LLM-judge / CascadeEvaluator docs](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/llm_judge.md), [accelerator docs](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/accelerator_intro.md), local clone [llm_judge.md](repos/opencompass/docs/en/advanced_guides/llm_judge.md). Useful when exact-match rules are insufficient, but judge model, judge prompt, and postprocessor must be frozen and reported.

## Why It Matters For LatentWire

- There is no honest all-peer task intersection across `C2C`, `KVComm`, and `LatentMAS`. A single blended table would either cherry-pick tasks or silently change parsers, prompts, and backends.
- The peers make different claims. `C2C` is the cleanest learned cross-model semantic-transfer baseline, `KVComm` is the cleanest selective KV-sharing baseline, `LatentMAS` is the strongest same-model latent-collaboration ceiling, `KVCOMM` is a reuse/TTFT baseline, and `Q-KVComm` is a compression reference. These should not be averaged into one headline number.
- The official repos already define the control ladder we should preserve: target-alone or `baseline`, text baseline (`text_mas` or natural-language exchange), repo-native upper bound (`skyline` or merged-input oracle), then the communication method. Dropping those controls makes improvements uninterpretable.
- Harness discipline matters because prompt drift, hidden CoT stripping, answer extraction, and contamination can move scores by more than the method delta. `lm-eval` now has explicit hooks for sample logging, seeds, chat templating, reasoning-trace stripping, and decontamination; `OpenCompass` now has explicit LLM-judge and cascade-evaluator configs that must be treated as part of the benchmark contract.
- Systems metrics need to be first-class. Recent peers report some combination of accuracy, token count, bytes, TTFT, answer latency, end-to-end latency, reuse rate, or compression ratio. LatentWire should always report transport cost next to accuracy.

## Concrete Bootstrap Steps

1. Freeze three benchmark lanes instead of one mixed table.
   `Reasoning / cross-model`: `GSM8K`, `GPQA`, `ARC-Challenge` for `LatentWire` vs `C2C` vs `LatentMAS`, plus target-alone and text baselines.
   `Context transfer`: `HotpotQA`, `Qasper`, `MuSiQue`, `2WikiMQA` for `LatentWire` vs `C2C` LongBench settings vs `KVComm`, plus skyline and text baselines.
   `Systems / rate-distortion`: `bytes_transmitted`, `input_tokens`, `output_tokens`, `TTFT`, `answer_latency_ms`, `e2e_latency_ms`, `reuse_rate`, and `compression_ratio`, with `KVCOMM` and `Q-KVComm` as the external references.
2. Keep competitor repos read-only and wrap them from LatentWire-side scripts into one schema.
   Minimum per-example fields: `example_id`, `dataset`, `split`, `source_model`, `target_model`, `backend`, `dtype`, `seed`, `prompt_hash`, `parser_version`, `prediction`, `gold`, `correct`, `input_tokens`, `output_tokens`, `bytes_transmitted`, `ttft_ms`, `answer_latency_ms`, `e2e_latency_ms`, `failure_tags`.
3. Enforce exact matched inference settings across rows.
   Same sender and receiver pair, same prompt topology, same `max_new_tokens`, same temperature and `top_p`, same chat template, same backend (`hf` vs `vllm`), same tokenizer revision, same seeds, and the exact same example ids.
4. Use repo-native lower and upper controls whenever available.
   Always include `target-alone` or `baseline`, text exchange (`text_mas`, NLD, or text-to-text), and `skyline` or merged-input oracle if the repo provides it.
5. Normalize reasoning-model outputs before scoring.
   If a model emits hidden reasoning traces, freeze `enable_thinking` and `think_end_token` across all rows, or strip those traces in one shared parser before evaluation.
6. Log samples before scaling.
   Use `lm-eval run ... --output_path ... --log_samples --seed ...` for smoke rows, and do not widen until 5-10 example runs from every method agree on prompt formatting and parser behavior.
7. Treat decontamination and judging as explicit experimental variables.
   If `lm-eval` decontamination is used, report both raw and clean numbers. If `OpenCompass` `GenericLLMEvaluator` or `CascadeEvaluator` is used, freeze and disclose the judge model, judge prompt, and postprocessor.
8. Do not publish a cross-lane average.
   Report one table per lane and one appendix schema manifest that proves prompt, parser, backend, and seed parity.
