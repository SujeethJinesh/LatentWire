# 381 Competitor Benchmark Bootstrap References

Date: 2026-04-21

Scope: recent primary sources plus harness docs for fair comparison against cross-model communication, latent communication, KV-cache communication, and connector-style baselines. The goal is not a generic eval survey. The goal is a concrete benchmark contract for LatentWire.

## Sources

### Recent competitor methods

| Source | Status | LatentWire read |
|---|---|---|
| [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215) / [repo](https://github.com/thu-nics/C2C) / [OpenReview](https://openreview.net/forum?id=LeatkxrBCi) | ICLR 2026 paper/repo | Cleanest direct KV-cache semantic transfer baseline. The official code already separates projector, evaluator, and native controls. The key comparison should be not just accuracy, but transport cost, latency, and whether the gain survives the same prompt contract. |
| [Enabling Efficient LLM Communication through Selective KV Sharing](https://arxiv.org/abs/2510.03346) / [repo](https://github.com/Zephyroam/KVComm) / [OpenReview](https://openreview.net/forum?id=F7rUng23nw) | ICLR 2026 poster/repo | Strong selective-cache baseline. Useful because it makes the selection problem explicit. If LatentWire beats it, we need matched per-example routing and identical budget accounting, not an averaged summary. |
| [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) / [repo](https://github.com/Gen-Verse/LatentMAS) | arXiv 2025 / repo | Strong same-model latent-collaboration ceiling. It is a useful upper comparison for shared-latent communication, but it should stay on a separate lane from cross-family communication because the task contract is different. |
| [Efficient Multi-Agent Communication Via Adaptive KV Cache Compression](https://arxiv.org/abs/2512.17914) | arXiv 2025 | Rate-distortion reference for communication under a budget. Even without an official repo found in this pass, it is useful for budgeting language: compression ratio, coherence quality, and task fidelity should be reported together. |
| [Online Cross-context KV-cache Communication](https://openreview.net/forum?id=yGOytgjurF) / [repo](https://github.com/FastMAS/KVCOMM) | NeurIPS 2025 poster/repo | Systems reuse baseline rather than semantic-transfer baseline. Useful for TTFT / prefill / reuse accounting and for distinguishing “reuse cache” from “understand cache.” |

### Benchmark harnesses and evaluation infrastructure

| Source | Status | LatentWire read |
|---|---|---|
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Active open-source harness | Best default for reproducible model-side evaluation. Relevant docs: [model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md), [task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md), [interface docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [decontamination docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md). |
| [OpenCompass](https://github.com/open-compass/opencompass) | Active open-source platform | Best default for judge-based or cascade-evaluator rows when exact-match parsing is insufficient. Relevant docs: [prompt template](https://opencompass.readthedocs.io/en/stable/prompt/prompt_template.html), [LLM judge](https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/llm_judge.md). |
| [CompassVerifier](https://github.com/open-compass/CompassVerifier) | 2025 verifier repo | Useful when we need a frozen, dedicated verifier instead of ad hoc judge prompts. Good reference for outcome-reward style matched evaluation rows. |
| [NVIDIA NeMo Evaluator SDK benchmark catalog](https://docs.nvidia.com/nemo/evaluator/0.1.0/evaluation/benchmarks.html) | Current docs | Good task catalog and benchmark taxonomy. Useful when we want to map our connector lanes onto standard reasoning, QA, and long-context suites. |
| [EvalScope supported benchmarks](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html) | Current docs | Useful for benchmark coverage and for seeing how the community packages AIME-2025, GPQA, MMLU-Pro, BrowseComp, HumanEval+, and similar suites. |

## Why These Sources Matter For LatentWire

- The 2025-2026 connector papers make the evaluation problem much sharper: a bridge can only be called good if the prompt contract, tokenizer contract, and budget contract are fixed. Otherwise the result is a backend artifact.
- C2C, KVComm, and LatentMAS should not be collapsed into one mixed leaderboard. They measure different things: direct semantic transfer, selective cache sharing, and same-model latent collaboration.
- The harness docs matter because the practical failure modes are boring but real: chat-template drift, answer extraction drift, hidden reasoning traces, contamination, and different backends can dominate the delta between methods.
- `lm-evaluation-harness` and `OpenCompass` are complementary. `lm-eval` is the cleanest path for exact-match and log-likelihood rows; `OpenCompass` is the cleaner path for judged or multi-step free-form rows. We should not mix those scoring policies inside one headline table.
- `CompassVerifier` is important because it gives a more defensible “frozen verifier” story than hand-tuned evaluation prompts. If we need judge-based rows later, the judge itself must become part of the contract.
- Recent papers and docs around KV reuse and cache communication reinforce a single rule: systems metrics must be reported next to semantic metrics. Accuracy alone is not enough for any cache-transfer claim.

## Concrete Fair-Comparison Contract

1. Freeze the exact prompt topology per lane.
   Use one prompt hash, one chat template, one system prompt, one answer extraction rule, one parser version, and one seed family for all methods in the same table.
2. Freeze the backend stack.
   Report the model backend, dtype, tokenizer revision, maximum context length, decoding settings, and whether `apply_chat_template` or an equivalent wrapper was used.
3. Freeze the example ids.
   Every row should be matched on the same sample ids, not merely the same dataset split.
4. Include the canonical controls in every lane.
   Always include `target-alone` or `baseline`, one text-exchange baseline, and one oracle or skyline row when the competitor repo supports it.
5. Report budget and transport together.
   Every row should include `input_tokens`, `output_tokens`, `bytes_transmitted`, `TTFT_ms`, `answer_latency_ms`, `e2e_latency_ms`, and `reuse_rate` or cache-hit ratio if applicable.
6. Keep raw and cleaned scores separate.
   If decontamination is used, report raw and clean numbers side by side. If a judge is used, freeze the judge model, judge prompt, and postprocessor and list them in the caption or appendix.
7. Do not average across incompatible lanes.
   Separate cross-model semantic transfer, context transfer, and systems reuse. A blended average hides the actual failure mode.
8. Treat parser versioning as part of the benchmark.
   Any change to extraction, normalization, or hidden reasoning stripping should increment the parser version and be logged in the table metadata.

## Matched Evaluation Rows To Add Later

Recommended row schema for every future benchmark table:

- `example_id`
- `dataset`
- `split`
- `lane` (`semantic_transfer`, `context_transfer`, `systems_reuse`)
- `source_model`
- `target_model`
- `family_pair` or `same_family`
- `communication_mode` (`target-alone`, `text_exchange`, `latent_bridge`, `cache_share`, `oracle`, `skyline`)
- `connector_variant`
- `tokenizer_relation` (`same`, `cross`, `byte_fallback`)
- `route_policy`
- `repair_policy`
- `budget_tokens`
- `budget_bytes`
- `seed`
- `prompt_hash`
- `parser_version`
- `prediction`
- `gold`
- `correct`
- `exact_match` or `judge_score`
- `input_tokens`
- `output_tokens`
- `bytes_transmitted`
- `ttft_ms`
- `answer_latency_ms`
- `e2e_latency_ms`
- `reuse_rate`
- `failure_tags`

Suggested matched rows to include once the method stabilizes:

- `target-alone`
- `text_exchange`
- `skyline` or `oracle`
- `c2c`
- `kvcomm`
- `latentmas`
- `q-kvcomm`
- `kvcomm_systems_reuse`
- `ours_raw`
- `ours_gauge_fix`
- `ours_gpa_sparse_dictionary`
- `ours_byte_interface`
- `ours_tokenizer_aligned`
- `ours_repair_on`
- `ours_repair_off`

## Benchmark Lane Structure I Would Use

### Lane A: Semantic transfer

Use `GSM8K`, `GPQA`, and `ARC-Challenge` style rows for direct cross-model communication. This lane should answer: does the connector improve correctness under the same prompt and budget?

### Lane B: Context transfer

Use `HotpotQA`, `Qasper`, `MuSiQue`, and `2WikiMQA` style rows. This lane should answer: does the connector preserve useful context better than text exchange or cache reuse?

### Lane C: Systems / rate-distortion

Always report `bytes_transmitted`, `TTFT_ms`, `answer_latency_ms`, `e2e_latency_ms`, and reuse rate. This lane should answer: how much semantic gain do we buy per byte or per millisecond?

### Lane D: Parser / judge robustness

Only if needed, add judge-based or verifier-based rows under a frozen `OpenCompass` or `CompassVerifier` contract. This lane should answer: are improvements robust to scoring policy, or are they extraction artifacts?

## Practical Recommendation

- Keep `lm-evaluation-harness` as the default backbone for exact-match and log-probability rows.
- Keep `OpenCompass` or `CompassVerifier` for judge-based or cascade-style rows only.
- Keep `C2C`, `KVComm`, and `LatentMAS` in separate comparison blocks with matched controls, not in one headline average.
- Treat any future improvement as real only if it survives the same prompt hash, same example ids, same parser version, and the same budget accounting.
