# 384 Competitor Contract Latest References

Date: 2026-04-21

Scope: 2025-2026 competitor methods and evaluation contracts relevant to direct semantic communication, selective KV sharing, latent collaboration, KV-cache compression, and judge-based evaluation. This memo is meant to freeze the fair-comparison contract LatentWire should use later, not to define a blended leaderboard.

## Sources

| Source | Direct links | Why it matters |
|---|---|---|
| C2C, *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* | [OpenReview](https://openreview.net/forum?id=LeatkxrBCi), [arXiv](https://arxiv.org/abs/2510.03215), [repo](https://github.com/thu-nics/C2C), [official evaluator](https://github.com/thu-nics/C2C/blob/main/script/evaluation/unified_evaluator.py) | Cleanest learned cross-model semantic-transfer baseline. Its official evaluator already separates projector, evaluator, and native/oracle controls, which is the right comparison shape for LatentWire. |
| KVComm, *Enabling Efficient LLM Communication through Selective KV Sharing* | [OpenReview](https://openreview.net/forum?id=F7rUng23nw), [arXiv](https://arxiv.org/abs/2510.03346), [repo](https://github.com/Zephyroam/KVComm), [eval.py](https://github.com/Zephyroam/KVComm/blob/main/eval.py) | Strong selective-cache baseline. It makes selection, budget, and reuse explicit, so LatentWire should match its accounting rather than compare only end accuracy. |
| LatentMAS, *Latent Collaboration in Multi-Agent Systems* | [arXiv](https://arxiv.org/abs/2511.20639), [repo](https://github.com/Gen-Verse/LatentMAS) | Same-model latent-collaboration ceiling. Useful, but should stay in a separate lane from cross-family communication because the task contract is different. |
| Q-KVComm, *Efficient Multi-Agent Communication Via Adaptive KV Cache Compression* | [arXiv](https://arxiv.org/abs/2512.17914) | Rate-distortion reference for adaptive KV compression. Useful for compression language and budget reporting even if no official reproduction harness is available. |
| CompassVerifier, *A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward* | [arXiv](https://arxiv.org/abs/2508.03686), [repo](https://github.com/open-compass/CompassVerifier) | Best recent frozen-verifier reference if a judge-based row is needed. Good alternative to ad hoc judge prompting. |
| `lm-evaluation-harness` | [repo](https://github.com/EleutherAI/lm-evaluation-harness), [model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md), [task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md), [interface docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [decontamination docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md) | Default exact-match / log-probability backbone. Current docs explicitly support `--log_samples`, `--output_path`, decontamination, `apply_chat_template`, and `think_end_token` stripping for reasoning models. |
| OpenCompass | [site](https://www.opencompass.us/), [prompt template docs](https://doc.opencompass.org.cn/prompt/prompt_template.html), [LLM judge docs](https://opencompass.readthedocs.io/en/latest/advanced_guides/llm_judge.html) | Best judge-based and cascade-evaluator stack if exact-match parsing is not enough. Prompt templates and judge configuration must be frozen as part of the benchmark contract. |

## What These Sources Say About Fair Evaluation

- Do not blend cross-model semantic transfer, KV reuse, and same-model latent collaboration into one average.
- Keep the control ladder intact in every lane: target-alone or baseline, text-exchange baseline, oracle or skyline row when available, then the method under test.
- Treat prompt topology, parser version, chat template, tokenizer revision, backend, dtype, and seed as part of the benchmark contract.
- If hidden reasoning traces exist, freeze `think_end_token` or strip them with one shared parser across all rows.
- If a judge is used, freeze the judge model, judge prompt, postprocessor, and any cascade configuration.
- Report systems metrics next to semantic metrics. Accuracy alone is not sufficient for communication claims.

## Exact LatentWire Matched Rows To Use Later

These are the rows we should freeze when the method stabilizes:

- `target-alone`
- `text_exchange`
- `skyline` or `oracle`
- `c2c`
- `kvcomm`
- `latentmas`
- `q-kvcomm` if and only if a comparable runnable contract exists; otherwise keep it as reference-only
- `ours_raw`
- `ours_gauge_fix`
- `ours_gpa_sparse_dictionary`
- `ours_byte_interface`
- `ours_tokenizer_aligned`
- `ours_repair_off`
- `ours_repair_on`

## Exact Metrics To Report For Every Matched Row

- `example_id`
- `dataset`
- `split`
- `lane`
- `source_model`
- `target_model`
- `family_pair` or `same_family`
- `communication_mode`
- `connector_variant`
- `tokenizer_relation`
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

## Caveats LatentWire Should Enforce

- Use the exact same sample ids across methods inside a table.
- Keep one prompt topology, one chat template, one parser version, one backend, and one tokenizer revision per table.
- Report raw and cleaned scores side by side if decontamination or parser cleanup is used.
- Keep judge-based rows separate from exact-match rows.
- Do not compare systems-reuse baselines like `KVCOMM` to semantic-transfer baselines as if they were the same task.
- Do not publish a single cross-lane average; publish one table per lane and one appendix manifest that proves parity.
