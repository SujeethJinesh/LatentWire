# Competitor Benchmark Bootstrap Plan

This note is for LatentWire benchmarking only. It is meant to make competitor comparisons reproducible, budget-matched, and paper-safe.

## Primary Source Anchors

- Cache-to-Cache (C2C): https://arxiv.org/abs/2510.03215 and official code: https://github.com/thu-nics/C2C
- KVComm: https://arxiv.org/abs/2510.03346 and official code: https://github.com/HankYe/KVCOMM
- KVPress: https://github.com/NVIDIA/kvpress
- KVzip: https://arxiv.org/abs/2505.23416, project page: https://janghyun1230.github.io/kvzip/, code: https://github.com/snu-mllab/KVzip
- Quest: https://arxiv.org/abs/2406.10774 and official code: https://github.com/mit-han-lab/quest
- H2O: https://arxiv.org/abs/2306.14048 and official code: https://github.com/FMInference/H2O
- SnapKV: https://arxiv.org/abs/2404.14469
- LM Evaluation Harness: https://github.com/EleutherAI/lm-evaluation-harness and docs: https://lm-evaluation-harness.readthedocs.io/

## Local Clone Snapshot

The cloned competitor repositories are kept under ignored `references/repos/`
so the paper can audit exact upstream code without committing vendor trees.

| Repo | Local path | Snapshot |
|---|---|---:|
| `C2C` | `references/repos/C2C` | `113c3a9` |
| `KVComm` | `references/repos/KVComm` | `1fa086e` |
| `KVzip` | `references/repos/KVzip` | `5d84729` |
| `Quest` | `references/repos/Quest` | `01c1623` |
| `H2O` | `references/repos/H2O` | `ac75c2a` |
| `SnapKV` | `references/repos/SnapKV` | `e216ddc` |
| `KVPress` | `references/repos/kvpress` | `2173234` |
| `AWQ` | `references/repos/llm-awq` | `d6e797a` |
| `EXL2 / ExLlamaV2` | `references/repos/exllamav2` | `7dc12af` |

## What To Compare

Use one benchmark harness for all methods, then vary only the compression or communication mechanism.

1. C2C-style semantic communication.
2. KVComm-style selective KV sharing.
3. KVPress/KVzip/Quest/H2O/SnapKV-style KV reduction or retrieval.
4. lm-eval harness reasoning tasks for end-to-end accuracy under a fixed prompt template.

Prefer the same downstream tasks across all methods:

- GSM-style math reasoning.
- SVAMP-style arithmetic reasoning.
- One or two general reasoning tasks from lm-eval harness to catch regressions in non-math settings.

## Matched-Budget Rules

Report every competitor under the same budget family, not just the same task.

- Prompt budget: identical system prompt, user prompt, and output schema.
- Token budget: same max input tokens and same max generated tokens.
- Byte budget: count serialized prompt bytes plus any communicated latent/cache payload bytes.
- Latency budget: same hardware, same batch size, same decoding settings, report p50 and p95 wall clock.
- Repair budget: if the method adds verifier or repair passes, count them explicitly as separate compute.

If a method changes the prompt structure, the generated tokens, or the repair loop, it must be reported as a budget increase, not as a free win.

## Exact Bootstrap Steps

1. Freeze the competitor commit or package version and record it in the artifact header.
2. Fix one canonical prompt manifest per task and never rewrite it during the comparison sweep.
3. Run a plain text baseline first.
4. Run the competitor method at the same task, model, seed, and decoding settings.
5. Run the LatentWire variant under the same prompt and token limits.
6. Run the matched-budget variant where the competitor is allowed the same token or latency envelope as LatentWire.
7. Save full telemetry for every sample, not just aggregate accuracy.
8. Bootstrap confidence intervals over seeds and over samples.
9. Produce one summary table that includes raw score, budget-normalized score, and repair/communication cost.
10. Keep a separate oracle table for analysis only; do not use oracle numbers for the main claim.

## Artifact Naming Recommendations

Use a stable naming scheme that makes later audits easy.

- Route pool telemetry: `results/<benchmark>_<split>_<date>/<model>_<method>_<budgettag>_salt<seed>_telemetry.jsonl`
- Telemetry metadata: same stem with `.meta.json`
- Summary markdown: same stem with `.md`
- Bootstrap summary: `..._<method>_bootstrap_summary.md`
- Prompt manifest: `..._prompt_manifest.json`
- Verification traces: `..._verifier_trace.jsonl`

Recommended fields in `<budgettag>`:

- `tok` for token budget
- `byte` for serialized byte budget
- `lat` for latency budget
- `repair` for repair-loop budget
- `kv` for KV-sharing or cache-transfer budget

## Telemetry Schema

Keep the schema flat and audit-friendly.

- `run_id`
- `benchmark`
- `split`
- `task`
- `model`
- `method`
- `seed`
- `prompt_hash`
- `prompt_tokens`
- `output_tokens`
- `raw_bytes`
- `compressed_bytes`
- `kv_bytes`
- `latency_ms`
- `repair_rounds`
- `verifier_calls`
- `selected_route`
- `target_route`
- `oracle_route`
- `pred_answer`
- `gold_answer`
- `correct`
- `help`
- `harm`
- `notes`

For cache-sharing methods, also log:

- `shared_kv_fraction`
- `loaded_kv_bytes`
- `retrieval_pages`
- `page_evictions`

For repair methods, also log:

- `pre_repair_answer`
- `post_repair_answer`
- `repair_changed`
- `repair_help`
- `repair_harm`
- `repair_budget_tokens`

## Claim Risks

These are the main paper-risk failure modes.

- A repair method that uses extra target-side generation is not automatically a communication-efficiency gain.
- A KV compression method that improves throughput on one task can still lose accuracy on reasoning tasks.
- A route-selection gain can disappear once prompts, seeds, or budget accounting change.
- A method can look better at fixed prompt length but worse at fixed wall-clock latency.
- An oracle comparison is useful for analysis but should not be used as the headline result.
- Cross-method comparisons are invalid if decoding settings, chat templates, or task-specific prompt formats differ.
- If the method changes the number of verifier passes, the main table must show that cost explicitly.

## Practical Interpretation Guidance

Use this paper-safe framing:

- "X improves accuracy under budget Y."
- "X reduces bytes or KV transfer at comparable accuracy."
- "X remains positive after matched token or latency controls."
- "X is a promising communication primitive, but the cost of repair must be reported."

Avoid this framing unless matched-budget evidence exists:

- "X is more efficient" without a cost table.
- "X beats the competitor" without noting extra repair or verifier compute.
- "X is better than C2C/KVComm" unless the prompt, token, and latency budgets are aligned.

## Notes For LatentWire

The most useful next comparison family is:

1. `C2C` versus LatentWire route selection and repair.
2. `KVComm` versus LatentWire cache-sharing or route-sharing variants.
3. `KVPress`, `KVzip`, `Quest`, `H2O`, and `SnapKV` as long-context efficiency baselines.
4. `lm-eval-harness` tasks as a sanity check that the method does not only help one arithmetic benchmark.

If a future result depends on repair, the paper should explicitly state whether the gain comes from selection, repair, or both.
