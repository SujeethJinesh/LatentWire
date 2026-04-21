# Matched Competitor Bootstrap for Recent LLM Communication Methods

## Sources

- C2C: [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
- KVComm: [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://arxiv.org/abs/2510.03346)
- LatentMAS: [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)
- Q-KVComm: [Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression](https://arxiv.org/abs/2512.17914)
- LM Evaluation Harness repo: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- LM Eval decontamination docs: [docs/decontamination.md](/Users/sujeethjinesh/Desktop/LatentWire/references/repos/lm-evaluation-harness/docs/decontamination.md)
- LM Eval task and prompt docs: [docs/task_guide.md](/Users/sujeethjinesh/Desktop/LatentWire/references/repos/lm-evaluation-harness/docs/task_guide.md), [docs/interface.md](/Users/sujeethjinesh/Desktop/LatentWire/references/repos/lm-evaluation-harness/docs/interface.md), [docs/prompt overview](https://opencompass.readthedocs.io/en/latest/prompt/overview.html)
- OpenCompass repo: [open-compass/opencompass](https://github.com/open-compass/opencompass)
- OpenCompass dataset config docs: [Configure Datasets](https://opencompass.readthedocs.io/en/latest/user_guides/datasets.html)
- OpenCompass contamination eval: [Data Contamination Assessment](https://opencompass.readthedocs.io/en/stable/advanced_guides/contamination_eval.html)
- OpenCompass long-context/tokenizer parity: [RULER README](/Users/sujeethjinesh/Desktop/LatentWire/references/repos/opencompass/opencompass/configs/datasets/ruler/README.md)

## Why it matters for LatentWire

- Recent communication papers are easy to overclaim against if we do not hold the evaluation stack fixed. Small changes in prompt template, tokenizer choice, answer stripping, or decoding budget can move scores as much as the method itself.
- `lm-eval-harness` and OpenCompass expose different prompt/config abstractions. OpenCompass explicitly separates data-side and model-side templates and supports versioned dataset configs, which means a “same benchmark” row can still differ unless the prompt version, tokenizer, and chat template are pinned.
- Contamination handling is not optional for paper-facing claims. `lm-eval-harness` documents 13-gram decontamination, and OpenCompass has a dedicated contamination workflow. If LatentWire compares against recent LLM communication methods on public benchmarks, contaminated examples must be excluded or reported separately.
- Parser behavior is a common hidden failure mode. Communication methods often emit latent traces, chain-of-thought, or chat wrappers that can break exact-match extraction even when the answer is semantically correct. We need one answer postprocessor per benchmark family and a logged parser-failure bucket.
- LatentMAS/C2C-style methods are especially sensitive to matched pair assumptions. A direct peer comparison only makes sense if source model, target model, budget, template, and answer extraction are frozen across all rows.

## Concrete bootstrap steps

1. Freeze a matched benchmark contract before any new runs.
   - Pin `source_model`, `target_model`, decoding budget, `max_new_tokens`, chat template, `think_end_token` / CoT stripping, and answer postprocessor.
   - Keep same-model compression controls, direct communication peers, and LatentWire latent-bridge rows in separate tables.

2. Run prompt/parsing parity checks before scoring.
   - Use `lm-eval-harness` prompt/config inspection and OpenCompass prompt-viewer style checks to confirm the exact text each model sees.
   - Verify that the same evaluation slice produces the same extracted answer across harnesses before treating a delta as methodological.

3. Decontaminate first, then compare.
   - For LM Eval tasks, apply the documented 13-gram decontamination path where available.
   - For OpenCompass-style benchmark slices, run the contamination workflow or note when a benchmark has a built-in contaminated-clean split.
   - Exclude contaminated rows from method claims, or report them in a separate appendix table.

4. Bootstrap LatentMAS/C2C-style comparisons with a held-out pair protocol.
   - Use one fixed source/target model pair across LatentWire, C2C, and any LatentMAS-style latent-sharing proxy.
   - Hold prompt template, tokenizer, output budget, and scoring rule fixed.
   - Report latency, output tokens, retained-cache/route telemetry, and exact-match/aggregate accuracy together.

5. Keep matched controls separate from communication claims.
   - Same-model controls: KVPress, KVzip, Quest, H2O, StreamingLLM, SnapKV, PyramidKV, AdaKV.
   - Cross-model peers: C2C, KVComm, LatentMAS, Q-KVComm.
   - Do not mix these in a single leaderboard row without a note explaining why they are comparable.

6. Store parser failures and prompt diffs alongside results.
   - Save rejected generations, parse failures, and prompt snapshots per run.
   - Add a tiny run manifest that records benchmark name, prompt version, tokenizer, decontamination status, and competitor family.

7. Use the vendored harness clones only for smoke and parity.
   - `references/repos/lm-evaluation-harness`
   - `references/repos/opencompass`
   - Prefer the vendored repos for prompt inspection and config parity checks; keep paper claims tied to a single frozen run recipe.
