# 373 Competitor Benchmark Bootstrap Recent References

Date: 2026-04-21

## Sources

- `lm-evaluation-harness` README and docs: https://github.com/EleutherAI/lm-evaluation-harness
- `lm-evaluation-harness` decontamination guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
- Local harness clone: `references/repos/lm-evaluation-harness`
- `OpenCompass` README and examples: https://github.com/open-compass/opencompass
- Local OpenCompass clone: `references/repos/opencompass`
- `LatentMAS` competitor repo for direct latent communication baselines: https://github.com/Gen-Verse/LatentMAS
- `Cache-to-Cache (C2C)` competitor repo for KV-cache communication baselines: https://github.com/thu-nics/C2C

Recent harness patterns worth copying:

- `lm-evaluation-harness` supports clean task config files, multiple backends, request caching, and explicit decontamination hooks.
- `OpenCompass` supports CLI and Python-script evaluation, multiple backends (`hf`, `vllm`, `lmdeploy`, API), OpenAI-style models, and broader leaderboard-style orchestration.

## Why It Matters For Us

- Cross-model communication papers are easy to overclaim if the benchmark path is not matched. We need harness parity before we compare methods.
- `lm-evaluation-harness` is the best default for reproducible smoke rows, parser validation, and exact-match / multiple-choice style tasks.
- `OpenCompass` is useful as a second orchestration shape when we want to catch backend-specific or prompt-template-specific mistakes before freezing tables.
- Direct-communication competitors like `LatentMAS` and `C2C` give us the right baseline family for cache/latent transport comparisons, but they should stay in a separate benchmark track from the LatentWire method story.
- The main pitfalls are contamination, prompt-template mismatch, backend drift, and different answer extraction rules. If any of those vary, the table is not fair even if the numbers look better.

## Concrete Bootstrap Steps

1. Clone and pin the harness repos under `references/repos/` and treat them as read-only vendor code.
2. Build thin LatentWire wrappers that emit JSONL predictions plus `.meta.json` sidecars with `model`, `backend`, `dtype`, `seed`, `tokens`, `latency`, and `parser_version`.
3. Run 5-10 example smoke rows first on the exact same cached example list for LatentWire, LatentMAS, and C2C-style baselines.
4. Use the same answer parser and normalization everywhere. If a repo ships its own parser, record that explicitly and do not mix it silently into the method table.
5. Enable decontamination or contamination logging for harness-driven runs when the dataset supports it, and record whether a row is clean or raw.
6. Keep matched tables separate from method-ablation tables until the harness is stable.
7. Only after the smoke rows are consistent, scale to larger matched sweeps and promote one benchmark row at a time into paper-facing tables.

Recommended comparison ladder:

- LatentWire target-alone control
- LatentMAS baseline
- LatentMAS text-MAS
- LatentMAS latent-MAS
- C2C projector baseline
- OpenCompass smoke row on the same dataset
- `lm-evaluation-harness` smoke row on the same dataset

Operational guardrails:

- Do not let prompt templates differ across rows unless the template change is itself the ablation.
- Do not compare rows with different max token budgets or decode temperatures.
- Do not mix benchmark setup notes with method claims in the same table.
- Keep route help/harm, parser failures, and decontamination status in the telemetry for every row.
