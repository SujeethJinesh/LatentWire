# Random Selector Salt Repeat Summary

Date: 2026-04-21

Question: does the GSM30 `random/random` selector gain survive different
deterministic random masks?

Common settings:

- Source: `Qwen/Qwen2.5-0.5B-Instruct`
- Target: `Qwen/Qwen3-0.6B`
- Gate: fixed `0.10`
- K/V split: route `0.25`, value `0.75`
- Route metric: `random`
- Value metric: `random`
- Position metric: `attention`
- Source/target chat templates enabled, thinking disabled

| Salt | Target | RotAlign | Delta | Method-only | Baseline-only | Both correct | Both wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0667 | 0.1333 | +0.0667 | 4 | 2 | 0 | 24 |
| 1 | 0.0667 | 0.1333 | +0.0667 | 3 | 1 | 1 | 25 |
| 2 | 0.0667 | 0.0000 | -0.0667 | 0 | 2 | 0 | 28 |

Flip indices:

| Salt | Method-only indices | Baseline-only indices |
|---:|---|---|
| 0 | `5, 9, 28, 29` | `13, 17` |
| 1 | `7, 8, 24` | `17` |
| 2 | none | `13, 17` |

Interpretation:

Single-mask stochastic routing is not stable enough to be a positive method
claim. Salt 1 replicates the salt 0 aggregate gain, but salt 2 collapses below
target-alone. The useful signal is likely route diversity or perturbation
regularization, not one deterministic random mask.

Next method lane:

- Treat stochastic routing as a candidate generator, not the final method.
- Add seed ensembles with fixed byte/latency accounting.
- Add answer reranking or verifier passes against `target_alone`.
- Log answer entropy, route entropy, method-only/baseline-only flips, and seed
  variance for every stochastic run.
- Keep deterministic attention/energy as a semantic selector baseline, but do
  not claim selector semantics from the random result.
