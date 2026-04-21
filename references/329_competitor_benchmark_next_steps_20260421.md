# Competitor Benchmark Next Steps

Web check: 2026-04-21. This memo updates the competitor plan for the next
paper-facing benchmark cycle. It is scoped to the methods that are either
already runnable in this repo or are the most plausible near-term bootstraps.

## Primary sources to anchor the matrix

- C2C / Cache-to-Cache: https://arxiv.org/abs/2510.03215
- C2C code: https://github.com/thu-nics/C2C
- KVComm: https://arxiv.org/abs/2510.12872
- KVComm code: https://github.com/Zephyroam/KVComm
- Expected Attention / KVPress: https://arxiv.org/abs/2510.00636
- KVPress code: https://github.com/NVIDIA/kvpress
- KVzip: https://arxiv.org/abs/2505.23416
- KVzip code: https://github.com/snu-mllab/KVzip
- Quest: https://arxiv.org/abs/2406.10774
- Quest code: https://github.com/mit-han-lab/Quest
- Latent space communication via KV-cache alignment: https://arxiv.org/abs/2601.06123
- Q-KVComm: https://arxiv.org/abs/2512.17914
- OjaKV: https://arxiv.org/abs/2509.21623
- HCAttention: https://arxiv.org/abs/2507.19823
- KQ-SVD: https://arxiv.org/abs/2512.05916
- KV Packet: https://arxiv.org/abs/2604.13226

## What is feasible to bootstrap here

| Method | Feasible now? | How to bootstrap in this repo | Main caveat |
| --- | --- | --- | --- |
| C2C | Yes | `scripts/bootstrap_c2c.py`, `scripts/run_c2c_eval.py`, published fuser artifact path in `latent_bridge.baselines.C2CAdapter` | Paper-safe only if we keep the exact model pair, decoding budget, and scoring harness fixed |
| KVComm | Partial | `scripts/run_kvcomm_eval.py` plus the existing ported replay artifacts | Native repo is dirty locally; do not report a GSM claim until the dirty diff is isolated or a clean scratch clone is used |
| KVPress / Expected Attention | Yes | `scripts/run_kvpress_eval.py` against the same GSM slices we already use | Same-model compression control, not a cross-model communication baseline |
| KVzip | Conditional | Vendored clone under `references/repos/KVzip`; use its native GSM harness only if CUDA/flash-attn constraints are satisfied | Likely CUDA-heavy and more fragile than KVPress |
| Quest | Conditional | Vendored clone under `references/repos/Quest`; use as a long-context selection control, not as a GSM comparator | Not GSM-native; keep it in the compression/selection block |
| Latent KV-cache alignment | Indirect | Use the existing internal bridge family as the best proxy: `bridge_ridge`, `headwise_route_atom`, and the grouped transport branches | External code is not yet part of the repo; report it as an internal alignment proxy, not an external baseline |
| Q-KVComm / OjaKV / HCAttention / KQ-SVD / KV Packet | Watchlist | No immediate benchmark row unless a clean repo and stable harness appear | These are better treated as next-cycle baselines or literature pivots, not this week’s primary table |

## Exact benchmark rows to run next

### Priority 1: stabilize the direct competitor block

1. `gsm8k_eval_70` on the exact Qwen2.5-0.5B -> Qwen3-0.6B pair with **C2C**.
2. `gsm8k_100` on the exact Qwen pair with **C2C**.
3. `svamp_eval_70` on the exact Qwen pair with **C2C**.

Why first: C2C is still the strongest direct peer and the current paper risk is
whether we can close the gap on held-out GSM/SVAMP rather than on tiny smokes.

### Priority 2: hold the same-model compression controls fixed

4. `gsm8k_eval_70` with **KVPress none**.
5. `gsm8k_eval_70` with **KVPress ExpectedAttentionPress** at the matched compression budget.
6. `gsm8k_100` with **KVPress none** and **ExpectedAttentionPress** if the harness stays stable.

Why: these rows tell us whether a change is actually cross-model semantic
transfer or just same-model cache selection.

### Priority 3: expand the internal latent-alignment proxy

7. `gsm8k_eval_70` for the best current internal alignment proxy, not just
`target-alone`, but the strongest non-oracle branch from the bridge family.
8. `gsm8k_100` for the same branch.
9. `svamp_eval_70` for the same branch.

Recommended proxy rows to keep in the table:
- `bridge_ridge`
- `headwise_route_atom`
- `grouped_subspace_transport + rank-4 residual`

Why: if the paper is going to claim anything about latent alignment, the
proxy family must be measured on the same held-out slices as C2C.

### Priority 4: run the next compression/computation comparator only if CUDA is available

10. `gsm8k_eval_70` with **KVzip** at a matched compression ratio.
11. `gsm8k_100` with **KVzip** at the same ratio.
12. `gsm8k_eval_70` with **Quest** only if we are explicitly using it as a
query-aware long-context control, not as a semantic communication claim.

Why: KVzip is the best next compression comparator after KVPress; Quest is a
selection-control baseline, not a direct peer.

### Priority 5: isolate KVComm before giving it a paper row

13. Run **KVComm** on its native supported tasks first, not GSM, unless the
ported replay path is validated on a clean clone.
14. If and only if the port is clean, add a GSM-compatible replay row for the
exact Qwen pair.

Why: KVComm is relevant, but the local clone is dirty and the native task set is
not GSM. This is the highest-risk comparator in the stack.

## Compute and risk estimates

| Row family | Estimated cost | Risk level | Main failure mode |
| --- | ---: | --- | --- |
| C2C GSM70 / GSM100 / SVAMP70 | Low to moderate | Low | Parser mismatch or artifact path mismatch |
| KVPress GSM70 / GSM100 | Low | Low | Neutral result that only confirms the same-model control |
| Internal latent proxy rows | Low to moderate | Low | Overfitting to the current Qwen pair if we do not keep the held-out split fixed |
| KVzip GSM rows | Moderate to high | Medium | CUDA / flash-attn / local patch friction |
| Quest rows | Moderate to high | Medium | Not GSM-native; easy to misstate as semantic communication |
| KVComm rows | Moderate to high | High | Dirty clone, native-task mismatch, and replay semantics |

Rough wall-clock guidance on this machine:
- C2C / KVPress: usually the cheapest paper-safe rows, roughly single-digit
  minutes per small held-out slice and tens of minutes for the larger ones.
- Internal proxy rows: similar to C2C or cheaper, because they stay inside the
  existing harness.
- KVzip / Quest: expect a larger setup tax and longer run times, especially if
  CUDA is needed for faithful reproduction.
- KVComm: do not spend a full paper day until the clean-clone status is settled.

## Reporting rules

- Keep direct communication baselines separate from same-model compression
  controls.
- Use the same source-target pair, prompt template, decoding budget, and answer
  extractor for every comparable row.
- Report bytes, latency, tokens/sec, and route/retention telemetry alongside
  accuracy.
- Do not promote a row to the main paper unless it survives the held-out
  slices, not just the GSM30 smoke.

## Immediate recommendation

If we only run four rows next, run these:

1. `gsm8k_eval_70` C2C
2. `gsm8k_100` C2C
3. `gsm8k_eval_70` KVPress ExpectedAttentionPress
4. `gsm8k_eval_70` best internal latent-alignment proxy

That gives the cleanest next update on whether the gap is still semantic
transfer, same-model selection, or bridge geometry.
