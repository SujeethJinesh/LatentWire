# 2026-04-21 Telemetry Summary

## GSM30 Separable K/V Candidate

Run:

`qwen_gsm30_dynalign_prefdist_asym_kv_routeattn_valueenergy_r025_v075_cal16_chat_telemetry.jsonl`

Configuration:

- Source: `Qwen/Qwen2.5-0.5B-Instruct`
- Target: `Qwen/Qwen3-0.6B`
- Translator:
  `checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`
- Gate: fixed `0.10`
- K/V split: route `0.25`, value `0.75`
- Route metric: `attention`
- Value metric: `energy`

Aggregate:

| Method | Accuracy | Avg bytes | Avg latency sec | Avg tokens/sec |
|---|---:|---:|---:|---:|
| target_alone | 0.0667 | n/a | 7.1639 | 8.9337 |
| rotalign_kv_gate_0.10 | 0.0667 | 1,406,292.3 | 7.7724 | 8.0198 |

Paired outcome against `target_alone`:

| Paired n | Delta accuracy | Method-only | Baseline-only | Both correct | Both wrong | McNemar p | Bootstrap delta low | Bootstrap delta high |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.0000 | 1 | 1 | 1 | 27 | 1.0000 | -0.1000 | 0.1000 |

Non-neutral examples:

| Index | Example id | Flip | Method answer | Baseline answer | Generated tokens | Bytes | Route/value overlap | Route/value Jaccard | Source/target token ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `c11b1f65a91b1796` | method-only | 20 | 25 | 37 | 2,127,972.0 | 0.697 | 0.213 | 1.189 |
| 13 | `c1d4c219268d7f10` | baseline-only | 2 | 20 | 64 | 1,635,392.5 | 0.659 | 0.196 | 1.246 |
| 17 | `d750c66e733a2837` | both-correct | 3 | 3 | 64 | 1,172,272.5 | 0.664 | 0.198 | 1.341 |

Interpretation:

The separable K/V branch is neutral on GSM30. It can rescue a target-alone
failure, but it can also introduce a new failure. The method is still useful as
an ablation lane because the paired sidecar now exposes example-level flips and
mechanism telemetry.

## Matched Selector Matrix

All rows use the same byte budget and fixed gate.

| Selector | Target | Method | Delta | Method-only | Baseline-only | McNemar p | Bootstrap low | Bootstrap high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| route attention / value energy | 0.0667 | 0.0667 | 0.0000 | 1 | 1 | 1.0000 | -0.1000 | 0.1000 |
| route attention / value attention | 0.0667 | 0.1000 | +0.0333 | 2 | 1 | 1.0000 | -0.0667 | 0.1667 |
| route random / value random | 0.0667 | 0.1333 | +0.0667 | 4 | 2 | 0.6831 | -0.1000 | 0.2333 |

Interpretation:

The current GSM30 evidence is not selector-semantic. Random routing currently
has the largest method-only excess, while attention/energy is neutral. The next
real-model blocker is therefore not just better attention metrics; it is
separating beneficial cache perturbation from true cross-model communication.

## Random Selector Seed Variance

Run summary:

`random_salt_repeat_summary_20260421.md`

| Salt | Target | Method | Delta | Method-only | Baseline-only | Method-only indices | Baseline-only indices |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0.0667 | 0.1333 | +0.0667 | 4 | 2 | `5, 9, 28, 29` | `13, 17` |
| 1 | 0.0667 | 0.1333 | +0.0667 | 3 | 1 | `7, 8, 24` | `17` |
| 2 | 0.0667 | 0.0000 | -0.0667 | 0 | 2 | none | `13, 17` |

Interpretation:

The stochastic result is real enough to study but not stable enough to claim.
Two masks are positive and one is negative. The flip indices move across seeds,
which suggests the branch is sampling useful and harmful route perturbations.
This points to multi-route aggregation, uncertainty-triggered routing, or a
target verifier as the next positive-method lane.

## Protected-Channel Residual Codebook Toy

Run:

`../query_pool_toy_20260421/query_pool_protected_channel_residual_codebook_vs_topk.md`

| Scenario | Residual codebook | Protected residual codebook | Delta |
|---|---:|---:|---:|
| aligned | 0.5417 | 0.5938 | +0.0521 |
| rotated | 0.6406 | 0.5729 | -0.0677 |
| outlier | 0.5677 | 0.6198 | +0.0521 |
| slot_permuted | 0.5417 | 0.4948 | -0.0469 |

Interpretation:

Fixed protected channels help when the protected coordinates are aligned or
carry true outlier energy, but they hurt under gauge rotation and slot
permutation. This supports a more constrained next step: protect channels only
after gauge alignment, or learn the protected mask jointly with the bridge.

## Gauge-Aware Protected-Channel Toy

Run:

`../query_pool_toy_20260421/query_pool_gauge_aware_protected_channel_vs_topk.md`

| Scenario | Residual | Fixed protected | Gauge-aware protected | Gauge-aware delta vs fixed | Gauge-aware delta vs residual |
|---|---:|---:|---:|---:|---:|
| aligned | 0.5417 | 0.5938 | 0.5469 | -0.0469 | +0.0052 |
| rotated | 0.6406 | 0.5729 | 0.5677 | -0.0052 | -0.0729 |
| outlier | 0.5677 | 0.6198 | 0.5677 | -0.0521 | 0.0000 |
| slot_permuted | 0.5417 | 0.4948 | 0.5260 | +0.0313 | -0.0156 |

Interpretation:

PCA-style gauge canonicalization improves reconstruction and partially repairs
slot permutation, but it does not recover the rotated case and does not beat
residual codebook. The protected basis needs task/signal-aware alignment, not
only covariance alignment.

## Signal-Aware Protected-Channel Toy

Run:

`../query_pool_toy_20260421/query_pool_signal_aware_protected_channel_vs_topk.md`

| Scenario | Residual | Fixed protected | Gauge-aware protected | Signal-aware protected |
|---|---:|---:|---:|---:|
| aligned | 0.5417 | 0.5938 | 0.5469 | 0.5469 |
| rotated | 0.6406 | 0.5729 | 0.5677 | 0.5573 |
| outlier | 0.5677 | 0.6198 | 0.5677 | 0.5104 |
| slot_permuted | 0.5417 | 0.4948 | 0.5260 | 0.5781 |

Interpretation:

Supervised signal bases expose useful telemetry and repair the slot-permuted
toy case, but they are not a universal fix. The protected-channel lane needs a
stacked solution: task-aware basis selection plus orientation alignment plus
slot/permutation robustness.
