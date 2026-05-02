# Source-Private Pass/Fail Ledger

- source frontier: `/Users/sujeethjinesh/Desktop/LatentWire/results/source_private_cpu_systems_frontier_20260429/cpu_systems_frontier.json`
- total rows: `104`

## Reviewer Buckets

| Bucket | Rows |
|---|---:|
| `failed_or_pruned` | 42 |
| `paper_ready_evidence` | 3 |
| `positive_needs_more_evidence` | 58 |
| `weak_positive` | 1 |

## Contribution Summary

| Contribution | Paper-ready | Positive needs more evidence | Weak positive | Failed/pruned |
|---|---:|---:|---:|---:|
| Mac endpoint-proxy byte/TTFT frontier | 0 | 16 | 0 | 6 |
| anchor-relative sparse packet cross-family falsification | 0 | 1 | 1 | 6 |
| byte-rate systems frontier | 0 | 2 | 0 | 0 |
| canonical RASP cross-family falsification | 0 | 1 | 0 | 1 |
| canonical RASP larger-slice confirmation | 0 | 1 | 0 | 0 |
| canonical RASP remap robustness | 0 | 0 | 0 | 7 |
| consistency posterior negative ablation | 0 | 1 | 0 | 1 |
| endpoint paired uncertainty | 3 | 0 | 0 | 0 |
| learned Wyner-Ziv cross-family falsification | 0 | 1 | 0 | 5 |
| learned Wyner-Ziv syndrome packet | 0 | 9 | 0 | 0 |
| learned scalar packet | 0 | 8 | 0 | 0 |
| learned target-preserving receiver | 0 | 4 | 0 | 4 |
| model-emitted source packet | 0 | 7 | 0 | 1 |
| protected rotated residual packet ablation | 0 | 0 | 0 | 9 |
| target model decoder ablation | 0 | 7 | 0 | 2 |

## Paper-Ready Evidence Rows

- `endpoint_label_strict_n64_paired_uncertainty`
- `endpoint_core_label_strict_n160_paired_uncertainty`
- `endpoint_label_strict_n160_paired_uncertainty`

## Highest-Risk Failed/Pruned Rows

| Row | Contribution | Surface | Accuracy | Target | Best control | Reason |
|---|---|---|---:|---:|---:|---|
| `endpoint_proxy_holdout_n16_audit_strict_controls` | Mac endpoint-proxy byte/TTFT frontier | holdout seed30 n16 CPU audit strict controls | 0.875 | 0.25 | 0.1875 | valid rate below 0.95 (0.875) |
| `endpoint_proxy_holdout_n32_audit_strict_controls` | Mac endpoint-proxy byte/TTFT frontier | holdout seed30 n32 CPU audit strict controls | 0.84375 | 0.25 | 0.1875 | valid rate below 0.95 (0.844) |
| `endpoint_proxy_core_n64_audit_payload_gated_nearmiss` | Mac endpoint-proxy byte/TTFT frontier | core seed29 n64 CPU audit payload-gated near miss | 0.75 | 0.25 | 0.203125 | valid rate below 0.95 (0.781) |
| `endpoint_proxy_core_n16_audit_strict_controls` | Mac endpoint-proxy byte/TTFT frontier | core seed29 n16 CPU audit strict controls | 0.75 | 0.25 | 0.25 | valid rate below 0.95 (0.750) |
| `endpoint_proxy_core_n32_audit_strict_controls` | Mac endpoint-proxy byte/TTFT frontier | core seed29 n32 CPU audit strict controls | 0.71875 | 0.25 | 0.21875 | valid rate below 0.95 (0.750) |
| `target_decoder_holdout_n16_all_controls` | target model decoder ablation | holdout n16 CPU all controls | 0.75 | 0.25 | 0.3125 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap103_20260429` | canonical RASP remap robustness | remap 103 | 0.519531 | 0.25 | 0.255859 | source artifact marks row fail/near-miss |
| `protected_residual::remap103::budget6` | protected rotated residual packet ablation | remap 103 | 0.478516 | 0.25 | 0.244141 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap131_20260429` | canonical RASP remap robustness | remap 131 | 0.505859 | 0.25 | 0.28125 | source artifact marks row fail/near-miss |
| `protected_residual::remap103::budget4` | protected rotated residual packet ablation | remap 103 | 0.464844 | 0.25 | 0.25 | source artifact marks row fail/near-miss |
| `protected_residual::remap107::budget6` | protected rotated residual packet ablation | remap 107 | 0.453125 | 0.25 | 0.25 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap101_20260429` | canonical RASP remap robustness | remap 101 | 0.494141 | 0.25 | 0.294922 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap109_20260429` | canonical RASP remap robustness | remap 109 | 0.476562 | 0.25 | 0.279297 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap113_20260429` | canonical RASP remap robustness | remap 113 | 0.472656 | 0.25 | 0.279297 | source artifact marks row fail/near-miss |
| `protected_residual::remap101::budget4` | protected rotated residual packet ablation | remap 101 | 0.447266 | 0.25 | 0.263672 | source artifact marks row fail/near-miss |
| `protected_residual::remap101::budget6` | protected rotated residual packet ablation | remap 101 | 0.447266 | 0.25 | 0.263672 | source artifact marks row fail/near-miss |
| `canonical_rasp::source_private_relative_canonical_remap127_20260429` | canonical RASP remap robustness | remap 127 | 0.453125 | 0.25 | 0.275391 | source artifact marks row fail/near-miss |
| `protected_residual::remap107::budget4` | protected rotated residual packet ablation | remap 107 | 0.435547 | 0.25 | 0.257812 | source artifact marks row fail/near-miss |
| `protected_residual::remap107::budget2` | protected rotated residual packet ablation | remap 107 | 0.431641 | 0.25 | 0.255859 | source artifact marks row fail/near-miss |
| `protected_residual::remap103::budget2` | protected rotated residual packet ablation | remap 103 | 0.4375 | 0.25 | 0.265625 | source artifact marks row fail/near-miss |

## 2026-05-02 Addendum: Qwen2.5-1.5B Hidden/Query Connector Smoke

| Row | Contribution | Surface | Accuracy | Target | Best control | Reason |
|---|---|---|---:|---:|---:|---|
| `arc_qwen15_hidden_query_common_basis_n32_n64_budget8` | target-conditioned common-feature connector | ARC Qwen2.5-1.5B vs Qwen0.5 disagreement n32/n64 | 0.234 | 0.328 | 0.385 | loses to cached-source packet, target-only, same-byte text, and destructive controls; CI low vs Qwen-substituted negative |
| `arc_qwen15_sparse_query_bottleneck_n32_n64_budget8` | target-conditioned sparse innovation connector | ARC Qwen2.5-1.5B vs Qwen0.5 disagreement n32/n64 | 0.266 | 0.328 | 0.385 | improves over Qwen-substituted by +0.042 mean but loses to target-only, same-byte text, and cached-source packet; CI low vs Qwen-substituted negative |

Decision: both rows are `failed_or_pruned` for ICLR-positive evidence. The
hidden/query branch is not dead forever, but the next gate should switch to a
target-loss soft-prefix/query-resampler or rematerialized source-score
innovation rather than widening this exact global hidden/query basis.
