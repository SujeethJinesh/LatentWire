# L-IB1 Privacy-Bottleneck Report

RUN_STATUS: CPU_CACHED_ANALYSIS_ONLY

## Verdict

- verdict: `KILL_UTILITY_IS_IDENTITY`
- reason: No adversarial bottleneck preserved current-level utility; the best private code dropped well below the current packet and still leaked source/evidence structure.
- input rows: `512` matched packet rows, `512` target-only rows
- dev/gate split: `341` / `171`
- current packet cached utility: `0.875000`
- current packet max leakage: `1.000000`

## Frontier

| Variant | Kind | Bytes | Cached utility | Proxy utility | Family leak | Source-top1 leak | Candidate-id leak | Evidence-atom leak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `target_only` | `baseline` | 0 | 0.250000 | 0.222222 | 0.099415 | 0.385965 | 0.222222 | 0.321637 |
| `current_high_utility_packet` | `baseline` | 8 | 0.875000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `same_byte_visible_exact_public_code` | `text_baseline` | 8 | 0.875000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `adaptive_anonymized_text_coarse_atoms` | `text_baseline` | 8 |  | 0.754386 | 0.900585 | 0.754386 | 0.754386 | 0.853801 |
| `random_same_byte` | `control` | 8 |  | 0.280702 | 0.099415 | 0.345029 | 0.280702 | 0.321637 |
| `adv_select_lam0_k2` | `l_ib1_candidate` | 4 |  | 0.602339 | 0.391813 | 0.491228 | 0.602339 | 0.602339 |

Top L-IB1 candidates by proxy utility:

| Rank | Variant | Proxy utility | Max leakage | Notes |
|---:|---|---:|---:|---|
| 1 | `adv_select_lam0_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 2 | `adv_select_lam0.25_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 3 | `adv_select_lam0.5_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 4 | `adv_select_lam1_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 5 | `adv_select_lam2_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 6 | `adv_select_lam4_k2` | 0.602339 | 0.602339 | selected atoms=['default', 'integer'] |
| 7 | `adv_select_lam0_k3` | 0.602339 | 0.713450 | selected atoms=['default', 'integer', 'average'] |
| 8 | `adv_select_lam0_k4` | 0.602339 | 0.713450 | selected atoms=['default', 'integer', 'average', 'mean'] |

## Interpretation

The cached high-utility packet is useful only when it preserves the exact source/evidence structure. Adversarial feature-selection bottlenecks that hide those atoms drop well below current-packet utility, while exact-evidence baselines expose the same source/evidence labels. Under this cached proxy, task utility and source/evidence identity are not separable enough for a privacy-positive claim.

This is not a live model-forward result and does not test a trained neural IB encoder. It is the requested last cheap cached gate: PASS would have justified a full privacy-bottleneck build; this KILL strengthens the bounded-negative paper by showing that the useful cached packet signal is identity/evidence-bearing.
