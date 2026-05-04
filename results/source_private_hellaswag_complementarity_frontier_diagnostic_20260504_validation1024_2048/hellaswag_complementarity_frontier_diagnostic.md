# HellaSwag Complementarity-Frontier Diagnostic

- created UTC: `2026-05-04T22:10:29.139672+00:00`
- pass gate: `False`
- eval rows: `768`

## Headline

- target-only accuracy: `0.263021`
- fixed-hybrid accuracy: `0.467448`
- source top1/top2 oracle accuracy: `0.675781`
- target-wrong/source-can-help rows: `386`
- fixed-wrong/source-can-help rows: `174`
- selected selector accuracy: `0.467448`
- selected selector delta vs fixed: `0.000000`
- best destructive control: `source_row_shuffle_frontier_control` at `0.467448`

## Method Rows

| Method | Accuracy | Baseline | Delta | CI95 Low | Helps | Harms | Overrides | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_or_source_top1_top2_oracle_diagnostic | `0.694010` | `0.467448` | `0.226562` | `0.196615` | `174` | `0` | `174` | `0` |
| frontier_selector_over_fixed_hybrid | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `4` |
| source_row_shuffle_frontier_control | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `4` |
| candidate_roll_frontier_control | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `4` |
| random_same_byte_frontier_control | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `4` |
| target_derived_packet_frontier_control | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `0` |
| zero_source_frontier_control | `0.467448` | `0.467448` | `0.000000` | `0.000000` | `0` | `0` | `0` | `0` |
| candidate_only | `0.455729` | `0.467448` | `-0.011719` | `-0.022135` | `5` | `14` | `26` | `4` |
| source_top1_choice_control | `0.411458` | `0.467448` | `-0.055990` | `-0.079427` | `22` | `65` | `134` | `4` |
| source_pair_phi_choice | `0.329427` | `0.467448` | `-0.138021` | `-0.178385` | `79` | `185` | `375` | `4` |
| target_only | `0.263021` | `0.467448` | `-0.204427` | `-0.255208` | `105` | `262` | `551` | `0` |

## Interpretation

The diagnostic tests whether source helpfulness is separable before another receiver is trained. A large oracle gap with a non-positive selected frontier selector means the benchmark has source headroom, but the current packet fields do not expose a stable source-causal decision boundary.

## Lay Explanation

We looked for questions where Phi is wrong but Qwen's top guesses include the right answer. Then we asked whether a tiny packet can predict those moments without seeing test answers. If this fails, another decoder on the same packet is unlikely to become a strong paper result.
