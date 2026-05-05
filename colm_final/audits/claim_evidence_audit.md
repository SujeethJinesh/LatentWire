# Claim-to-Evidence Audit

Date: 2026-05-05

## Scope

This audit checks whether the COLM paper claims are supported by tracked code,
tests, ablations, and frozen result artifacts in `colm_final`.

## COLM_v3 Addendum

The integrated COLM_v3 draft now uses the generated review packet at
`results/latentwire_colm_v3_review_packet_20260505/` as its source-of-truth
claim audit. The main paper claim is:

> LatentWire provides a practical protocol and evaluation framework for
> byte-scale, source-private model-to-model communication, with controlled
> evidence of narrow packet utility, explicit utility-per-byte accounting, and
> strong safeguards against shortcut claims.

Additional v3-integrated claims:

| Paper claim | Evidence artifact | Status | Notes |
|---|---|---|---|
| The source-private threat model forbids source text, hidden states, KV cache, source logits, score vectors, and labels | `colm_final/paper/latentwire_colm2026.tex`; `results/latentwire_colm_v3_review_packet_20260505/claim_audit.csv` | Supported | Now visible in the main method section. |
| The control suite includes target-only, same-budget text, source-index/rank/score, wrong-row, same-source-choice, coordinate shuffle, candidate derangement, source-family substitution, and cached connector repair controls | `colm_final/paper/latentwire_colm2026.tex`; `results/latentwire_colm_v3_review_packet_20260505/table_figure_inventory.csv` | Supported | Now visible as a main paper table. |
| Systems rows separate measured/accounted packet objects from analytical KV/cache byte floors and native-pending systems rows | `results/latentwire_colm_v3_review_packet_20260505/systems_measured_vs_estimated.csv`; `colm_final/paper/latentwire_colm2026.tex` | Supported | No native GPU throughput, HBM, energy, latency, or C2C superiority claim is made. |
| HybridKernel, SinkAware, and ThoughtFlow-FP8 are future systems side lanes, not COLM_v3 evidence | `experimental/status_20260505.md`; `results/latentwire_colm_v3_review_packet_20260505/experiment_scoping.csv` | Supported | They should not enter main claims until measured artifacts exist. |
| The reviewer-facing claim should be source-private candidate-transfer packets rather than broad latent communication | `colm_final/audits/colm_v3_10_reviewer_panel_20260505.md`; `colm_final/paper/latentwire_colm2026.tex` | Supported | Ten-reviewer panel mean is 6.4/10 under scoped workshop framing; stronger framing is unsupported. |
| The fragile comparator surfaces are visible in main text | `colm_final/paper/latentwire_colm2026.tex`; `evidence/results/source_private_colm_acceptance_baselines_20260502/colm_acceptance_baseline_audit.md` | Supported | New uncertainty table reports packet-target, packet-text, and packet-source lower bounds. |

## Headline Claims

| Paper claim | Evidence artifact | Status | Notes |
|---|---|---|---|
| ARC-Challenge 8-byte packet reaches 0.344 over 10 seeds on 1172 test examples | `evidence/results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed_b2000/arc_challenge_fourier_anchor_syndrome_gate.md` | Supported | Reports packet 0.344, target 0.265, text 0.300, min CI95 lower lift +0.038, pass 10/10. |
| ARC source-choice accuracy is 0.346 and packet follows source choice at about 0.995 | Same ARC artifact plus `matched_predictions.jsonl` | Supported | This is now stated as a central claim boundary: the current positive mostly transports source-selected candidates. |
| Explicit ARC source-index baseline reaches 0.346 and is not beaten by the packet | `evidence/results/source_private_colm_acceptance_baselines_20260502/colm_acceptance_baseline_audit.md` | Supported | Packet-source paired CI95 lower bound is -0.008. This is now in the main table, not hidden in limitations. |
| ARC destructive controls fail in 0/10 seeds | Same ARC artifact and `per_variant_seed_metrics.csv` | Supported | Anchor-ID shuffle, anchor-value shuffle, and spectral-bin permutation all fail 0/10. |
| Random shared anchors work nearly as well as named anchors | Same ARC artifact | Supported with caveat | The paper correctly frames this as shared-coordinate evidence, not semantic-anchor evidence. |
| OpenBookQA 3-byte packet reaches 0.378 over 5 seeds on 500 test examples | `evidence/results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/arc_challenge_seed_stability.md` | Supported | Reports 0.378-0.380, target 0.276, text 0.350, min CI95 lower lift +0.038. |
| Explicit OpenBookQA source-index baseline reaches 0.378 and is not beaten by the packet | `evidence/results/source_private_colm_acceptance_baselines_20260502/colm_acceptance_baseline_audit.md` | Supported | Packet-source paired CI95 lower bound is -0.006; packet-text mean gap is +0.028 with lower bound +0.000. |
| Packet payload rate curve over 2/3/4/8 bytes | `evidence/results/source_private_colm_acceptance_baselines_20260502/rate_curve.csv` and `paper/figures/rate_curve.pdf` | Supported | The 1-byte source-index row is reported separately because the packet encoder has a 2-byte payload minimum. |
| Phi-3 cross-family replacement fails | `evidence/results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000/source_family_cache_falsification.md` | Supported | Reports pass gate false; full test packet 0.244 vs target 0.265. |
| Failure decomposition shows packet follows source choice at about 0.996-0.997 | `evidence/results/source_private_arc_cross_family_failure_decomposition_20260502/arc_cross_family_failure_decomposition.md` | Supported | This is a major reviewer caveat: packet transfer currently resembles source-choice-preserving candidate evidence. |
| Cached candidate/hidden-query connector repairs fail | `evidence/results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502` and `evidence/results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement` | Supported | Paper states these are negative cached-artifact diagnostics, not completed learned connectors. |
| Systems result is 6-11 framed bytes vs a 768B one-token 1-bit-per-KV-element accounting floor, a 69.8x object-size gap | `evidence/results/source_private_systems_boundary_figure_table_20260502/systems_boundary_table.md` | Supported with boundary | The paper correctly says this is byte/exposure accounting, not native latency, QJL-native, or HBM bandwidth. |

## Test Coverage

The full repository suite and targeted COLM tests passed:

```text
1328 passed in 146.19s
16 passed in 18.84s
4 passed in 0.12s
```

The full command and output are in `audits/test_report.txt`.

## Gaps That Remain

- The strict positive-beyond-source-index gate fails; the paper now scopes the
  positive as source-candidate transfer rather than selected-candidate
  compression.
- No native GPU serving measurements are claimed or provided.
- No strict cross-family positive has passed.
- Calibrated source-score-vector quantization is not available for the headline
  frozen caches; the packaged audit includes source-index, source-label,
  source-rank-code, entropy-matched random-index, and same-budget text.
- The appendix artifact list is useful but compact. Full commands and artifact
  hashes are in `audits/reproducibility_report.md`; model snapshot IDs and a
  normalized JSON rerun-diff mode remain useful additions.
- Late HellaSwag diagnostics are copied under
  `evidence/excluded_diagnostics/`; they are relevant to next-paper strategy but
  are not current PDF claims.

## Lay Explanation

The main experiment asks whether one model can send a tiny hint to another
model when both see the same multiple-choice question. The hint helps for
closely related model families and stops helping when the shared coordinate
system or sender family is changed. That supports a narrow packet-transfer
result, not a general claim that all models now share a latent language.
