# Claim-to-Evidence Audit

Date: 2026-05-02

## Scope

This audit checks whether the COLM paper claims are supported by tracked code,
tests, ablations, and frozen result artifacts in `colm_final`.

## Headline Claims

| Paper claim | Evidence artifact | Status | Notes |
|---|---|---|---|
| ARC-Challenge 8-byte packet reaches 0.344 over 10 seeds on 1172 test examples | `evidence/results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502_budget8_10seed_b2000/arc_challenge_fourier_anchor_syndrome_gate.md` | Supported | Reports packet 0.344, target 0.265, text 0.300, min CI95 lower lift +0.038, pass 10/10. |
| ARC destructive controls fail in 0/10 seeds | Same ARC artifact and `per_variant_seed_metrics.csv` | Supported | Anchor-ID shuffle, anchor-value shuffle, and spectral-bin permutation all fail 0/10. |
| Random shared anchors work nearly as well as named anchors | Same ARC artifact | Supported with caveat | The paper correctly frames this as shared-coordinate evidence, not semantic-anchor evidence. |
| OpenBookQA 3-byte packet reaches 0.378 over 5 seeds on 500 test examples | `evidence/results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/arc_challenge_seed_stability.md` | Supported | Reports 0.378-0.380, target 0.276, text 0.350, min CI95 lower lift +0.038. |
| Phi-3 cross-family replacement fails | `evidence/results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000/source_family_cache_falsification.md` | Supported | Reports pass gate false; full test packet 0.244 vs target 0.265. |
| Failure decomposition shows packet follows source choice at about 0.996-0.997 | `evidence/results/source_private_arc_cross_family_failure_decomposition_20260502/arc_cross_family_failure_decomposition.md` | Supported | This is a major reviewer caveat: packet transfer currently resembles source-choice-preserving candidate evidence. |
| Cached candidate/hidden-query connector repairs fail | `evidence/results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502` and `evidence/results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement` | Supported | Paper states these are negative Mac-local diagnostics, not completed learned connectors. |
| Systems result is 6-11 framed bytes vs 768B one-token 1-bit KV floor, a 69.8x object-size gap | `evidence/results/source_private_systems_boundary_figure_table_20260502/systems_boundary_table.md` | Supported with boundary | The paper correctly says this is byte/exposure accounting, not native latency or HBM bandwidth. |

## Test Coverage

The targeted COLM tests passed:

```text
16 passed in 1.09s
```

The full command and output are in `audits/test_report.txt`.

## Gaps That Remain

- No direct source-choice/index baseline is in the paper yet.
- No native GPU serving measurements are claimed or provided.
- No strict cross-family positive has passed.
- The appendix artifact list is useful but not a complete reproducibility
  manifest with model snapshot hashes and exact cache provenance.
- Late HellaSwag diagnostics are copied under
  `evidence/excluded_diagnostics/`; they are relevant to next-paper strategy but
  are not current PDF claims.

## Lay Explanation

The main experiment asks whether one model can send a tiny hint to another
model when both see the same multiple-choice question. The hint helps for
closely related model families and stops helping when the shared coordinate
system or sender family is changed. That supports a narrow packet-transfer
result, not a general claim that all models now share a latent language.
