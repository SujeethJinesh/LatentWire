# HellaSwag Fixed-Hybrid True Candidate-Text Permutation Gate 1024

Date: 2026-05-04

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full remains
  blocked by full-validation/seed/cross-family evidence and native systems
  measurements.
- Current story: the HellaSwag fixed-hybrid packet now survives a doubled
  physical candidate-text permutation surface over `1024` rows with fresh score
  and hidden caches.
- Exact gap: this is still not all-24 candidate permutation invariance, not a
  full-validation permutation rerun, and not a learned common-basis receiver.

## Contribution Status

Current defensible contributions:

1. Source-private fixed-byte packet protocol with no source text, KV, hidden
   vector, score vector, or logits transmitted.
2. Full cached HellaSwag fixed-hybrid packet row plus strict candidate-position
   and candidate-text hardening.
3. Systems boundary accounting at `1B` raw / `4B` framed for the final packet.

Still needs work:

- seed repeats and full-validation candidate-text hardening;
- strict same-family versus cross-family separation;
- a learned receiver/common-basis method that beats packet-only;
- native GPU latency, memory-traffic, and energy rows.

## Gate

Artifact:
`results/source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate_20260504_validation0_1024/`

Script:
`scripts/build_source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.py`

Design:

- Use HellaSwag validation rows `0:1024`.
- Apply one non-identity candidate-ending permutation per example, cycling
  through eight fixed permutations.
- Physically reorder candidate text and update the displayed answer index.
- Rerun the hidden fixed-hybrid pipeline from scratch on the permuted rows.
- Require fresh score and hidden caches.
- Remap display-coordinate predictions back to canonical candidate IDs.
- Compare against original same-row fixed-hybrid predictions from the
  unpermuted validation-first1024 artifact.

## Result

The `1024`-row physical candidate-text hardening gate passes its promotion
flag.

| Metric | Value |
| --- | ---: |
| eval rows | `1024` |
| original fixed-hybrid accuracy | `0.518555` |
| remapped fixed-hybrid accuracy | `0.523438` |
| remapped minus original | `+0.004883` |
| paired CI95 low vs original | `-0.004883` |
| canonical consistency rate | `0.957031` |
| unremapped fixed-hybrid accuracy | `0.212891` |
| wrong-remap fixed-hybrid accuracy | `0.167969` |
| wrong-remap CI95 high vs remapped | `-0.307593` |
| score cache hit | `false` |
| hidden cache hit | `false` |
| raw packet bytes | `1` |
| framed packet bytes | `4` |
| source text/KV/hidden/score-vector/logits transmitted | `false` |

The permuted hidden pipeline itself also remains positive:

| Metric | Value |
| --- | ---: |
| selected accuracy | `0.523438` |
| best label-copy accuracy | `0.461914` |
| selected minus best label-copy | `+0.061523` |
| paired CI95 low vs best label-copy | `+0.042944` |
| score-channel roll hidden control | `0.258789` |
| wrong-example hidden control | `0.407227` |
| candidate-roll hidden control | `0.389648` |
| runtime | `529.57s` wall on Mac CPU |

## Interpretation

This strengthens the reviewer-facing option-order defense. The full hidden
fixed-hybrid packet follows candidate text under a fresh physical shuffle on a
standard `1024`-row surface. The remapped predictions preserve accuracy and
remain close to the original same-row predictions, while unremapped and
wrong-remapped controls collapse.

The result is not exact invariance. Consistency is `0.957031`, not `0.99+`.
It uses one non-identity permutation per row, not all `24` permutations. It is
also not a learned receiver or common latent language.

## Lay Explanation

We shuffled the four answer endings, reran the source pipeline, and translated
the displayed answer number back to the original answer number. If the method
were just memorizing answer slots, it would fail after the shuffle. It mostly
kept choosing the same original candidate, while wrong remaps failed badly.

## Decision

Promote this as `1024`-row physical candidate-text hardening for the full
hidden fixed-hybrid HellaSwag packet.

Next exact gate: either full-validation candidate-text hardening if compute is
available, or all-24 permutations over a smaller frozen slice. The next method
branch after this hardening surface is decision-supervised SAE/crosscoder
hidden-innovation packets with atom-shuffle, wrong-row, and top-atom knockout
controls.
