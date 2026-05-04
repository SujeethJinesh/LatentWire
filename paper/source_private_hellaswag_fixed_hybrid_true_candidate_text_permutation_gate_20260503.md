# HellaSwag Fixed-Hybrid True Candidate-Text Permutation Gate

Date: 2026-05-03

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full remains
  blocked.
- Current story: the HellaSwag fixed-byte packet row now has full cached
  validation, cached option-position hardening, score-channel physical
  candidate-text permutation evidence, and a full hidden fixed-hybrid physical
  candidate-text permutation smoke over `512` rows.
- Exact gap: this still is not a learned receiver/common-basis method and not
  full-validation/all-permutation invariance. ICLR still needs a positive
  learned method or much stronger systems/benchmark story.

## Contribution Status

Current defensible contributions:

1. Source-private fixed-byte packet protocol with source-destroying controls.
2. HellaSwag fixed hybrid vote-on-score-agreement packet policy over full
   cached validation.
3. Candidate-position and physical candidate-text hardening for the packet
   stack.
4. Byte/exposure accounting separating tiny packet protocols from source KV or
   vector transport systems.

Still weak:

- learned receiver/common-basis novelty;
- cross-family latent reasoning beyond source-choice packets;
- benchmark breadth;
- native NVIDIA systems rows.

## Gate

Artifact:
`results/source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate_20260503_validation0_512/`

Script:
`scripts/build_source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.py`

Test:
`tests/test_build_source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.py`

Design:

- Use HellaSwag validation rows `0:512`.
- Assign one non-identity fixed candidate-ending permutation per canonical
  row, cycling through eight fixed permutations.
- Physically reorder the four candidate endings and update the displayed
  answer index.
- Rerun the existing hidden-innovation eval-slice stress pipeline on the
  permuted candidate text.
- Verify the rerun generated fresh score and hidden caches.
- Remap display-space fixed-hybrid predictions back to canonical candidate IDs.
- Compare remapped predictions with the original same-row fixed-hybrid
  predictions from the unpermuted first validation slice.

The heavy rerun used:

- `source_model.cache_hit`: `false`
- `hidden_model.cache_hit`: `false`
- `eval rows`: `512`
- `source_lm_device`: `cpu`
- `source_lm_dtype`: `float32`
- `source_lm_prompt_mode`: `continuation`

## Result

The 512-row hidden fixed-hybrid candidate-text hardening gate passes its
promotion flag.

| Metric | Value |
| --- | ---: |
| eval rows | `512` |
| original fixed-hybrid accuracy | `0.525391` |
| remapped fixed-hybrid accuracy | `0.531250` |
| remapped minus original | `+0.005859` |
| paired CI95 low vs original | `-0.009766` |
| canonical consistency rate | `0.955078` |
| unremapped fixed-hybrid accuracy | `0.236328` |
| wrong-remap fixed-hybrid accuracy | `0.177734` |
| wrong-remap CI95 high vs remapped | `-0.288965` |
| score cache hit | `false` |
| hidden cache hit | `false` |
| raw packet bytes | `1` |
| framed packet bytes | `4` |
| source text/KV/hidden/score-vector/logits transmitted | `false` |

The permuted hidden pipeline itself also remains positive on the shuffled
candidate text:

| Metric | Value |
| --- | ---: |
| selected accuracy | `0.531250` |
| best label-copy accuracy | `0.482422` |
| selected minus best label-copy | `+0.048828` |
| paired CI95 low vs best label-copy | `+0.021484` |
| score-channel roll hidden control | `0.246094` |
| wrong-example hidden control | `0.414062` |

## Interpretation

This is the first direct evidence that the full hidden fixed-hybrid packet is
not merely tied to the original answer slots. The source model sees physically
reordered candidate strings, the hidden and score caches are regenerated, and
the remapped fixed-hybrid predictions stay close to the original same-row
predictions while unremapped and wrong-remapped predictions collapse.

The canonical consistency rate is `0.955078`, not `0.99+`, so the result should
be framed as strong hardening rather than exact invariance. It is one
non-identity permutation per row, not all `24` permutations, and it covers
`512` rows, not full validation.

## Lay Explanation

We shuffled the answer endings, reran the hidden hybrid source pipeline from
scratch, and then translated the displayed answer number back to the original
candidate number. The remapped predictions stayed close to the original
predictions, while treating the displayed number as if it were the original
number collapsed badly. That means the full hybrid hint is mostly following the
answer text, not just the answer slot.

## Decision

Promote this as 512-row physical candidate-text hardening for the full hidden
fixed-hybrid HellaSwag packet.

Do not promote it as:

- exact candidate-order invariance;
- full-validation permutation invariance;
- learned receiver/common-basis evidence;
- native systems speedup;
- final ICLR-ready positive latent communication.

The next exact gate is either:

1. widen this hidden fixed-hybrid permutation gate to `1024` rows or all `24`
   permutations on `512` rows, or
2. start the higher-EV learned method branch: decision-supervised
   SAE/crosscoder hidden-innovation packets with atom-shuffle, wrong-row, and
   top-atom knockout controls.
