# HellaSwag Fixed-Hybrid Seeded Candidate-Text Permutation Gate

Date: 2026-05-04

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full remains
  blocked by cross-slice/cross-family stability, learned/common-basis evidence,
  and native systems rows.
- Current story: the HellaSwag fixed-hybrid `1B` raw / `4B` framed packet now
  survives both the fixed cyclic `1024`-row physical candidate-text shuffle and
  a seed-controlled shuffle schedule drawn from all non-identity permutations.
- Exact gap: this is seed/permutation-schedule hardening, not all-24
  per-example invariance, full-validation permutation invariance, or learned
  latent communication.

## Contribution Status

Current defensible contributions:

1. Source-private byte-scale packet protocol with no source text, KV, hidden
   vector, score vector, logits, or raw scores transmitted.
2. Full cached HellaSwag fixed-hybrid packet row.
3. Candidate-position, physical candidate-text, and now seed-controlled
   permutation-schedule hardening.
4. Systems boundary accounting for the final `1B` raw / `4B` framed packet.

Still weak:

- no second non-overlapping physical candidate-text slice yet;
- no all-24 clustered permutation expansion;
- no learned SAE/crosscoder receiver row beating packet-only;
- no native NVIDIA serving metrics.

## Gate

Artifact:
`results/source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_seed_gate_20260504_validation0_1024_seed314159/`

Script:
`scripts/build_source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate.py`

Code change:

- `--seed` now affects seeded permutation preparation.
- New modes:
  - `seeded_fixed8_nonidentity`
  - `seeded_all24_nonidentity`
- `seeded_all24_nonidentity` chooses one of the `23` non-identity candidate
  permutations per row using a stable hash of `seed`, global row index, and
  row ID.

Design:

- Use HellaSwag validation rows `0:1024`.
- Draw one non-identity candidate-ending permutation per row from all `23`
  non-identity permutations with seed `314159`.
- Physically reorder candidate text and answer labels.
- Rerun the hidden fixed-hybrid pipeline with fresh score and hidden caches.
- Remap display-coordinate predictions back to canonical candidate IDs.
- Compare against the original same-row first1024 prediction artifact.

## Result

The seeded permutation-schedule hardening gate passes its promotion flag.

| Metric | Value |
| --- | ---: |
| eval rows | `1024` |
| permutation mode | `seeded_all24_nonidentity` |
| permutation seed | `314159` |
| permutation count | `23` |
| original fixed-hybrid accuracy | `0.518555` |
| remapped fixed-hybrid accuracy | `0.523438` |
| remapped minus original | `+0.004883` |
| paired CI95 low vs original | `-0.005859` |
| canonical consistency rate | `0.964844` |
| unremapped fixed-hybrid accuracy | `0.208008` |
| wrong-remap fixed-hybrid accuracy | `0.157227` |
| wrong-remap CI95 high vs remapped | `-0.318359` |
| score cache hit | `false` |
| hidden cache hit | `false` |
| raw packet bytes | `1` |
| framed packet bytes | `4` |

The seeded permuted hidden pipeline itself remains positive:

| Metric | Value |
| --- | ---: |
| selected accuracy | `0.523438` |
| best label-copy accuracy | `0.461914` |
| selected minus best label-copy | `+0.061523` |
| paired CI95 low vs best label-copy | `+0.041943` |
| score-channel roll hidden control | `0.245117` |
| wrong-example hidden control | `0.419922` |
| candidate-roll hidden control | `0.403320` |
| runtime | `521.89s` wall on Mac CPU |

## Interpretation

This closes a real reproducibility gap in the prior permutation evidence. The
previous script accepted `--seed` but ignored it, so seed repeats would have
been fake. The new seeded scheduler produces a different physical
candidate-order assignment and the fixed-hybrid packet still follows candidate
text after canonical remapping.

This should be framed narrowly:

- promoted: seed-controlled physical candidate-text hardening on `1024` rows;
- not promoted: all-24-per-example invariance;
- not promoted: full-validation permutation invariance;
- not promoted: learned common-basis latent transfer;
- not promoted: native systems speedup.

## Lay Explanation

Before this change, asking for a different shuffle seed did nothing. We fixed
that, shuffled the answer endings with a real new random schedule, reran the
source pipeline, and checked whether the final tiny packet still pointed to
the same original answer after translating back from the displayed choice
number. It did.

## Decision

Promote this as seed/permutation-schedule hardening for the full hidden
fixed-hybrid HellaSwag packet.

Next exact gate: run the same physical candidate-text permutation hardening on
the non-overlapping `1024:2048` validation slice with fresh caches and the
correct original prediction artifact. After that, move to the learned
decision-supervised SAE/crosscoder hidden-innovation packet branch.
