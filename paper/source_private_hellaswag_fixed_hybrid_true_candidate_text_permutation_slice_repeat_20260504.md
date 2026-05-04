# HellaSwag Fixed-Hybrid Candidate-Text Permutation Slice Repeat

Date: 2026-05-04

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full remains
  blocked by cross-family generalization, learned/common-basis packet evidence,
  and native systems rows.
- Current story: the HellaSwag fixed-hybrid `1B` raw / `4B` framed
  source-private candidate packet now survives physical candidate-text
  permutation hardening on two non-overlapping `1024`-row validation slices.
- Exact gap: this is a robustness result for a fixed packet policy. It is not
  learned latent-language communication, not all-24-per-example permutation
  invariance, not cross-family proof, and not native serving speedup.

## Contribution Status

Current defensible contributions:

1. Source-private byte-scale candidate packet with no source text, KV, raw
   hidden vector, score vector, logits, or raw scores transmitted.
2. Full cached HellaSwag fixed-hybrid packet row.
3. Physical candidate-text permutation gates with canonical remapping,
   wrong-remap collapse, and fresh score/hidden caches.
4. Two-slice stability check: `0:1024` and `1024:2048`.

Still weak:

- no all-24-per-example clustered permutation expansion;
- no learned SAE/crosscoder/common-basis receiver row;
- no strict cross-family falsification pair for the same gate;
- no native NVIDIA latency, memory-traffic, or energy row.

## Gate

Artifact:
`results/source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate_20260504_validation1024_2048/`

Original prediction artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260503_rank_score_channel_qwen05_train512_validation1024_2048/bagged_gate/predictions.jsonl`

Design:

- Use HellaSwag validation rows `1024:2048`.
- Apply one fixed non-identity candidate-ending permutation per row.
- Physically reorder candidate text and displayed answer labels.
- Rerun the Qwen hidden fixed-hybrid pipeline with fresh score and hidden
  caches.
- Remap display-coordinate predictions back to canonical candidate IDs.
- Compare against the original same-row `1024:2048` prediction artifact.

## Result

The non-overlapping slice repeat passes the promotion flag, but as a robustness
result rather than an accuracy lift over the original unshuffled same-row
artifact.

| Metric | Value |
| --- | ---: |
| eval rows | `1024` |
| permutation mode | `fixed8_nonidentity` |
| original fixed-hybrid accuracy | `0.465820` |
| remapped fixed-hybrid accuracy | `0.463867` |
| remapped minus original | `-0.001953` |
| paired CI95 low vs original | `-0.010742` |
| paired CI95 high vs original | `+0.007812` |
| canonical consistency rate | `0.964844` |
| helps vs original | `11` |
| harms vs original | `13` |
| unremapped fixed-hybrid accuracy | `0.221680` |
| wrong-remap fixed-hybrid accuracy | `0.188477` |
| wrong-remap CI95 high vs remapped | `-0.230469` |
| score cache hit | `false` |
| hidden cache hit | `false` |
| raw packet bytes | `1` |
| framed packet bytes | `4` |

The permuted hidden pipeline itself remains positive on the shuffled candidate
text:

| Metric | Value |
| --- | ---: |
| selected accuracy | `0.463867` |
| best label-copy accuracy | `0.416992` |
| selected minus best label-copy | `+0.046875` |
| paired CI95 low vs best label-copy | `+0.024390` |
| paired CI95 low vs score-only bagged | `+0.034131` |
| paired CI95 low vs source-rank-only bagged | `+0.035132` |
| jackknife pass count | `3 / 3` |
| score-channel roll hidden control | `0.264648` |
| wrong-example hidden control | `0.382812` |
| candidate-roll hidden control | `0.353516` |
| runtime | `515.27s` wall on Mac CPU |

## Interpretation

This repeat materially reduces the risk that the first `0:1024` permutation
gate was a slice artifact. The key invariance signal is high canonical
prediction consistency (`0.964844`) under physical candidate-text reordering,
while unremapped and intentionally wrong remaps collapse. That supports the
claim that the final byte-scale packet follows candidate identity rather than
display slot.

The accuracy delta versus the original same-row artifact is slightly negative
(`-0.001953`) and the paired interval crosses zero. That means this result
should not be framed as a fresh accuracy improvement. It should be framed as a
second-slice robustness/hardening result for an already positive fixed-hybrid
packet policy.

## Related-Work Boundary

The uniqueness boundary remains narrow. Prefix tuning and gist tokens are
continuous prompt/context compression methods, not per-example source-private
candidate packets. DroidSpeak, C2C, KVComm, and Interlat move dense KV/cache or
hidden-state objects, while this gate transmits no source state vector. SAE and
crosscoder work is the right inspiration for the next learned common-basis
branch, but this gate itself is fixed-policy hardening. TurboQuant and QJL are
quantized-vector/KV comparators, not final task-packet baselines.

## Lay Explanation

We took a different block of `1024` HellaSwag questions, shuffled the four
answer endings for each question, reran the local source model from scratch,
and translated the displayed answer number back to the original answer number.
The tiny final packet usually pointed to the same original candidate after the
shuffle. If we refused to translate it back, or translated it back in the
wrong way, performance collapsed.

## Decision

Promote this as non-overlapping-slice physical candidate-text hardening for the
full hidden fixed-hybrid HellaSwag packet.

Do not promote it as:

- all-24-per-example invariance;
- full-validation physical permutation invariance;
- learned cross-model latent language;
- a cross-family result;
- a native systems speedup.

Next exact gate: implement a decision-supervised sparse common-basis
hidden-innovation packet on this frozen `1024:2048` surface, with source-index,
target-derived, row-shuffle, candidate-roll, atom-ID permutation, and top-atom
knockout controls.
