# Behavior-Aligned Sparse Resonance Packet Control Plan

Date: 2026-05-04

## Readiness

- Current paper readiness: ICLR is still blocked. COLM-style source-private
  packet evaluation is plausible, but the Sparse Resonance Packet pivot does
  not yet have a positive learned method.
- Current story: low-byte, source-private packets are reproducible and useful
  as systems/privacy artifacts, but current learned hidden/common-basis and
  target-aligned soft-prefix packet gates fail against target-cache,
  candidate-only, atom-shuffle, and source-substitution controls.
- Exact blocker: the next behavior-aligned packet must beat target-only or
  fixed-hybrid, compact candidate/source-choice packets, source-index/rank/score
  codes, target-derived same-byte packets, label/order controls, destructive
  atom controls, and one strict cross-family falsification pair with positive
  paired uncertainty.

## Critique of Current Evidence

The current HellaSwag positive row is a strong systems/privacy row, not a
positive Sparse Resonance method. The strict candidate-only audit shows that
the previous selected packet can be reduced to a 1B raw / 4B framed candidate
ID while preserving the full validation result. That makes candidate-only,
source-index, source-rank, and source-score codes mandatory baselines for any
next method.

The May 4 decision-sparse common-basis gate weakened shallow atom claims. Its
selected sparse packet only improves over packet-only by 0.001953 with negative
CI95 low, while source-shared shuffle, atom-index permutation, and label-
permuted SAE controls match or exceed it. Top-atom knockout gives a small
diagnostic drop, but not enough to show source-specific atom semantics.

The May 4 target-aligned ARC gate also remains negative. Target-derived and
Qwen-substitution controls dominate the matched TinyLlama-to-Qwen packet. Do
not widen shallow target-PCA plus one-step soft-prefix decoding.

## Must-Run Controls

1. Source-choice copying controls.
   Treat candidate-only as the live null. Include explicit source-selected
   candidate ID, source-index, source-rank, top-k rank bucket, quantized
   source-score, and quantized source-margin packets at the same framed bytes.
   Add same-source-choice wrong-row controls: donor packets must come from rows
   with the same source-selected label, same source rank bucket, and similar
   source margin but different content. A method that survives this control is
   likely source-choice copying, not behavior transfer.

2. Target-cache controls.
   Include target-only, slots-only, zero-source, target-derived same-byte
   packet, target-native same-capacity soft slots, target self-resonance, and
   target-model substituted packet. The method must beat the best target-cache
   control, not just the raw target-only baseline.

3. Row-causality controls.
   Include source-row shuffle, wrong-row within the same benchmark, wrong-row
   across benchmark, wrong-row within same source-choice/margin bucket,
   same-byte random packet, and same-byte visible text/code. The matched packet
   should collapse under wrong-row controls while same-byte visible text should
   not match the method.

4. Label/order controls.
   Use physical candidate-text permutation before both source and target
   scoring, then canonical remapping for evaluation. Also run wrong-remap
   collapse, candidate-roll, candidate derangement, global label permutation,
   rowwise label permutation, label-copy baselines, and position-copy baselines.
   For HellaSwag, use balanced permutation schedules or all 24 permutations for
   publishable slices.

5. Packet-content destructive controls.
   Include atom-ID shuffle, coefficient shuffle, sign flip, magnitude-only,
   atom-only, random same-sparsity atoms, top-atom knockout, top-2 atom
   knockout, coefficient donor from same source-choice bucket, and label-
   permuted packet encoder. Sparse Resonance cannot claim interpretable atoms
   if atom identity or coefficient pairing can be destroyed without a paired
   drop.

6. Leakage and provenance controls.
   Materialize frozen row files with split/content hashes, prompt hashes, model
   snapshot IDs, forbidden-field manifests, and exact train/select/eval ranges.
   For OpenBookQA, packet builders must never see answerKey, fact1, humanScore,
   clarity, or worker metadata. For public benchmarks, log exact and fuzzy
   overlap checks and whether any eval row was used during method selection.

7. Source/target complementarity diagnostics.
   Report source-only, target-only, source-and-target agreement, source-only
   correct, target-only correct, both-wrong, target top-2 oracle, source top-2
   oracle, and union oracle. The behavior-aligned gate should be evaluated
   separately on target-wrong/source-useful disagreement buckets and on the
   full frozen slice.

8. Selection and multiplicity controls.
   Keep n8/n16 as scouts only. Predeclare one frozen final slice and one final
   method configuration before publishable evaluation. Report every tried gate,
   all seeds, all excluded slices, and the best required control selected by a
   family-wise max over controls.

9. Same-family and cross-family separation.
   Report same-family and cross-family as separate claims. Same-family success
   does not imply cross-family transfer. A strict cross-family pair must include
   family-substitution controls and must be nonnegative against target-only or
   fixed-hybrid before any ICLR positive-method claim.

## Decision Thresholds

Scout gates:

- n8 is an implementation and confound smoke only. Keep a branch alive only if
  it beats target-only, candidate-only/source-choice, target-derived, same-byte
  text, and the best destructive control by at least 1 example, and at least
  one source-destroying control loses at least 1 example.
- n16 is still scouting. Promote to n64/n128 only if the method beats the best
  source-choice, target-cache, and destructive controls by at least 2 examples,
  has no label/order sanity failure, and its help examples are not all explained
  by source-selected candidate ID.
- Any n8/n16 result can kill a branch if a control is tied or better for a
  structural reason: same-source-choice wrong-row survives, target-derived
  matches, atom shuffle matches, candidate-roll matches, or wrong-remap does not
  collapse.

Publishable slices:

- Minimum: n >= 500 for ARC/OpenBookQA live public benchmark gates; HellaSwag
  should use at least one 1024-row frozen slice and preferably multiple
  non-overlapping 1024-row slices.
- Seeds: at least 5 seeds for randomized packet builders/receivers. All seeds
  must be nonnegative against the best required control; at least 4/5 should
  clear the point-estimate margin.
- Main method pass: paired accuracy delta over the best of target-only,
  fixed-hybrid, and compact candidate-only must be >= +0.02 absolute with
  CI95 low > 0.00.
- Control pass: paired delta over the best required source-choice,
  target-cache, label/order, and destructive control must be >= +0.01 absolute
  with CI95 low > 0.00.
- Permutation pass: canonical-remapped physical candidate permutations should
  stay within 1 percentage point of the unpermuted method row, while wrong
  remap, candidate-roll, and label-copy controls must be below the method with
  positive paired separation.
- Source-use pass: on target-wrong/source-useful disagreement rows, the method
  must show positive paired lift and must collapse under same-source-choice
  wrong-row donors. If not, call it source-choice compression.
- Cross-family pass: at least one strict cross-family pair must be nonnegative
  against target-only/fixed-hybrid and separated from source-family substitution
  controls. If cross-family fails, the ICLR claim remains blocked.
- Utility-per-byte pass: only report favorable utility-per-byte after the
  numerator, paired utility over the best baseline, has CI95 low > 0.00.

## Table and Figure Plan

1. Confound pass/fail matrix.
   Rows: method, candidate-only, source-index, source-score, target-derived,
   same-byte text, same-source-choice wrong-row, atom shuffle, coefficient
   shuffle, candidate-roll, label permutation, family substitution. Columns:
   ARC, OpenBookQA, HellaSwag same-family, HellaSwag cross-family.

2. Source-choice residual table.
   For each slice, show method minus candidate-only, method minus source-rank,
   method minus source-score, same-source-choice wrong-row delta, help/harm
   counts, and CI95 low.

3. Target-cache waterfall.
   Start at target-only/fixed-hybrid, then add slots-only, target-derived,
   target-substituted, matched source packet, row shuffle, atom shuffle, and
   top-atom knockout. This should make target-cache explanations visually
   obvious.

4. Candidate-permutation invariance panel.
   Show original, canonical-remapped permutations, wrong-remap collapse,
   candidate-roll, rowwise label permutation, and label-copy/position-copy
   baselines.

5. Help/harm and oracle headroom panel.
   Use four-way bars for both-correct, source-only correct, target-only
   correct, both-wrong, plus target top-2, source top-2, and union top-2
   oracle lines.

6. Packet destructiveness figure.
   Plot method accuracy and paired deltas after atom-ID shuffle, coefficient
   shuffle, magnitude-only, atom-only, random same-sparsity atoms, top-atom
   knockout, and same-choice coefficient donors.

7. Discovery-to-final ledger figure.
   Show n8/n16 scouts, selected configuration, frozen final slice, all excluded
   branches, seed pass counts, and the family-wise best required control.

8. Utility-per-byte frontier.
   X-axis framed bytes per row on log scale; Y-axis paired utility over the
   best target/fixed/candidate baseline. Mark whether source text, source KV,
   raw hidden states, raw scores, or target-derived packets are exposed.

## Sources Used

- ARC: Clark et al., 2018, https://arxiv.org/abs/1803.05457
- OpenBookQA: Mihaylov et al., 2018, https://aclanthology.org/D18-1260/
- HellaSwag: Zellers et al., 2019, https://aclanthology.org/P19-1472/
- MCQ option-order sensitivity: Pezeshkpour and Hruschka, 2024,
  https://aclanthology.org/2024.findings-naacl.130/
- MCQ option-ID selection bias: Zheng et al., 2024,
  https://arxiv.org/abs/2309.03882
- Benchmark leakage transparency cards: Xu et al., 2024,
  https://arxiv.org/abs/2404.18824
- Black-box contamination via exchangeability/order tests: Oren et al., 2023,
  https://arxiv.org/abs/2310.17623
- Benchmark perturbation and leaderboard sensitivity: Alzahrani et al., 2024,
  https://arxiv.org/abs/2402.01781
- NLP significance-test practice: Dror et al., 2018,
  https://aclanthology.org/P18-1128/
- Paired/bootstrap testing precedent: Koehn, 2004,
  https://aclanthology.org/W04-3250/

## Decision

The single most important next gate is not another shallow atom variant. It is
a behavior-aligned packet evaluated with candidate-only and same-source-choice
wrong-row as first-class nulls. If the packet cannot beat those while also
beating target-derived same-byte controls and physical candidate permutation
controls, the result must be logged as source-choice compression or target-cache
reuse, not Sparse Resonance transfer.
