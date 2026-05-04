# Source-Private Strict Benchmark Reviewer-Risk Design

Date: 2026-05-04

## Readiness Status

- Current paper readiness: COLM_v1 is plausible as a scoped source-private
  packet/evaluation paper; COLM_v2/ICLR Sparse Resonance remains not ready.
- Current story: LatentWire has credible strict ARC/OpenBookQA same-family
  packet evidence and strong HellaSwag robustness diagnostics, but the new
  Sparse Resonance/soft-prefix branches are negative under target-only,
  target-derived, atom-shuffle, and source-family substitution controls.
- Exact gap blocking ICLR: a positive source-private method must beat
  target-only or fixed-hybrid, source-index/rank/score, same-byte text,
  target-derived, wrong-row, candidate-roll, atom/coeff shuffles, and at least
  one strict cross-family falsification pair with positive paired uncertainty.

## Fresh Search Takeaways

ARC-Challenge, OpenBookQA, and HellaSwag are appropriate public MCQ surfaces
because they respectively stress science reasoning, open-book plus common
knowledge, and adversarially filtered commonsense continuation. Their public
status also creates leakage and option-order risks. Recent contamination work
argues for explicit benchmark transparency cards, paraphrase/overlap checks,
and accounting for indirect leakage from repeated evaluation. Recent MCQ-bias
work argues that answer option order and option ID priors can change model
scores, so candidate permutation controls are not optional.

## Must-Run Controls

1. Source absence and target cache controls:
   `target_only`, `slots_only`, `zero_source`, `target_derived_packet`, and
   target-native same-capacity soft slots. A method fails if any target-only or
   target-derived row is statistically tied or better.

2. Source-private leakage controls:
   no source text, no source KV, no raw hidden vector, no raw logits/scores, no
   OpenBookQA `answerKey`, `fact1`, `humanScore`, `clarity`, or worker metadata.
   Include split/content hashes and train/dev/test overlap logs.

3. Source identity and row-causality controls:
   source-row shuffle, wrong-row within benchmark, wrong-row across benchmark,
   zero-source packet, same-byte random packet, and same-byte visible text/code.
   A positive row must collapse under wrong-row while remaining above text.

4. Candidate and label controls:
   candidate-roll, candidate derangement, physical candidate-text permutation
   with canonical remapping, wrong-remap collapse, label permutation, and
   label-copy/position-copy baselines. HellaSwag needs at least two
   non-overlapping 1024-row permutation slices before being a robustness claim.

5. Packet-content destructive controls:
   atom-ID shuffle, coefficient shuffle, sign flip, magnitude-only, atom-only,
   top-atom knockout, random same-sparsity atoms, source-rank-only,
   source-score-only, source-index-only, and explicit best source-index code.
   Sparse Resonance cannot claim atom semantics if atom shuffle is tied.

6. Family substitution controls:
   same-family source substitution, cross-family source substitution, stronger
   target-derived packet, and non-source-family packet. ARC/OpenBookQA positives
   must stay separated from HellaSwag Qwen-to-Phi cross-family negatives.

7. Oracle/headroom controls:
   target-or-source oracle, source top1/top2 oracle, target top2 oracle, union
   top2 oracle, and source-unique repair bucket counts. These explain whether
   negative learned gates are method failures or absent headroom.

8. Selection and multiplicity controls:
   frozen train/select/eval split, one-shot final eval, all tried gates logged,
   paired bootstrap over item IDs, paired sign/McNemar discordance counts, and
   a family-wise "best required control" row.

## Decision Thresholds

- Minimum slice size: n >= 500 for a live public benchmark gate; HellaSwag
  should use >=1024-row slices. n8/n32 gates are scouting only.
- Seed stability: at least 5 seeds for randomized packets/receivers; every seed
  must beat target-only, same-byte text, and best destructive control.
- Main pass: paired delta over target-only or fixed-hybrid >= +0.02 absolute
  with CI95 lower bound > 0.00 on each benchmark slice.
- Control pass: paired delta over best required destructive control >= +0.01
  absolute with CI95 lower bound > 0.00.
- Source-index novelty pass: packet must beat explicit source-index/source-rank
  and calibrated source-score quantization, or the claim must be framed as
  source-choice packet transfer rather than latent method.
- Cross-family pass: at least one strict cross-family pair must be nonnegative
  against target-only/fixed-hybrid and beat family-substitution controls. If it
  fails, keep ICLR blocked.
- Utility-per-byte pass: report
  `UPB = mean(correct_method - correct_target_or_fixed_hybrid) / framed_bytes`;
  require positive paired CI on numerator before discussing favorable UPB.
- Saturation rule: if source-index, target-derived, or same-byte text matches
  the packet, method novelty is not positive; move the row to COLM_v2 negative
  map or ablation.

## Table and Figure Design

1. Main pass/fail matrix:
   rows are ARC, OpenBookQA, HellaSwag same-family, HellaSwag cross-family;
   columns are target/fixed baseline, method, source-index, source-score,
   same-byte text, best destructive control, source-family substitution,
   paired delta, CI95 low, seeds passed, bytes.

2. Destructive-control waterfall:
   start from method accuracy, then show deltas to target-only, target-derived,
   row-shuffle, candidate-roll, atom-shuffle, top-atom knockout, same-byte text,
   source-index, and source-family substitution.

3. Utility-per-byte frontier:
   x-axis framed bytes per row on log scale; y-axis paired utility delta over
   target/fixed baseline; annotate source text exposed, source KV exposed, and
   native throughput measured true/false.

4. Leakage audit card:
   dataset split hashes, model snapshots, prompt templates, forbidden fields,
   fit/select/eval ranges, overlap checks, number of gates tried, and whether
   the row was used for method selection.

5. Oracle/headroom panel:
   stacked counts for both correct, source-only correct, target-only correct,
   both wrong; include top2/union oracle lines. This explains negative learned
   receivers without hiding headroom.

6. Candidate-permutation invariance panel:
   canonical consistency, remapped accuracy, unremapped collapse, wrong-remap
   collapse, label-copy baseline, and two-slice repeat.

## Reviewer Objections and Responses

- "This is source-choice transfer, not latent communication."
  Response: include source-index/source-rank/source-score rows in the main
  table. Only claim latent/Sparse Resonance if the new packet beats them; if
  not, claim a source-private evaluation framework and negative map.

- "The packet leaks answer labels or OpenBookQA facts."
  Response: make forbidden fields explicit, hash the materialized rows, log
  zero train/eval content overlap, and run label permutation plus wrong-remap
  collapse.

- "The gain is option-position or label-token bias."
  Response: physical candidate-text permutation with canonical remapping,
  candidate-roll, label-copy, wrong-remap, and all label-position baselines.

- "The gain is target cache, not source communication."
  Response: target-derived, slots-only, zero-source, Qwen-substitution, and
  same-byte controls must be in the main table; failure against these controls
  blocks positive claims.

- "The result is cherry-picked after many gates."
  Response: show the experiment ledger, predeclare frozen final slices, report
  all tried variants, and use paired CIs plus seed pass counts.

- "Utility per byte hides accuracy failure."
  Response: define UPB only after the numerator has positive paired CI. Report
  payload bytes, framed bytes, dense KV floor, quantized KV floor, and native
  throughput status separately.

- "C2C/KVComm/TurboQuant already solve this."
  Response: treat them as dense/quantized KV baselines. LatentWire's only
  defensible differentiator is source-private, low-rate, interpretable packets
  with strict destructive controls; do not claim native speed without native
  vLLM/SGLang measurements.

## What COLM_v2 Can Claim If Gates Stay Negative

If current gates stay negative, COLM_v2 can claim a benchmark-backed
falsification framework for source-private model communication: strict
ARC/OpenBookQA/HellaSwag frozen slices, destructive controls, leakage cards,
paired uncertainty, oracle/headroom diagnostics, and utility-per-byte
accounting. It can also claim that shallow PCA sparse packets, score/top2
packets, and one-step target soft-prefix receivers fail because target-cache,
source-index, atom-shuffle, or family-substitution controls explain the apparent
signal.

It cannot claim an ICLR-ready positive method, cross-family latent transfer,
hidden-state communication, native serving acceleration, or superiority over
dense KV communication unless a future gate clears the thresholds above.
