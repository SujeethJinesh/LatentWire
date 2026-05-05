# COLM_v3 Reviewer Response Memo

- date: 2026-05-05
- scope: paper edits after external reviewer critique

## Decision

Address the reviewer comments by tightening the workshop claim rather than
adding speculative benchmark breadth. The existing source-index audit already
answers the main accept/reject question: the packet transfers a candidate hint,
but does not beat an explicit selected-candidate code on ARC-Challenge or
OpenBookQA.

## Reviewer 2 Fixes Integrated

| Reviewer issue | Action taken in `colm_final/paper/latentwire_colm2026.tex` |
|---|---|
| OBQA overclaimed vs same-budget text | Abstract, main results, and uncertainty table now state OBQA is statistically indistinguishable from same-budget text under the paired-bootstrap lower-bound gate. |
| Missing packet-vs-source-index emphasis | Main table and claim-boundary paragraph foreground packet-source lower bounds; appendix now includes per-seed source-index audit rows. |
| Privacy terminology collision | Replaced source-private framing with content-private/source-state-private and explicitly states the packet leaks source choice with 0.995/0.999 match. |
| Slepian-Wolf/Wyner-Ziv overclaim | Intro now says side-information framing is inspirational only and disclaims formal joint distribution, distortion, or converse. |
| Falsification novelty overclaim | Related work now frames destructive controls as an aggregation of established control-task and behavioral-testing methods. |
| C2C/KV systems apples-to-oranges risk | Systems section now separates shape-restricted task hints from general-purpose state transfer and says the byte ratio is message-size only, not function equivalence. |
| Missing recent baselines | Bibliography and related work now include KVCOMM, CIPHER, embedding-space steganography, neural distributed source coding, and control-method precedents. |

## Remaining Paper Risks

1. The main empirical result is useful but narrow: it is candidate-hint transfer,
   not reasoning synthesis beyond the source.
2. Four-way MCQA leaves a low answer-channel ceiling, so reviewers may still ask
   for a higher-entropy task.
3. OpenBookQA has only 5 seeds and should remain a secondary row.
4. Native C2C/KV throughput comparisons remain future work and must not appear
   as a workshop claim.

## Side Systems Project Readout

| Project | Current status | Paper action |
|---|---|---|
| HybridKernel | weakly alive; only NVIDIA/vLLM profiling can justify it now | COLM-style scaffold plus profiler runbook added under `experimental/hybridkernel/`. |
| SinkAware | alive as approximate low-rank sink prior; rank-2 improves held-out output error over position-only | COLM-style shell plus per-head softmax/output probe added under `experimental/sinkaware/`. |
| ThoughtFlow-FP8 | weakened; retained-context NLL proxy ties LongFlow-like and loses to R-KV-like | COLM-style shell plus perplexity proxy gate added under `experimental/thoughtflow_fp8/`. |

## Next Gate

The COLM_v3 PDF has been rebuilt with these fixes. Send this version for human
review after a page-budget check. Additional experiments should only target the
specific remaining risks: higher-entropy candidate space, calibration or
abstention, or direct native dense-transfer comparisons.
