# COLM_v3 Reviewer Response Memo

- date: 2026-05-05
- scope: paper edits after external reviewer critique

## Decision

Address the reviewer comments by tightening the workshop claim rather than
adding speculative benchmark breadth. The existing source-index audit already
answers the main accept/reject question: the packet transfers a candidate hint,
but does not beat an explicit selected-candidate code on ARC-Challenge or
OpenBookQA.

## Camera-Ready Follow-Up After Second Review

The second review round shifted the remaining work from reframing to concrete
reviewer-facing cleanup. I made the accept-critical edits in the main COLM_v3
paper rather than widening into speculative benchmarks:

| Reviewer issue | Action taken |
|---|---|
| Missing aggregate packet-vs-source-index CI | Added `scripts/summarize_colm_source_index_aggregate_ci.py` and the tracked aggregate readout under `results/source_private_colm_acceptance_baselines_20260502/aggregate_source_index_ci.*`. The paper now reports ARC pkt-src mean -0.0016, 95% CI [-0.0030, -0.0003], and OBQA mean +0.0004, 95% CI [-0.0012, +0.0020]. |
| Source-index not visually prominent | Regenerated `figures/accuracy_overview.pdf` with a source-index bar and a caption stating it is the decisive boundary. |
| Title still sounded too privacy-forward | Changed the title to "No-Text Candidate-Transfer Packets under Destructive Controls." |
| Same-budget text baseline was underdefined | Added an explicit definition: it sends the first B bytes of the source-selected public candidate string, not an answer label or source-index code. |
| Method details too compressed | Added a main-text algorithm table covering source evidence, public chart, DCT projection, packet encoding, receiver decode, validation, and hyperparameters. |
| Qwen2.5-1.5B row looked incomplete | Removed the validation-incomplete Qwen2.5-1.5B diagnostic from the main falsification table and left it out of the workshop claim. |
| Missing score-quantization boundary | Added a validation-only score-quantization diagnostic and explicitly states it does not beat source-label transfer. |

I did not run new model-heavy benchmark breadth in this pass. OBQA K>=10,
higher-entropy tasks, and matched C2C/KVComm comparisons remain archival
blockers, not workshop blockers, after the current paper scope.

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
| HybridKernel | weakly alive; only NVIDIA/vLLM profiling can justify it now | COLM-style PDF generated at `experimental/hybridkernel/paper/hybridkernel_colm2026.pdf`. |
| SinkAware | alive as approximate low-rank sink prior; rank-2 improves held-out output error over position-only | COLM-style PDF generated at `experimental/sinkaware/paper/sinkaware_colm2026.pdf`. |
| ThoughtFlow-FP8 | weakened; retained-context NLL proxy improves over some policies but still loses to the strongest retained-prefix proxy | COLM-style PDF generated at `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`. |

## Next Gate

The COLM_v3 PDF has been rebuilt with these fixes. Send this version for human
review after a page-budget check. Additional experiments should only target the
specific remaining risks: higher-entropy candidate space, calibration or
abstention, or direct native dense-transfer comparisons.
