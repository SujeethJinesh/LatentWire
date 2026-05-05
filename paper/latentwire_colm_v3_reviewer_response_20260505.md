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

## Caveat-Fix Pass

The next review request was to address three caveats experimentally rather than
only listing them in limitations. I ran the Mac-feasible gates and updated the
paper accordingly:

| Caveat | Action taken | New readout |
|---|---|---|
| OBQA had only 5 seeds | Expanded OBQA packet/control seed stability to 20 seeds and rebuilt the source-index audit/aggregate CI under `results/source_private_colm_acceptance_baselines_20260505_obqa20`. | OBQA remains stable versus target-only: packet 0.378, target 0.276. It still does not clear same-budget text (`pkt-text` lower bound -0.002) or source-index (aggregate pkt-src mean -0.0001, CI [-0.0009,+0.0007]). |
| No matched dense-baseline attempt | Added MCQA-to-generation conversion, ran published C2C on same-task ARC/OpenBookQA smoke rows on MPS, and attempted KVComm through local compatibility patches. | C2C runtime path works locally on smoke rows; parsed-letter accuracy is 0.25 on OBQA smoke4 and 0.75 on ARC smoke4. KVComm loads models and source cache but fails at Qwen3 cache/mask compatibility, now recorded as a concrete blocker. |
| Phi-3 was only a falsification row | Added `scripts/build_latentwire_phi3_failure_diagnostic.py` and generated `results/latentwire_phi3_failure_diagnostic_20260505`. | Phi-3 source accuracy is 0.246 vs Qwen 0.346 on ARC test, Phi-3/Qwen choice agreement is 0.289, and the packet follows Phi-3 at 0.997. Failure is source-choice/family boundary at this interface, not decoded-packet corruption. |

These fixes do not create a stronger headline claim. They make the workshop
paper more defensible by replacing unresolved caveats with evidence-backed
boundaries.

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
3. OpenBookQA now has 20 packet/control seeds, but it still does not beat
   same-budget text or source-index under lower-bound gates.
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
