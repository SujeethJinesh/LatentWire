# LatentWire COLM_v3 Completion Audit

- date: 2026-05-05
- scope: reviewer-readiness check after breadth packaging and Mac-feasible
  experimental kernel scaffolds

## Submission Status

COLM_v3 is complete enough for reviewer circulation after the latest reviewer
overclaim fixes are rebuilt into the PDF, then human copyedited and page-budget
checked. No NVIDIA GPU work is required for the workshop claim as written. The
paper must not claim GPU throughput, HBM, energy, latency, or C2C superiority.

## Current Main Claim

LatentWire provides a content-private, source-state-private packet protocol and
evaluation framework for byte-scale model-to-model candidate hints. It reports
narrow low-byte packet positives on ARC-Challenge and OpenBookQA, explicit
utility-per-byte and systems accounting, and destructive controls that expose
source-choice and target-cache shortcuts. The OpenBookQA row is not claimed as
a statistically supported improvement over same-budget text, and neither row
beats explicit source-index communication.

## Required Assets

| Asset | Status | Artifact |
|---|---|---|
| unified paper draft | integrated | `colm_final/paper/latentwire_colm2026.tex` |
| main result table | integrated | `paper/latentwire_colm_v3_review_packet_20260505.md` |
| destructive-control/claim audit | integrated | `colm_final/paper/latentwire_colm2026.tex`, appendix |
| utility-per-byte/systems table | integrated | review packet systems table |
| related-work/baseline matrix | integrated | `tab:baseline-matrix` |
| benchmark breadth audit | reviewer-pack only | ARC, OpenBookQA, HellaSwag, CommonsenseQA, SciQ |
| latest-model breadth audit | reviewer-pack only | Qwen3.5, Gemma 4, Granite 3.3, OLMo-2 |
| negative/failure boundaries | integrated | limitations plus claim audit |
| artifact manifest | integrated | paper appendix and review packet |
| reproducibility checklist | integrated | paper appendix |
| side systems experiments | scoped papers started | `experimental/*/paper/`, phase2 artifacts |

## Baseline Coverage

The current package includes the baselines needed for a disciplined workshop
claim:

- local destructive controls: target-only, same-byte structured text,
  wrong-row/source-choice controls where available, source-index/rank/score
  boundaries, and cross-family falsification rows;
- systems comparators: dense KV/cache floors, C2C/KVComm/KVCOMM style cache
  transfer boundaries, vLLM/SGLang serving boundaries, and low-bit KV
  compression comparators;
- related-work matrix: dense cache fusion, selective KV sharing, KV
  compression, serving substrates, prompt/prefix compression, and local shortcut
  baselines.

The paper should not imply that all breadth diagnostics are headline benchmark
evidence. HellaSwag, CommonsenseQA, SciQ, and latest-model rows remain reviewer
pack support unless a future gate turns them into controlled headline rows.

## Systems Status

Measured or locally accounted:

- raw packet bytes;
- framed/cacheline/batch-rounded packet estimates;
- utility-per-byte tables;
- source-exposure labels;
- CPU/Mac-local packet accounting and selected local traces.

Estimated or future-only:

- dense KV/cache byte floors;
- C2C/KVComm/KVCOMM/TurboQuant/KIVI/KVQuant boundary comparisons;
- native GPU latency, HBM, energy, occupancy, and throughput;
- side-project Triton kernel performance.

## Experimental Side Projects

All three side projects now have Macbook correctness scaffolds:

| Experiment | Current local status | Next gate |
|---|---|---|
| HybridKernel | weakly alive; pre-GPU threshold model says only profiling can justify the branch after vLLM hybrid layout work | native NVIDIA/vLLM profiling for real boundary conversion/materialization overhead |
| SinkAware | alive as approximate low-rank sink prior; exact static-prior reuse remains killed | GPU prototype comparing exact attention, exact decomposition, and rank-2 approximate sink-logit prediction |
| ThoughtFlow-FP8 | weakened; retained-context NLL proxy ties LongFlow-like and loses to R-KV-like at matched budget | design a sharper hidden/KV saliency policy before any GPU/KV work |

These are not core COLM_v3 evidence today, but each now has a scoped COLM-style
paper shell so positive future data can be accumulated without contaminating
the LatentWire claim.

## Remaining Workshop Work

1. Rebuild the PDF after Reviewer 2 fixes.
2. Human copyedit the PDF.
3. Check table widths and page budget.
4. Ensure abstract, intro, limitations, and claim audit use the same claim
   boundary.
5. Keep side systems material in future work or reviewer-pack notes only.
