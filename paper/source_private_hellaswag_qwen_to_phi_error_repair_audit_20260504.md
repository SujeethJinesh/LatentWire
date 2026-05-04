# Qwen-to-Phi Target-Error Repair Audit

Date: 2026-05-04

## Status

Paper readiness: COLM-plausible as a scoped evaluation artifact, but not
ICLR-ready. Estimated ICLR distance remains one positive learned method gate
plus strict same-family/cross-family validation, seed repeats, paired
uncertainty, and native systems rows.

Current story: source-private fixed-byte packets and destructive controls are
solid infrastructure. The strongest live positive direction is no longer
unconditioned score/rank repair; it is target-error-conditioned source
syndrome repair, where the source sends a tiny clue only for cases the target
or fixed packet cannot resolve.

Exact blocker: a learned receiver must convert the source-only repair mass into
held-out overrides that beat fixed Qwen-hybrid packet, Phi target-only,
candidate-only, target-side top-2 controls, row-shuffled source, candidate-roll
source, and source-index/rank/score controls with positive paired CI.

## What Ran

- script:
  `scripts/build_source_private_hellaswag_qwen_to_phi_error_repair_audit.py`
- test:
  `tests/test_build_source_private_hellaswag_qwen_to_phi_error_repair_audit.py`
- artifact:
  `results/source_private_hellaswag_qwen_to_phi_error_repair_audit_20260504_validation1024_2048/`
- related reference memo:
  `references/716_target_error_syndrome_side_information_refs_20260504.md`

This uses cached Qwen packet predictions, cached Phi score simplices, and cached
Qwen source scores for HellaSwag validation `1024:2048`, with the first
`64 + 64` rows per cached slice reserved for fit/select and `768` held-out eval
rows audited. No model loading was required.

## Key Readout

| diagnostic | value |
| --- | ---: |
| held-out eval rows | `768` |
| Phi target-only accuracy | `0.263021` |
| fixed Qwen-hybrid packet accuracy | `0.467448` |
| Qwen candidate-only accuracy | `0.455729` |
| Qwen source-score top-1 accuracy | `0.411458` |
| Qwen source top-2 oracle accuracy | `0.675781` |
| fixed-hybrid or Qwen top-2 oracle accuracy | `0.694010` |
| fixed-hybrid or Phi top-2 oracle accuracy | `0.727865` |
| fixed-hybrid or union top-2 oracle accuracy | `0.845052` |
| fixed-hybrid wrong but Qwen top-2 contains gold | `174` rows |
| fixed-hybrid wrong, Phi top-2 misses, Qwen top-2 contains gold | `90` rows |
| source-unique share of fixed-hybrid errors | `0.220049` |

The target-error branch is alive because `90 / 768` held-out rows are
source-unique repair opportunities: fixed Qwen-hybrid is wrong, Phi's local
top-2 does not contain the gold answer, but Qwen source top-2 does. Those rows
are precisely where a Wyner-Ziv/ECOC-style source syndrome could add something
that target-side uncertainty alone cannot recover.

## Decision

Promote target-error-conditioned source top-2 repair as the next live method
branch. Weaken generic unconditioned source-score/top2 decoders: they have
large oracle headroom but repeatedly fail to decide when to use it.

Do not claim a positive result from this audit. It is an oracle/headroom
surface. The next gate must train a receiver on official-train rows and require
held-out improvements over:

- fixed Qwen-hybrid packet;
- Phi target-only and Phi top-2 target-side oracle diagnostics;
- Qwen candidate-only/source-top1/source-rank/source-score packets;
- source-row shuffle, candidate-roll, code-value permutation, random same-byte,
  target-derived, and label-permuted controls;
- same-byte visible text/code where applicable.

## Lay Explanation

Imagine Phi has already guessed an answer, and Qwen can send only a tiny note.
We checked whether Qwen's tiny note could contain the right answer when Phi is
wrong. On 90 held-out questions, Phi's own top guesses do not include the right
answer, but Qwen's top two guesses do. That means there is real useful source
information left; the hard part is teaching the receiver when to trust it
without peeking at the answer.
