# ThoughtFlow-FP8 Progress

## Status

- Phase 0: partial quick pass.
- Phase 1: quick forensics pass complete.
- Current viability: pivot/proceed only with narrowed framing.
- Current risk: high field crowding; ThinKV already occupies much of the
  thought-adaptive quantization/eviction space, and DeepSeek V4 raises the
  production compressed-attention systems bar.

## Deliverables

### Phase 0

- Created `experimental/thoughtflow_fp8/.venv` (`Python 3.9.13`).
- `phase0/setup_complete.md`: present.
- Full repo/dataset cloning: not done in this quick Mac-only pass.

### Phase 1

- `phase1/lit_review.md`: present.
- `phase1/competitive_matrix.md`: present.
- `phase1/longflow_failure_hypothesis.md`: present.
- `phase1/v4_differentiation.md`: present.

## Next Gate

Do not proceed to kernel implementation yet. The next exact gate is a Mac-only
Phase 2 trace/telemetry simulation that tests whether anchor/fair-span/phase
transition protection preserves a concrete token class that LongFlow-like,
ThinKV-like, R-KV-like, and sink+recent baselines would drop.

Proceed condition: Phase 2 must produce keep-rate/recurrence evidence for a
specific protected-token class and a policy definition that is portable to
existing models without retraining. Otherwise pivot to a critique/re-evaluation
paper or kill the project.

## Log

- Initial scaffold created. No setup, downloads, literature audit, experiments,
  or tests have been verified.
- 2026-05-05: Quick Phase 0/1 forensics completed without SSH, global installs,
  GPU, large model downloads, or edits outside `experimental/thoughtflow_fp8/`.
  Primary sources found for LongFlow, Pitfalls, DeepSeek V4/SGLang, ThinKV,
  R-KV/R-KVHash, RaaS, LazyEviction, ForesightKV, and PM-KVQ. LongFlow official
  reviews were accessible through the OpenReview API and identify concrete
  weaknesses: no production E2E speedup, weak Pareto evidence versus R-KV,
  numerical/approximation concerns, and limited efficiency scaling evidence.
  Recommendation: pivot/proceed with a narrowed retrofit + bias-controlled
  retention framing; do not proceed as a generic LongFlow+FP8+phase kernel.
