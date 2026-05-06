# ThoughtFlow-FP8 Progress

## Status

- Phase 0: partial quick pass.
- Phase 1: quick forensics pass complete.
- Phase 2: synthetic retention simulation complete; real-trace text proxy gate
  weakened the branch; hidden/KV saliency telemetry and the stopped
  anchor/recent/phase/math sparse-cache family were mixed/negative, but the
  pre-registered `rdu_topk` successor revived the branch on the frozen
  sparse-cache quality gate
- Phase 4: anchor/phase retention reference plus Triton interpreter correctness
  scaffold added, but not phase-complete
- Current viability: the stopped current policy family remains ruled out.
  `rdu_topk` cleared the first Mac-local frozen sparse-cache gate and reproduced
  exactly on the same deterministic slice, but it is no longer promoted as a
  positive method. The first alternate surface failed same-family separation,
  and the larger independent saved-trace surface failed cross-family
  separation.
- Current risk: high field crowding; ThinKV already occupies much of the
  thought-adaptive quantization/eviction space, and DeepSeek V4 raises the
  production compressed-attention systems bar.
- Current pre-registration:
  `phase2/preregister_recurrence_distance_utility_20260506.md` defined one
  recurrence-distance utility signal and the one-shot frozen evaluation has now
  been run. No post-result tuning is allowed on this trace set.

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

## Phase 2 Result

`phase2/phase_eviction_analysis.md` and `.json` record a synthetic Mac-local
simulation. At the same 0.368 keep rate, ThoughtFlow preserves phase markers
with 1.000 recall, while LongFlow-like, ThinKV-like, and R-KV-like proxies retain
0.143, 0.286, and 0.286 respectively. Anchor recall stays 1.000.

Status: **ALIVE, but not reviewer-pack-ready**. The result is synthetic policy
evidence, not accuracy evidence and not a GPU systems result.

## Next Gate

The real generated-trace proxy in
`phase2/real_trace_retention_analysis.md` did not show a protected-token
advantage over the LongFlow-like importance proxy. At a matched 0.211 keep
rate, ThoughtFlow and LongFlow-like both reached 0.941 phase recall, while
ThoughtFlow did not improve math-state recall.

Status: **WEAKENED**. Do not prepare a reviewer pack from this branch yet. The
next gate must use real KV/cache telemetry or a sharper hidden-state phase
signal; text-marker heuristics are not enough.

`phase2/real_trace_retention_sweep.md` confirms that weakness across keep
fractions 0.10 to 0.35: ThoughtFlow never beats the strongest proxy on phase
recall and sometimes trails on math-state recall.

The first version of `phase2/hidden_saliency_retention_probe.md` gave a mixed
result on distilgpt2 attention-received saliency: protected phase markers beat
the pure attention-received proxy, but still tied the strongest importance
proxy. This was not enough for a reviewer pack or GPU work.

Current status: **MIXED/WEAKENED**. The only pre-GPU route left is a real hidden
or KV saliency policy that beats the strongest importance proxy and shows a
quality/perplexity benefit under actual cache dropping.

`phase2/perplexity_impact_proxy.md` now includes a stronger successor,
ThoughtFlow-saliency-recent. It protects anchors, reserves half the retained
budget for recent tokens, and lets phase markers compete with math-state and
high-importance reasoning tokens. On 24 distilgpt2 traces at 0.20 keep
fraction, it improves the best ThoughtFlow-family NLL from 3.562 to 3.434,
nearly tying but still losing to the R-KV-like retained-prefix proxy at 3.419.

`phase2/policy_sweep.md` runs the next ablation with matched budgets: select a
small recency/phase/math-state policy on 12 train traces and report on 12
held-out traces. The train-selected policy ties R-KV-like on held-out traces
with NLL 3.480 versus 3.482, margin +0.001 in favor of the ThoughtFlow-family
policy. This is a tie-range result, not a robust win.

Status: **MIXED, not revived as a positive method**. The next useful gate was
real hidden/KV saliency telemetry; GPU sparse-KV work should wait until the
policy beats R-KV-like by a nontrivial margin on held-out traces.

`phase2/hidden_saliency_retention_probe.md` now uses real distilgpt2 telemetry:
attention-received mass, final-hidden norms, key norms, value norms, and
combined KV norms. On the same 24-trace, 0.20 keep-fraction slice, the best
ThoughtFlow-family policy is the original marker-protecting `thoughtflow` row.
It beats the strongest real-saliency proxy (`value_norm_topk`) on phase recall
by +0.508 paired mean, but the math-state margin is only +0.073 with a 95%
normal CI of [-0.078, +0.223]. It also ties the local LongFlow-like importance
proxy on phase and math-state recall because both policies preserve the same
high-importance marker classes.

Status: **MIXED/NOT REVIVED**. Promoted: real hidden/KV telemetry as a useful
diagnostic. Weakened: the claim that phase-marker preservation alone is a
positive method. Saturated: synthetic marker-retention and text-prefix-only
policy tuning. The next exact gate is actual cache-dropping or sparse-KV quality
validation, not more marker-recall-only sweeps.

`phase2/kv_drop_quality_probe.md` now runs the closest Mac-local sparse-cache
quality gate: process the full prefix once, prune the returned KV cache by each
policy, then score the continuation from the sparse cache on CPU. On 24
distilgpt2 traces at a 0.20 keep fraction, the train-fixed ThoughtFlow sweep
best has NLL `3.432` versus R-KV-like `3.435`, with paired delta `-0.003` and
95% CI `[-0.037, +0.034]`. That is a tie-range result, not a win.

Status: **MIXED/NOT REVIVED**. This probe strengthens the negative/mixed
conclusion because it uses actual cache dropping rather than retained text, but
it still fails the pre-registered 0.03 NLL margin over R-KV-like.

The 2026-05-05 rerun of the same CPU sparse-cache quality gate changes the
best mean row but not the decision. ThoughtFlow-saliency-recent now has the
lowest compressed-cache NLL (`3.372`) versus ThinKV-like (`3.389`) and R-KV-like
(`3.438`). The margin over the strongest non-ThoughtFlow row is only `0.017`
NLL, below the pre-registered `0.03` promotion threshold, and the paired delta
versus R-KV-like is `-0.067` with 95% CI `[-0.151, +0.011]`. The branch remains
**MIXED/NOT REVIVED** until a train-fixed policy clears the margin with paired
uncertainty.

The 2026-05-06 bounded train-fixed sparse-cache sweep tested 24 nearby
anchor/recent/phase/math configurations on the even-index train traces and
reported the train-selected row on odd-index held-out traces. It selected
`tf_sparse_r0.55_p0.05_m0.12_a2`. On held-out traces that row reaches NLL
`3.340` versus ThinKV-like `3.385` and R-KV-like `3.420`; the mean margin over
the strongest non-ThoughtFlow held-out baseline is `+0.045`. Paired uncertainty
is still not enough: the paired delta versus R-KV-like is `-0.080` with 95% CI
`[-0.152, -0.014]`, but the paired delta versus ThinKV-like is `-0.045` with
95% CI `[-0.226, +0.182]`. The fixed incumbent
`thoughtflow_saliency_recent` has the best held-out mean NLL (`3.304`), but its
paired CIs also cross zero versus both R-KV-like and ThinKV-like.

Status: **MIXED/PROMISING BUT NOT REVIVED**. Promoted: the sparse-cache
candidate now has a clean held-out mean margin and a paired win over R-KV-like.
Still blocking: uncertainty versus ThinKV-like, the strongest non-ThoughtFlow
held-out baseline. The next exact gate is a larger frozen sparse-cache slice
with the two train-fixed candidates only, no further policy tuning.

`phase2/frozen_sparse_cache_probe.md` executes that larger no-retuning gate. It
freezes only `thoughtflow_saliency_recent` and
`tf_sparse_r0.55_p0.05_m0.12_a2`, then scores 74 saved traces with actual CPU
cache pruning. This weakens the branch: ThinKV-like is the best compressed row
with NLL `3.900`, followed by `tf_sparse_r0.55_p0.05_m0.12_a2` at `3.908`,
`thoughtflow_saliency_recent` at `3.920`, and R-KV-like at `3.939`. The frozen
sparse candidate's paired delta is `-0.031` versus R-KV-like with 95% CI
`[-0.078, +0.020]`, but `+0.008` versus ThinKV-like with 95% CI
`[-0.060, +0.085]`.

Status: **WEAKENED/NOT REVIVED**. Ruled out for now: the small-split
train-selected sparse policy as a robust positive method on the available
Mac-local distilgpt2 sparse-cache slice. Still alive only as diagnostic
infrastructure. The next exact gate is not further policy tuning on these
traces; it is either a pre-registered new utility signal evaluated once on this
frozen probe, or a stop/pivot decision for ThoughtFlow-FP8.

`phase2/stop_pivot_decision_20260506.md` records that decision. Further
anchor/recent/phase/math weight tuning on the current saved traces is stopped.
The current policy family is ruled out as a robust positive method on this
Mac-local sparse-cache surface. A future attempt must first pre-register one
genuinely new utility signal and then evaluate it once on the frozen sparse-cache
probe; otherwise ThoughtFlow-FP8 should remain a negative/mixed workshop
artifact.

`phase2/preregister_recurrence_distance_utility_20260506.md` then registered
exactly one successor signal, `rdu_topk`, based on recurrence-distance utility
from prefix self-attention lag buckets. The 2026-05-06 one-shot frozen
sparse-cache run revived the branch on the current Mac-local distilgpt2
surface. At the same 0.20 keep fraction and 74 scored traces, `rdu_topk` reaches
NLL `3.779`, beating ThinKV-like (`3.900`) by `0.121` NLL and R-KV-like
(`3.939`) by `0.160` NLL. Paired uncertainty clears the pre-registered rule:
delta versus R-KV-like is `-0.160` with 95% CI `[-0.264, -0.050]`; delta versus
ThinKV-like is `-0.121` with 95% CI `[-0.211, -0.037]`.

Status: **REVIVED ON THE FIRST FROZEN SUCCESSOR GATE, NOT ICLR-READY**. Alive:
recurrence-distance utility as a training-free sparse-cache signal. Still
stopped: all anchor/recent/phase/math tuning on this trace set. Highest-priority
next gate: reproduce `rdu_topk` on a larger or seed-repeated frozen slice with
strict same-family versus cross-family separation and oracle/headroom
diagnostics before widening to competitor or long-context benchmarks.

`phase2/rdu_robustness_diagnostic.md` adds the smallest cached robustness
artifact around that 0.20 run without retuning or changing the scoring rule.
On deterministic half-size splits, `rdu_topk` remains the best compressed row
and keeps positive mean margins of at least 0.03 NLL versus both R-KV-like and
ThinKV-like in all four partitions. Only 2/4 split partitions clear both paired
CI highs below zero, so this supports promotion on the existing frozen gate but
does not replace a true larger or seed-repeated reproduction.

`phase2/rdu_no_retune_reproduction_check.md` adds the cheapest measured
reproduction-style check on current Mac hardware. It reruns the frozen 74-trace
distilgpt2 sparse-cache probe with the same pre-registered `rdu_topk` rule,
writes a separate measured artifact, and compares it against the cached
promoted gate without overwriting the baseline. The measured rerun reproduces
the cached result exactly on this deterministic local stack: `rdu_topk` NLL
3.779, margin +0.160 versus R-KV-like and +0.121 versus ThinKV-like, paired
deltas -0.160 [-0.264,-0.050] and -0.121 [-0.211,-0.037], with zero measured
minus cached NLL drift for all policies. Strict separation is preserved:
`rdu_topk` beats the stopped ThoughtFlow rows by +0.141 and +0.129 NLL and the
cross-family baselines by +0.160 (R-KV-like), +0.121 (ThinKV-like), and +0.379
(LongFlow-like). Oracle/headroom remains material: a per-trace compressed
oracle reaches NLL 3.634, leaving `rdu_topk` 0.145 NLL above that oracle and
0.931 NLL above full cache, with a 0.419 oracle-hit rate.

Status: **REPRODUCED LOCALLY, STILL NOT ICLR-READY**. This is a measured
same-slice no-retuning reproduction, not a larger or independently seeded
reproduction. The exact next gate is a larger or seed-repeated frozen slice
with the same measured-vs-cached labeling, strict same-family/cross-family
reporting, and oracle/headroom diagnostics.

`phase2/rdu_alt_surface_reproduction_check.md` adds the first measured
alternate-surface check without retuning: same `rdu_topk` rule and 0.20 keep
fraction, but longer prefix/continuation settings (`max_length=112`,
`continuation_tokens=32`) against the cached 74-trace promoted gate. This is
not a clean reproduction. `rdu_topk` still clears the cross-family baselines:
NLL `3.594` versus R-KV-like `3.681` and ThinKV-like `3.851`, with paired
deltas `-0.087` 95% CI `[-0.139, -0.028]` and `-0.256` 95% CI
`[-0.465, -0.086]`. But strict same-family separation fails: the stopped
`tf_sparse_r0.55_p0.05_m0.12_a2` row reaches NLL `3.588`, beating `rdu_topk`
by `0.006`. Oracle/headroom remains material: per-trace compressed oracle NLL
is `3.460`, `rdu_topk` is `0.135` above that oracle and `0.847` above full
cache, and the oracle-hit rate is `0.438`.

Status: **WEAKENED, NOT REPRODUCED ON THE FIRST ALTERNATE SURFACE**. Promoted:
the recurrence-distance signal still beats cross-family local proxies under a
changed scoring surface. Weakened: `rdu_topk` is not yet separated from the
stopped same-family sparse row. Highest-priority next gate: do not retune;
run an independently seeded/larger frozen slice and require both cross-family
and same-family separation, paired uncertainty, and oracle/headroom before
claiming a positive method.

`phase2/rdu_independent_trace_reproduction_check.md` runs that larger
no-retuning saved-trace surface. It keeps the same `rdu_topk` rule and 0.20
keep fraction, changes only the trace inputs to the 96-row
`surface_scout_qwen25math_qwen3_svamp32_chat_20260426` source/text/target
surface, and scores the 89 traces that survive token-length filtering. This
does not reproduce the positive gate. R-KV-like is the best compressed row with
NLL `3.981`, while `rdu_topk` reaches `4.014`; the paired delta versus R-KV-like
is `+0.032` with 95% CI `[-0.071, +0.137]`. ThinKV-like remains only `0.030`
NLL worse than `rdu_topk`, with paired CI crossing zero (`-0.030`
`[-0.152, +0.085]`). Same-family stopped rows are worse than `rdu_topk`, but
only narrowly: `thoughtflow_saliency_recent` by `+0.006` and
`tf_sparse_r0.55_p0.05_m0.12_a2` by `+0.010`. Oracle/headroom remains material:
the compressed per-trace oracle reaches NLL `3.754`, leaving `rdu_topk` `0.260`
above compressed oracle and `1.083` above full cache, with `0.348` oracle-hit
rate. A coarse failure decomposition now shows the largest regression on
long-prefix/high-RDU-density rows: `rdu_topk - R-KV-like` is `+0.213` NLL there
versus `-0.049` on short-prefix/low-density rows. When R-KV-like is the
per-trace compressed oracle, `rdu_topk` is `+0.498` NLL worse; when the stopped
sparse ThoughtFlow row is oracle, `rdu_topk` is `+0.174` worse.

Status: **NOT REPRODUCED / DEMOTED TO DIAGNOSTIC**. Ruled out: claiming
`rdu_topk` as a robust positive method on the available Mac-local sparse-cache
surfaces. Still alive: the sparse-cache falsification harness and
oracle/headroom reporting. Highest-priority next gate: do not retune
`rdu_topk`; either stop/pivot ThoughtFlow-FP8 or pre-register a genuinely new
utility signal and evaluate it once on a fresh/larger frozen surface.

## Macbook Kernel Correctness Scaffold

Added an anchor/phase-protected int8 quantization primitive:

- CPU reference: `phase2/reference/anchor_phase_quant.py`
- CPU reference test: `phase2/tests/test_anchor_phase_quant_reference.py`
- Triton interpreter wrapper: `phase4/kernel/anchor_phase_quant_triton.py`
- Triton interpreter test: `phase4/tests/test_anchor_phase_quant_triton_interpret.py`

Run locally:

```bash
./venv_arm64/bin/python -m pytest experimental/thoughtflow_fp8/phase2/tests
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest experimental/thoughtflow_fp8/phase4/tests -rs
```

Current Mac status: CPU reference and Triton interpreter tests pass under the
repo-local `triton-cpu` source install with `TRITON_INTERPRET=1` and
`TRITON_CPU_BACKEND=1`. This validates the anchor/phase retention and int8
quantization kernel logic against the CPU reference only; it is not CUDA, FP8,
latency, throughput, or Blackwell evidence.

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
- 2026-05-05: Added real hidden/KV saliency telemetry to the Phase 2 probe and
  refreshed the COLM workshop shell. Result: mixed/not revived. ThoughtFlow
  beats `value_norm_topk` on phase recall but has uncertain math-state margin
  and still ties the local LongFlow-like importance proxy. Next gate is
  cache-dropping or sparse-KV quality validation for a train-fixed successor.
- 2026-05-05: Added CPU sparse-cache quality probe. Result: mixed/not revived.
  Best ThoughtFlow-family policy ties R-KV-like on continuation NLL with paired
  CI crossing zero; no GPU/FP8 performance claim is allowed.
- 2026-05-05: Reran the CPU sparse-cache quality probe and refreshed the paper.
  ThoughtFlow-saliency-recent is now the best compressed row in mean NLL, but
  the margin is below the promotion threshold and the paired interval still
  crosses zero. Decision remains mixed/not revived.
- 2026-05-06: Added a bounded train-fixed sparse-cache sweep. The selected
  sparse policy clears the held-out mean margin and paired R-KV-like comparison
  but not paired uncertainty versus ThinKV-like. Decision remains mixed/not
  revived; freeze candidates before any larger slice.
- 2026-05-06: Ran the larger frozen sparse-cache slice with no retuning.
  ThinKV-like beats both frozen ThoughtFlow candidates in mean NLL on 74 traces.
  Decision weakens to not revived; do not tune more on this trace set.
- 2026-05-06: Added the stop/pivot decision artifact. Current policy-family
  tuning is stopped on the available saved traces; any revival must be
  pre-registered as one new utility signal before a single frozen evaluation.
- 2026-05-06: Added a docs-only pre-registration for `rdu_topk`, a single
  recurrence-distance utility signal based on delayed prefix self-attention
  reuse. No code was changed and no frozen probe was run.
- 2026-05-06: Implemented the pre-registered `rdu_topk` signal in the frozen
  sparse-cache probe and ran the one allowed frozen evaluation. Result: revived
  on this Mac-local decision surface. `rdu_topk` NLL is 3.779 versus ThinKV-like
  3.900 and R-KV-like 3.939, with paired CIs below zero against both baselines.
  Do not tune this signal on the same saved traces.
- 2026-05-06: Added a cached deterministic split/paired diagnostic for the
  promoted `rdu_topk` row. All four half-size partitions keep positive mean
  margins versus R-KV-like and ThinKV-like, but only 2/4 partitions clear both
  paired CI highs below zero. `rdu_topk` remains promoted on the current frozen
  gate; the next gate remains real reproduction without retuning.
- 2026-05-06: Added and ran the measured no-retuning reproduction check for
  `rdu_topk` on the same 74-trace frozen sparse-cache surface. The measured
  rerun exactly matches the cached promoted gate on this deterministic Mac
  stack, preserves same-family and cross-family margins, and adds
  oracle/headroom diagnostics. This strengthens local reproduction status but
  does not replace a larger or seed-repeated frozen slice.
- 2026-05-06: Added and ran the measured alternate-surface no-retuning check
  for `rdu_topk` with `max_length=112` and `continuation_tokens=32`. Result:
  weakened/not reproduced as a strict positive-method gate. `rdu_topk` still
  beats R-KV-like and ThinKV-like with paired CIs below zero, but the stopped
  same-family sparse row beats it by 0.006 NLL, so same-family separation fails.
- 2026-05-06: Added and ran the independent saved-trace no-retuning check for
  `rdu_topk` on the larger chat SVAMP surface. Result: not reproduced. The
  96-row input surface yields 89 scored traces; R-KV-like is best compressed
  (NLL 3.981) and beats `rdu_topk` (NLL 4.014), so cross-family separation
  fails. Same-family stopped rows are worse than `rdu_topk` but by only
  +0.006/+0.010 NLL. Decision: demote `rdu_topk` to diagnostic; no GPU or paper
  claim without a new pre-registered signal on a fresh/larger surface.
- 2026-05-06: Installed the experimental `triton-cpu` backend from source into
  `./venv_arm64` and ran the Phase 4 interpreter gate. ThoughtFlow-FP8 Phase 4
  tests pass under `TRITON_INTERPRET=1`; the full owned Phase 0--4 side-project
  suite passes (`103 passed, 2 warnings`). This closes Mac kernel-correctness
  blocking, but the branch is still weakened by the alternate-surface result.

## 2026-05-06 Paper Claim Cleanup

Updated the workshop draft to remove the misleading implication that a live
positive successor remains. `rdu_topk` is now described as a demoted diagnostic,
not a live method branch. The title drops the headline FP8 claim; the abstract
and scope section state that current artifacts validate CPU sparse-cache scoring
and an int8/Triton-interpreter reference primitive only, with no real FP8,
CUDA, latency, throughput, or Blackwell result.

Decision: **LOCAL METHOD EVIDENCE SATURATED / STOP OR PIVOT**. The current
Mac-local artifacts are useful as a falsification ladder, not as a positive
method. A revival requires one genuinely new pre-registered utility signal and
a one-shot fresh/larger sparse-cache gate; no GPU work should be spent on the
current `rdu_topk` branch.
