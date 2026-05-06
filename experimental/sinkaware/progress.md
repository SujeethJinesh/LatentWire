# SinkAware Progress

## Status

- Phase 0 setup: partial Mac-only source-audit setup
- Phase 1 literature and code audit: quick-kill audit recorded
- Phase 2: exact static sink-prior gate failed; approximate low-rank revival
  and per-head softmax/output gates completed with a new paired layer-head
  caveat; simple head selection failed; all-rank2 split repeats and a bounded
  length/sink sweep passed weakly; trace-level frozen split repeat passed
  as bounded evidence; a 48-trace repeated held-out/model-family gate and a
  64/96-token cross-family length stability gate passed on distilgpt2 plus
  OPT-125M but are still not promotion evidence
- Phase 4: fixed sink-token decomposition reference plus Triton interpreter
  correctness scaffold added, but not phase-complete
- Last updated: 2026-05-06

## Phase 0 Checklist

- [x] Create `experimental/sinkaware/.venv` (`Python 3.9.13`)
- [ ] Install `requirements.txt`
- [x] Create local ignored directories for external repos and artifacts
- [x] Record partial setup verification in `phase0/setup_partial.md`
- [ ] Record full setup verification in `phase0/setup_complete.md`

Phase 0 is not complete until all checklist items are verified locally and the
deliverable exists.

## Phase 1 Checklist

- [x] Audit FlashInfer sink/static-mask handling
- [x] Audit FlashAttention sink/static-mask handling
- [x] Audit StreamingLLM sink handling
- [x] Audit DeepSeek DSA / FlashMLA handling of early positions
- [x] Audit NSA / block-sparse attention handling of early positions
- [ ] Audit BLASST and Block-Sparse Flash Attention if public code is available
- [x] Audit GPT-OSS reference attention pattern handling
- [x] Record file paths, line numbers, and comparison table in `phase1/lit_review.md`

Phase 1 is not complete until the audit is source-backed and the kill criterion
has been explicitly checked.

## Current Assessment

Quick-kill criterion was not triggered for fixed-position BOS/sink KV tokens:
the audited sources did not show an existing `output += sink_bias_precomputed`
path that skips fixed sink-token score computation.

Main risk is now sharper: FlashInfer, FlashMLA, and GPT-OSS already implement
learned/per-head attention sink terms in the softmax denominator. Broad
"sink-aware attention kernel" novelty is therefore occupied. The only remaining
wedge is fixed early-position/BOS K/V decomposition or precomputation.

## Phase 2 Result

`phase2/decomposition_decision.md` records the decision: **KILL as an exact
static sink-prior kernel**. The counterexample test shows fixed sink K/V cannot
be reused exactly while skipping per-query `QK_sink`, because the sink softmax
logits remain query-dependent.

What remains possible is a pivot to approximate/learned/low-rank sink priors or
a small fused path that still computes `QK_sink`. The original exact static-prior
claim should not proceed to GPU work.

## Approximate Revival Gate

`phase2/sink_predictability_probe.md` checks the weaker approximate question:
can fixed-sink logits be predicted from a tiny low-rank query representation
under favorable query geometry?

Result: **REVIVE only as approximate low-rank/clustered query prior**. Static
R2 remains near zero across synthetic cases, so the exact static-prior branch is
still dead. Rank-4 query features recover low-rank synthetic sink logits
(`R2=0.999`), and rank-8 features recover clustered synthetic sink logits
(`R2=0.976`), while random queries remain poor (`rank8 R2=0.102`).

Next gate: use real Q/K tensors or attention telemetry. Synthetic geometry is
only a reason to keep the approximate branch alive, not a reviewer-pack result.

`phase2/real_query_sink_probe.md` and
`phase2/real_qk_sink_logit_probe.md` then move the branch to real
distilgpt2 traces. Hidden-query features predict attention sink mass and
Q/K sink logits better than position-only structure. The QK-logit probe reaches
rank-8 hidden+position `R2=0.712` versus position-only `R2=0.153`.

`phase2/qk_sink_cost_model.md` keeps the systems claim narrow: rank-2 is the
only current compromise that has nontrivial QK-logit predictability while
staying below exact four-sink QK cost (`0.531x` estimated multiply-adds,
`R2=0.420`). Rank-8 is more accurate but too expensive under this simple model.

`phase2/real_qk_sink_softmax_output_probe.md` runs the next Mac-local quality
gate. It keeps all non-sink QK scores exact and replaces only fixed sink-token
logits with per-head predictors on held-out distilgpt2 trace tokens. Mean
across layers still favors rank-2 over position-only (`output rel-L2=0.141`
versus `0.170`, `sink-mass MAE=0.055` versus `0.076`). The new paired
layer-head readout is weaker: rank-2's output rel-L2 improvement over
position-only is `+0.0297 +/- 0.0378` across 72 layer-head cells, with only
20/72 output wins. Rank-8 is more accurate (`output rel-L2=0.107`) but remains
too expensive under the current simple cost model.

Current status before the stability gates: **WEAKLY ALIVE for a narrow
GPU/interpreter gate as an approximate low-rank SinkAware branch**, not as
exact static-prior reuse. The rank-2 aggregate improvement was real in the
saved probe, but per-head wins were concentrated enough that repeatability or a
stability mechanism had to be checked before any strong paper claim. Native
speed claims still require NVIDIA hardware.

## Head-Selective Rank-2 Gate

`phase2/head_selective_sink_gate.md` tested the obvious Mac-local rescue for
the fragile per-head result: fit predictors on a train split, select rank-2
heads on validation when rank-2 beats position-only on output rel-L2, then
evaluate the mixed policy on a held-out split. The rule selected 19/72 heads
but failed held-out: validation-selected rank-2 had output rel-L2 `0.2035`,
worse than position-only `0.1724` and all-rank2 `0.1419`.

Decision: simple validation head selection is **weakened/ruled out** as a
pre-GPU rescue. The approximate branch remains alive only as all-rank2 plus a
future stability mechanism; no paper should claim head-selective robustness.

## All-Rank2 Split/Seed Stability Gate

`phase2/rank2_split_stability_gate.md` tested the next Mac-feasible question
after the head-selector failure: does the all-head rank-2 row still beat
position-only when the token-level train/test split is randomized across seeds?
On the same 48 distilgpt2 traces with `max_length=96`, `sink_tokens=4`, and
three split seeds, all-rank2 beat position-only on every split. Mean output
rel-L2 improvement was `+0.0368 +/- 0.0006`; sink-mass MAE improvement was
`+0.0203 +/- 0.0003`; attention-L1 improvement was `+0.0435 +/- 0.0004`.

Decision: all-rank2 is **alive but still weak** for a correctness/interpreter
gate. The result sharpens the branch because the aggregate row is repeatable
under token split seeds, but it does not solve per-head fragility: the
layer-head output win rate remains only `0.282 +/- 0.024`. The next exact
pre-GPU gate should be a broader sequence-length/sink-token sweep or a
trace-level frozen split repeat, not another simple validation head selector.

## All-Rank2 Length/Sink Sweep

`phase2/rank2_length_sink_sweep_gate.md` ran the smallest Mac-feasible
sequence/sink sweep: `max_length in {64, 96}`, `sink_tokens in {2, 4}`, three
token split seeds per configuration, and 48 distilgpt2 traces. All four
configurations kept rank-2 positive against position-only on every seed. Across
configurations, output rel-L2 improvement averaged `+0.0366 +/- 0.0024`; the
minimum configuration improvement was `+0.0342`; layer-head output win rate was
still low at `0.286 +/- 0.010`.

Decision: all-head rank-2 is **alive but bounded** for interpreter/GPU
correctness work. This sweep is stronger than the single-config split repeat,
but it still does not establish end-to-end quality, per-head robustness, or
speed. The next exact pre-GPU gate should be a trace-level frozen split repeat
or a larger frozen slice; head selection should stay ruled out unless a
different stability signal appears.

## All-Rank2 Trace-Level Frozen Split Gate

`phase2/rank2_trace_frozen_split_gate.md` now records the larger Mac-feasible
trace-level repeat: 48 distilgpt2 traces, `max_length=96`, `sink_tokens=4`,
and three deterministic frozen trace splits. Each split trained on 32 whole
traces and evaluated on 16 held-out traces, so no text trace appeared in both
train and held-out sets.

All three splits kept all-head rank-2 positive against position-only. Across
trace splits, output rel-L2 improvement averaged `+0.0379 +/- 0.0014`;
sink-mass MAE improvement averaged `+0.0220 +/- 0.0008`; attention-L1
improvement averaged `+0.0461 +/- 0.0014`; the minimum split improvement was
`+0.0367`.

Decision: all-head rank-2 is **alive but still bounded** after the trace-level
frozen repeat. This strengthens repeatability over the prior 24-trace
whole-trace gate, but it remains Mac-local attention-output drift evidence.
The layer-head output win rate is still low at `0.278 +/- 0.016`, so no paper
should claim per-head robustness or downstream quality yet. The next exact gate
is the Triton interpreter correctness gate for the approximate operator or a
strict cross-family falsification pair before any native NVIDIA timing claim.

## Macbook Kernel Correctness Scaffold

Added a scalar fixed sink-token decomposition primitive:

- CPU reference: `phase2/reference/sink_decomposition.py`
- CPU reference test: `phase2/tests/test_sink_decomposition_reference.py`
- Triton interpreter wrapper: `phase4/kernel/sink_decomposition_triton.py`
- Triton interpreter test: `phase4/tests/test_sink_decomposition_triton_interpret.py`

Run locally:

```bash
./venv_arm64/bin/python -m pytest experimental/sinkaware/phase2/tests
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest experimental/sinkaware/phase4/tests -rs
```

Current Mac status: CPU reference and Triton interpreter execution tests pass
under the repo-local `triton-cpu` source install with `TRITON_INTERPRET=1` and
`TRITON_CPU_BACKEND=1`. The scaffold checks exact softmax composition for
synthetic scalar values only; it does not prove a GPU systems win.

## Approximate Attention Reference Gate

Added a Phase 3 reference for the live approximate branch:

- CPU reference: `phase3/reference/approx_sink_attention.py`
- tests: `phase3/tests/test_approx_sink_attention_reference.py`
- note: `phase3/approx_sink_attention_reference.md`

This reference keeps all non-sink tail logits exact, replaces only fixed-sink
logits with a predictor, and then runs the normal softmax denominator. Tests
verify that exact sink logits reproduce exact attention. This is the operator a
future Triton/CUDA kernel must match before native timing is meaningful.

## Approximate Triton Interpreter Scaffold

Added a Phase 4 Triton-interpreter scaffold for the live approximate operator:

- wrapper: `phase4/kernel/approx_sink_attention_triton.py`
- tests: `phase4/tests/test_approx_sink_attention_triton_interpret.py`
- note: `phase4/approx_sink_attention_triton_gate.md`

The tests now execute locally in interpreter mode under the repo-local
`triton-cpu` source install. This closes the Macbook correctness dependency
gate for the approximate operator. Native GPU timing should still wait for a
separate CUDA/NVIDIA run because interpreter success is not throughput, HBM, or
kernel scheduling evidence.

## 2026-05-05 Local Validation Rerun

At that time, the project-owned Phase 2/3/4 tests in `./venv_arm64` reported
16 passed and 4 Triton interpreter dependency skips because `triton` was not
importable in the Mac-local venv. Reran the head-selective rank-2 gate; it reproduced the
weakened decision with 19/72 selected heads and held-out output rel-L2 `0.2035`,
worse than position-only (`0.1724`) and all-rank2 (`0.1419`). The next useful
pre-GPU improvement is not another simple validation head selector; it is a
repeatable all-rank2 quality/stability result or a different stability
mechanism.

## 2026-05-06 Local Validation Rerun

Added and ran the trace-level frozen split gate in `./venv_arm64`. Targeted
tests for split stability, length/sink sweep, and trace-level frozen splits
passed (`9 passed`). The full SinkAware Phase 2/3/4 suite also passed locally
at that time with Triton dependency skips (`25 passed, 4 skipped`). The
trace-level gate initially produced
`phase2/rank2_trace_frozen_split_gate.md` and `.json` with 24 traces and status
**ALIVE but bounded**: all-head rank-2 beat position-only on every held-out
trace split, but the per-head caveat remained.

## 2026-05-06 Larger Frozen Trace Slice

Reran `phase2/rank2_trace_frozen_split_gate.py` in `./venv_arm64` with
`--max-traces 48 --max-length 96 --sink-tokens 4 --train-fraction 0.67 --seeds
0 1 2`. The larger frozen slice keeps the branch **ALIVE but bounded**: all
three whole-trace splits are positive, with output rel-L2 improvement
`+0.0379 +/- 0.0014`, minimum split `+0.0367`, sink-mass MAE improvement
`+0.0220 +/- 0.0008`, and attention-L1 improvement `+0.0461 +/- 0.0014`.
The head win rate remains low at `0.278 +/- 0.016`, so the stronger slice
supports aggregate repeatability but does not revive any per-head robustness
claim.

## 2026-05-06 Triton Recheck and Cross-Family Smoke Fallback

Before the later source build, rechecked `./venv_arm64` directly with
`importlib.util.find_spec("triton")` and `import triton`; Triton was still not
importable (`ModuleNotFoundError: No module named 'triton'`). Because the
interpreter gate was blocked locally, added and ran
`phase2/rank2_cross_model_falsification_gate.py`
with `--model-names distilgpt2 facebook/opt-125m --max-traces 12 --max-length
64 --sink-tokens 4 --train-fraction 0.67 --seeds 0`.

Result: **ALIVE but bounded** as a smallest Mac-feasible falsification smoke.
The gate fits all-head rank-2 predictors separately per model on whole-trace
train splits and evaluates held-out traces against position-only. It is not
cross-model predictor transfer and makes no GPU speed claim.

Aggregate output rel-L2 improvement across the two model rows was
`+0.0519 +/- 0.0372`; the minimum model improvement was `+0.0329`.
Per model:

- `distilgpt2` (`gpt2`): position output rel-L2 `0.1602`, rank-2 `0.1273`,
  improvement `+0.0329`.
- `facebook/opt-125m` (`opt`): position output rel-L2 `0.3562`, rank-2
  `0.2853`, improvement `+0.0709`.

Decision: this keeps the branch alive under a stricter model-family
falsification attempt, but only as a smoke result. It does not override the
48-trace per-head fragility caveat. The next exact gate is either Triton
interpreter correctness once `triton` is available in the repo-local venv, or a
larger cross-family repeat with more traces and split seeds.

## 2026-05-06 Larger Repeated Cross-Family Falsification

Reran `phase2/rank2_cross_model_falsification_gate.py` in `./venv_arm64` with
`--model-names distilgpt2 facebook/opt-125m --max-traces 24 --max-length 64
--sink-tokens 4 --train-fraction 0.67 --seeds 0 1 2`.

Result: **ALIVE but bounded** as a larger repeated held-out/model-family gate.
The gate still fits predictors separately per model, so it is not cross-model
predictor transfer and it makes no GPU speed claim. Both model families stayed
positive across all three whole-trace split seeds:

- Aggregate model-row output rel-L2 improvement: `+0.0557 +/- 0.0424`;
  minimum model-row improvement: `+0.0341`.
- `distilgpt2` (`gpt2`): output rel-L2 improvement `+0.0341 +/- 0.0018`,
  minimum split `+0.0323`, head win rate `0.958 +/- 0.016`.
- `facebook/opt-125m` (`opt`): output rel-L2 improvement
  `+0.0774 +/- 0.0043`, minimum split `+0.0742`, head win rate
  `0.972 +/- 0.016`.

Decision: the prior 12-trace one-seed smoke is promoted to a larger repeated
cross-family falsification pass, but only as bounded Mac-local evidence. The
method remains below ICLR readiness because there is still no Triton
interpreter pass, no native GPU timing, no end-to-end quality benchmark, and no
cross-model predictor transfer result.

## 2026-05-06 48-Trace Repeated Cross-Family Falsification

Reran `phase2/rank2_cross_model_falsification_gate.py` in `./venv_arm64` with
`--model-names distilgpt2 facebook/opt-125m --max-traces 48 --max-length 64
--sink-tokens 4 --train-fraction 0.67 --seeds 0 1 2`.

Result: **ALIVE but bounded** as a measured larger held-out/model-family gate.
The gate still fits predictors separately per model, so it is not cross-model
predictor transfer. It keeps all non-sink scores exact, reports Mac-local
attention-output drift only, and makes no GPU speed or end-to-end quality
claim. Both model families stayed positive across all three whole-trace split
seeds:

- Aggregate model-row output rel-L2 improvement: `+0.0547 +/- 0.0472`;
  minimum model-row improvement: `+0.0306`.
- `distilgpt2` (`gpt2`): measured output rel-L2 improvement
  `+0.0306 +/- 0.0023`, minimum split `+0.0283`, head win rate
  `0.986 +/- 0.016`.
- `facebook/opt-125m` (`opt`): measured output rel-L2 improvement
  `+0.0788 +/- 0.0069`, minimum split `+0.0731`, head win rate
  `0.991 +/- 0.009`.

Decision: the prior 24-trace repeated cross-family row is strengthened to a
48-trace repeat without changing the claim boundary. SinkAware remains below
ICLR readiness because there is still no native GPU timing, no downstream
quality benchmark, and no cross-model predictor transfer result. Focused
validation after the run passed:
`./venv_arm64/bin/python -m pytest experimental/sinkaware/phase2/tests/test_rank2_cross_model_falsification_gate.py experimental/sinkaware/phase2/tests/test_rank2_trace_frozen_split_gate.py experimental/sinkaware/phase3/tests/test_approx_sink_attention_reference.py`
(`10 passed`, 2 warnings).

## 2026-05-06 Cross-Family Length Stability Gate

Added and ran `phase2/rank2_cross_model_length_stability_gate.py` in
`./venv_arm64` with `--model-names distilgpt2 facebook/opt-125m --max-traces
48 --max-lengths 64 96 --sink-tokens 4 --train-fraction 0.67 --seeds 0 1 2`.

Result: **ALIVE but bounded** as a stronger non-Triton Mac gate. The gate
requires both GPT2-family and OPT-family rows to stay positive at lengths 64
and 96 under whole-trace splits. It still fits predictors separately per model
and length, keeps non-sink QK scores exact, reports attention-output drift
only, and makes no GPU speed, downstream-quality, or predictor-transfer claim.

- Aggregate model/length output rel-L2 improvement:
  `+0.0535 +/- 0.0262`; minimum model/length improvement: `+0.0301`.
- Layer-head output win rate across model/length rows: `0.982 +/- 0.008`.
- `distilgpt2` length 64: `+0.0306 +/- 0.0023`; length 96:
  `+0.0301 +/- 0.0018`.
- `facebook/opt-125m` length 64: `+0.0788 +/- 0.0069`; length 96:
  `+0.0744 +/- 0.0040`.

Decision: this strengthens the held-out/model-family evidence because the
positive row survives the stronger length axis without touching Triton. It
still does not close the paper blocker: SinkAware needs a downstream quality
control or a runnable interpreter/native path before any paper-grade positive
method claim.

## 2026-05-06 Triton CPU Interpreter Gate

Checked the official Triton source-install guidance and the experimental
`triton-lang/triton-cpu` repository. A repo-local source build now installs
`triton==3.7.0+git270e696d` into `./venv_arm64` from `triton-cpu` revision
`270e696` with `third_party/sleef` at `93f04d8`.

Validated with:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest experimental/sinkaware/phase4/tests -rs
```

Result: SinkAware Phase 4 interpreter tests pass (`5 passed` as part of the
`9 passed` cross-project Phase 4 suite). The owned Phase 0--4 project suite
also passes (`103 passed, 2 warnings`).

Decision: **Triton interpreter correctness is no longer the blocking pre-GPU
gate**. SinkAware is still only bounded and pre-submission because it lacks
downstream quality controls, predictor transfer, and native GPU speed/memory
evidence.

## 2026-05-06 Downstream Quality/Control Gate

Added and ran `phase3/downstream_quality_control_gate.py` in `./venv_arm64`
with `--model-names distilgpt2 facebook/opt-125m --max-traces 24 --max-length
64 --sink-tokens 4 --train-fraction 0.67 --seeds 0 1 2`.

Result: **ALIVE but bounded** as a first downstream causal-LM control smoke.
The gate patches live GPT2/OPT attention modules during full causal-LM
forwards and compares exact baseline attention, exact sink-logit replacement
as a no-op control, position-only replacement, and rank-2 replacement. The
exact replacement reproduced baseline loss and logits on both models. Rank-2
was closer than position-only on both downstream loss drift and KL:

- Aggregate rank-2 absolute loss-delta improvement over position-only:
  `+0.0809 +/- 0.0815`; minimum model improvement `+0.0393`.
- Aggregate rank-2 KL-to-exact improvement over position-only:
  `+0.0825 +/- 0.1078`.
- `distilgpt2`: mean position loss delta `+0.0947`, rank-2 `+0.0554`;
  loss improvement `+0.0393 +/- 0.0134`, KL improvement
  `+0.0275 +/- 0.0050`.
- `facebook/opt-125m`: mean position loss delta `+0.2213`, rank-2
  `+0.0987`; loss improvement `+0.1225 +/- 0.0284`, KL improvement
  `+0.1375 +/- 0.0180`.

Decision: this expands the first downstream-control smoke to a repeated
24-trace GPT2/OPT-family gate and keeps SinkAware alive as an approximate
rank-2 branch. It is still not benchmark success, predictor transfer, or GPU
speed evidence. The next exact Mac gate is length/sink expansion of the
downstream control; otherwise the remaining systems gate is native NVIDIA
timing only after the operator remains quality-bounded.

`phase3/downstream_quality_control_sweep.md` runs that length/sink expansion
over max lengths 64/96 and sink-token counts 2/4, with 24 traces, split seeds
0/1/2, and both distilgpt2 and facebook/opt-125m. Rank-2 stays closer than
position-only on every model/config row. Aggregate loss-delta improvements are
`+0.0544 +/- 0.0505` at length64/sink2, `+0.0809 +/- 0.0815` at length64/sink4,
`+0.0433 +/- 0.0317` at length96/sink2, and `+0.0728 +/- 0.0619` at
length96/sink4. Minimum model improvement remains positive in every config
(`+0.0272` or higher), and exact replacement remains a no-op.

Decision: **ALIVE but bounded** after the downstream length/sink sweep. The
Mac-local quality-control surface is now stronger; the remaining useful gate is
native timing, unless a reviewer requires a larger trace slice before GPU work.

## 2026-05-06 Larger Downstream Quality/Control Repeat

Ran the reviewer-requested larger downstream control repeat in `./venv_arm64`
with 48 traces, split seeds 0/1/2, sink count 4, and separately fit predictors
for `distilgpt2` and `facebook/opt-125m` at lengths 64 and 96. Exact
sink-logit replacement remained a no-op in both rows. Rank-2 stayed closer than
position-only on downstream loss drift and KL for both models.

- length 64 / sink 4: aggregate absolute loss-delta improvement
  `+0.0801 +/- 0.0894`; aggregate KL improvement `+0.0752 +/- 0.1003`;
  minimum model loss improvement `+0.0345`.
- length 96 / sink 4: aggregate absolute loss-delta improvement
  `+0.0721 +/- 0.0676`; aggregate KL improvement `+0.0694 +/- 0.0823`;
  minimum model loss improvement `+0.0376`.

Decision: **ALIVE but bounded** after the larger downstream repeat. This is the
strongest Mac-local quality-control surface now available. It still does not
support benchmark accuracy, cross-model predictor transfer, latency, HBM,
throughput, or native speed claims. The next exact gate is native GPU timing and
memory-traffic measurement, not more Mac-local quality controls unless a
reviewer requires a much larger CPU slice.

## 2026-05-06 Larger Downstream Sink-2 Repeat

Ran the missing reviewer-requested larger downstream control repeat for sink
count 2 in `./venv_arm64`, matching the stronger sink-4 setup: 48 traces, split
seeds 0/1/2, lengths 64 and 96, and separately fit predictors for `distilgpt2`
and `facebook/opt-125m`. Exact sink-logit replacement remained a no-op in both
rows. Rank-2 stayed closer than position-only on downstream loss drift and KL
for both models.

- length 64 / sink 2: aggregate absolute loss-delta improvement
  `+0.0621 +/- 0.0701`; aggregate KL improvement `+0.0532 +/- 0.0772`;
  minimum model loss improvement `+0.0263`.
- length 96 / sink 2: aggregate absolute loss-delta improvement
  `+0.0537 +/- 0.0509`; aggregate KL improvement `+0.0464 +/- 0.0580`;
  minimum model loss improvement `+0.0277`.

Decision: **MAC-LOCAL DOWNSTREAM CONTROL SURFACE SATURATED** for the current
branch. Sink counts 2 and 4, lengths 64 and 96, GPT2/OPT families, 48 traces,
and three split seeds are now covered. The remaining useful evidence is native
GPU timing/memory traffic and, for a stronger paper, a true benchmark-quality
result rather than more Mac-local patch controls.

## 2026-05-06 Downstream Rank Frontier

Added rank-frontier support to `phase3/downstream_quality_control_gate.py` and
ran one bounded downstream frontier in `./venv_arm64` with 24 traces, length 96,
sink count 4, split seeds 0/1/2, `distilgpt2`, `facebook/opt-125m`, and ranks
1/2/4/8.

Result: higher ranks monotonically reduce downstream drift, but the cost model
keeps rank 2 as the live systems compromise.

- rank1: abs loss delta `0.117`, loss improvement vs position `+0.037`, top-1
  disagreement `0.145`.
- rank2: abs loss delta `0.081`, loss improvement `+0.073`, top-1 disagreement
  `0.111`.
- rank4: abs loss delta `0.044`, loss improvement `+0.110`, top-1 disagreement
  `0.085`.
- rank8: abs loss delta `0.029`, loss improvement `+0.125`, top-1 disagreement
  `0.069`.

Decision: **RANK FRONTIER EXPLAINS THE QUALITY/COST TRADEOFF**. Rank4/rank8 are
better quality controls but lose the current multiply-add wedge against exact
four-sink QK. Rank2 remains the only plausible pre-GPU systems compromise.

## 2026-05-06 Larger Downstream Rank Frontier Repeat

Reran the downstream frontier in `./venv_arm64` with 48 traces, length 96,
sink count 4, split seeds 0/1/2, `distilgpt2`, `facebook/opt-125m`, and ranks
1/2/4/8:

```bash
HF_HOME="$PWD/experimental/sinkaware/.debug/hf_cache" \
TRANSFORMERS_CACHE="$PWD/experimental/sinkaware/.debug/hf_cache" \
./venv_arm64/bin/python -m experimental.sinkaware.phase3.downstream_quality_control_gate \
  --model-names distilgpt2 facebook/opt-125m \
  --max-traces 48 --max-length 96 --sink-tokens 4 \
  --train-fraction 0.67 --seeds 0 1 2 --ranks 1 2 4 8 \
  --artifact-stem downstream_rank_frontier_traces48_len96_sink4
```

Result: **RANK FRONTIER REPEATS ON 48 TRACES**. Exact sink-logit replacement
remains a no-op. Higher ranks again monotonically reduce downstream drift, but
rank4/rank8 still lose the simple multiply-add wedge against exact four-sink
QK.

- rank1: abs loss delta `0.137`, loss improvement vs position `+0.031`,
  KL improvement `+0.029`, top-1 disagreement `0.143`, minimum model
  improvement `+0.023`.
- rank2: abs loss delta `0.096`, loss improvement `+0.072`, KL improvement
  `+0.069`, top-1 disagreement `0.125`, minimum model improvement `+0.038`.
- rank4: abs loss delta `0.062`, loss improvement `+0.107`, KL improvement
  `+0.097`, top-1 disagreement `0.095`, minimum model improvement `+0.056`.
- rank8: abs loss delta `0.044`, loss improvement `+0.125`, KL improvement
  `+0.114`, top-1 disagreement `0.080`, minimum model improvement `+0.071`.

Decision: this supersedes the 24-trace frontier without changing the claim
boundary. Better predictors exist, but rank2 remains the live systems
compromise until native GPU timing/memory traffic shows whether the lower-rank
path has a real implementation advantage. No GPU speed, benchmark accuracy, or
cross-model predictor transfer claim is supported.

## 2026-05-06 Final Mac/Triton Repro Check

Found and fixed a Phase 4 reproducibility bug: setting only
`TRITON_INTERPRET=1` inside individual tests was too late on this Mac and could
fall through to the `triton-cpu` linker path, which fails without `libgcc`.
The Phase 4 pytest suite now sets `TRITON_INTERPRET=1`,
`TRITON_CPU_BACKEND=1`, and repo-local `TRITON_HOME` during collection.

Validated with:

```bash
./venv_arm64/bin/python -m pytest \
  experimental/sinkaware/phase4/tests \
  experimental/sinkaware/phase3/tests/test_downstream_quality_control_rank_frontier.py \
  experimental/sinkaware/phase3/tests/test_approx_sink_attention_reference.py -rs
```

Result: `10 passed, 2 warnings`.

Decision: no further Mac/Triton improvement remains before native GPU timing.
The remaining blocker is native GPU latency/HBM plus benchmark-quality
downstream evidence; additional local loops would mostly churn the same bounded
diagnostics.

## 2026-05-06 Native GPU Packet Validator

Added a Mac-local admissibility checker for future SinkAware native GPU packets:

- checker: `phase2/check_native_gpu_packet.py`
- tests: `phase2/tests/test_check_native_gpu_packet.py`

The checker validates required runbook artifacts (`metadata.json`,
`quality_drift.csv`, `quality_drift_by_head.csv`, `latency.csv`,
`ncu_summary.csv`, and `decision.md`), rejects placeholder packet files, checks
all four canonical rows, requires three distinct latency `run_id` values per
row, and rejects non-native metadata or non-numeric metric cells.

Validated with:

```bash
./venv_arm64/bin/python -m pytest \
  experimental/sinkaware/phase2/tests/test_check_native_gpu_packet.py -q
```

Result: `7 passed`.

Decision: this is a useful final Mac-side improvement because it prevents
missing or partial native timing/quality/NCU packets from being cited. It does
not create GPU evidence or change the method claim. The remaining gate is still
a native NVIDIA packet that passes the validator and then clears the runbook's
speed/memory plus quality criteria.

## 2026-05-06 NVIDIA Metadata Scope Hardening

Tightened the native packet validator to enforce the intended NVIDIA gate. The
checker already rejected CPU/MPS/Mac-local metadata; it now also rejects
non-NVIDIA accelerators such as AMD GPU metadata. Added a regression test so a
synthetic `metadata.json` with `gpu: AMD MI300X` fails admissibility.

Decision: **ADMISSIBILITY HARDENING ONLY**. This keeps returned native packets
aligned with the runbook and paper scope, but it does not create GPU evidence.

## 2026-05-06 Same-Shape Latency Repeat Hardening

Closed a native timing reproducibility loophole in `latency.csv` validation.
The validator already required three distinct `run_id` values per canonical
row. It now also requires those repeats to share the same
`model`/`sequence_length`/`batch_size` shape, so a packet cannot satisfy the
repeat gate by mixing sequence lengths or batch shapes. Added a regression test
that varies `sequence_length` by `run_id` and expects checker failure.

Decision: **TIMING PACKET HARDENED**. This improves future native packet
admissibility only; native timing/memory data are still required.

## 2026-05-06 Native Runbook Repeat Contract Alignment

Aligned the GPU runbook and paper wording with the stricter validator. Native
latency evidence now has one written contract everywhere: three distinct
`run_id` values are required for each row/model/sequence-length/batch-size
group, not merely three runs for a canonical row with mixed shapes.

Decision: **RUNBOOK/PAPER CONTRACT MATCHES CHECKER**. SinkAware remains
Mac-complete and blocked on native GPU timing/memory traffic plus quality
preservation; this edit only prevents ambiguous native packets from being
cited.

## 2026-05-06 Cross-Artifact Shape Consistency Hardening

Closed the remaining native-packet reviewer loophole found by subagent review.
The validator now requires the same row/model/sequence-length/batch-size groups
to appear across `quality_drift.csv`, `quality_drift_by_head.csv`,
`latency.csv`, and `ncu_summary.csv`. A packet can no longer pass by measuring
quality at one sequence length, latency at another, and NCU counters at a
third.

Decision: **NATIVE DECISION SURFACE MUST BE PAIRED**. SinkAware is now
Mac-complete unless a new reviewer asks for a specific local control. The next
exact gate is native NVIDIA timing/memory traffic over the paired packet.

## 2026-05-06 Metadata Shape Consistency Hardening

Closed the follow-up metadata loophole in the same native packet validator.
`metadata.json["sequence_shapes"]` must now match the measured
sequence-length/batch-size groups in the CSV artifacts. A packet can no longer
pass with quality, latency, and NCU rows at one shape while metadata claims a
different shape.

Decision: **NATIVE METADATA MUST MATCH MEASURED SHAPES**. This is
reproducibility hardening only; the scientific gate remains native NVIDIA
timing/memory traffic plus quality preservation.

## 2026-05-06 CUDA Metadata Negative-Token Hardening

Closed the final metadata token loophole found by subagent review. The native
packet validator now treats `cuda: "not available"` as non-native CUDA metadata
instead of accepting it because the GPU field says NVIDIA.

Decision: **NON-NATIVE CUDA METADATA REJECTED**. This is admissibility
hardening only; it does not create GPU evidence.
