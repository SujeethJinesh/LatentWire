# Repo Readiness Review

- date: `2026-04-26`
- status: not ICLR-ready
- estimated distance: one stable positive method plus larger-slice, seed-repeat,
  source-control, and cross-family replication gates

## Current Paper Story

The honest current story is conditional innovation rather than proven latent
transfer. Target-side candidate/self-repair gives a strong decoder floor, C2C
shows real cache-level headroom, and LatentWire has a mature evaluation harness.
What is missing is a deployable LatentWire method whose improvement is both
source-derived and stable under source-destroying controls.

Update `2026-04-27`: the no-harm CPU replay kills shallow source-predicate
decoding on current artifacts. A 4-bit candidate syndrome still has source
specificity on holdout (`4` clean source-necessary IDs, control clean union
`0`) but harms `14` target-self examples. Stronger abstention removes harms but
also removes clean gains. The live branch remains `none`; next viable branches
are learned semantic predicates with erasure-aware abstention and zero-init
target-preserving query bottlenecks after the MPS blocker clears. The blocker
persisted after both `kill -9` and `sudo kill -9`, so the next action is
OS/session-level cleanup before any MPS experiment.

Update `2026-04-27 00:50 PDT`: KVComm is now harness-ready for strict
source-control evaluation, but it remains baseline/tooling evidence rather than
a positive method. The wrapper supports matched, zero-source, shuffled-source,
and target-only final modes under one matched-only layer selection. A CPU smoke
over two examples verifies provenance and fixed-layer reuse, with all modes at
`0/2` and shuffled-source using nonmatching source IDs. The live branch remains
`none`; the highest-priority executable gate after MPS clears is the
`kvcomm_svamp32_controls_smoke_20260427` command recorded in the ledger.
Follow-up hardening replaced fixed-offset shuffled-source with deterministic
hash-based non-self pairing and answer-overlap logging; the CPU smoke still
passes as tooling-only evidence.
Second follow-up hardening added configured paired sidecar baselines so KVComm
artifacts now report matched-vs-target-only, matched-vs-zero-source, and
matched-vs-shuffled-source flip tables.
Third follow-up hardening added cache-derived byte telemetry. On the CPU smoke,
selected-layer KVComm controls average `530432` communicated bytes/example,
while target-only is `0`; this is systems telemetry only, not method evidence.

The strongest bound is the SVAMP32 C2C-derived syndrome sidecar:

- strict target-side pool: `14/32`
- clean source-necessary IDs: `2/6`
- controls clean union: `0/6`
- syndrome size: `1` byte
- blocker: the syndrome uses C2C numeric answers, so this is a bound, not a
  deployable positive method

The first C2C-mechanism distillation attempt is now negative:

- scalar prefill trace: matched `11/32`, clean source-necessary `0/6`
- residual prefill trace: matched `12/32`, clean source-necessary `0/6`
- target-only decoder floor: `14/32`
- decision: do not scale summary-feature C2C syndrome distillation without a
  new token/layer-level mechanism reason

The first Perceiver answer-teacher contrastive connector is also negative:

- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by shuffled-source,
  target-only, or slots-only controls
- decision: do not run generation for this checkpoint

The same branch also fails on the stronger SVAMP70 C2C-vs-process-repair
surface:

- clean C2C source-only IDs after excluding process-repair: `10`
- teacher-forced gate `0.15`: matched-positive clean `4/10`
- matched-only clean: `0/10`
- control-leak clean: `4/10`
- decision: kill this connector family until the objective changes

The first objective-level rescue also fails on SVAMP32:

- added training-time anti-memory controls against `target_only` and
  `slots_only`
- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by zero-source or
  slots-only controls
- mean matched-control delta remains negative at all tested gates
- decision: do not run generation; pivot away from receiver-conditioned
  Perceiver/delta-memory signal formation unless a materially new objective or
  architecture reason appears

The first true source-conditioned soft-prefix logprob gate is also negative:

- source-only matched connector with fold-local feature standardization,
  numeric-only distractors, and mean-token continuation logprob
- matched-only clean source-communication candidate IDs: `1/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.771126`
- decision: kill global summary soft-prefix connectors on this surface before
  generation

The first token-local cross-attention rescue also fails its first rung:

- target-query cross-attention into standardized source token states
- matched-only clean source-communication candidate IDs: `0/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.383649`
- decision: do not scale this exact tiny prefix-emitting cross-attention
  connector by epochs or width without a new hypothesis

The source-control contrastive variant of that cross-attention gate also fails:

- training penalizes zero-source, shuffled-source, same-norm-noise, and
  projected-source controls when they match or beat the real-source margin
- matched-only clean source-communication candidate IDs: `0/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.382854`
- decision: objective-level control penalties do not rescue this tiny
  prefix-emitting cross-attention architecture; do not tune this exact family
  further without a larger architectural change

The target-side continuation-loss rescue of the same family also fails:

- training objective changed from gold-vs-distractor margin to target
  continuation next-token CE
- heldout logprob on SVAMP32 clean C2C-headroom IDs: matched-only clean `0/6`,
  clean control leaks `4/6`, mean matched-minus-control clean margin
  `-0.194783`
- 64-token generation on the six clean IDs: matched `1/6`, while zero-source,
  shuffled-source, target-only prefix, and slots-only prefix each reach `2/6`
- decision: kill the low-capacity prefix-emitter family on this surface; the
  next learned-interface branch must expose a larger, rate-controlled source
  memory or use a different source/surface pair

The current live branch is now the query-innovation/source-memory resampler:

- historical GSM8K32 query-innovation rows are finite and target-safe enough to
  probe, but their small accuracy gains were retained under zero/shuffled
  source controls
- this cycle added eval-only gold-answer continuation scoring to
  `latent_bridge/evaluate.py`, so the next gate can compare matched source
  against zero-source, shuffled-source, target-only, and slots-only controls on
  the same generated-answer surface
- CPU micro-smoke on four GSM8K examples fails: matched mean answer logprob
  `-7.025400`, zero-source `-6.925437`, shuffled-source `-7.048394`,
  slots-only `-7.025400`, matched-best-control delta `-0.115530`, and
  matched best-control wins/losses/ties `0/4/0`
- decision: kill the current finite non-target-conditioned query-innovation
  checkpoint as a live source-communication row
- next branch: target-conditioned query-innovation/source-memory connector
  that supports `target_only` and `slots_only` controls from the first gate
- current blocker: an orphaned MPS calibration process, PID `31103`, is stuck
  in `STAT=UE` and ignores `SIGKILL`; do not start more MPS runs until it is
  cleared

The target-conditioned query-memory follow-up is now also negative:

- SVAMP32 delta-memory CPU answer-likelihood smoke fails because matched source
  loses to zero-source on mean answer likelihood and wins `0/4` against the
  best runnable control
- SVAMP70 Perceiver answer-teacher CPU answer-likelihood smoke fails with
  best-control wins `0/4` and mean matched-minus-best-control delta
  `-0.112360`
- Qwen2.5-Math SVAMP32 Perceiver has one partial positive clue on four clean
  IDs: matched beats every control with best-control wins `3/4` and mean
  live-best delta `+0.080362`
- the required six-clean-ID expansion fails: matched still beats zero-source,
  but loses on mean to shuffled-source, target-only, and slots-only controls;
  mean matched-minus-best-control delta is `-0.090384`
- decision: kill target-memory/query-memory Perceiver checkpoints as the
  current live positive-method branch; no method is live until a source
  surface/interface reset selects the next branch
- current blocker remains PID `31103`, the stuck MPS calibration process

The subsequent no-harm source-predicate replay is also negative:

- candidate syndrome bits4: live clean source-necessary `1` with `16`
  target-self harms; holdout clean source-necessary `4` with `14` target-self
  harms; control clean union `0`
- source predicate router with stronger no-harm pressure: best rows reach only
  `23/70`, clean `3`, accepted harm `1`, and fail the matched-correct gate
- source likelihood no-harm gate: accepted harm `0`, control clean union `0`,
  but clean source-necessary `0` on both live and holdout
- decision: prune shallow numeric/hash syndrome and source-text predicate
  routers on current artifacts; revive only learned semantic predicates with
  erasure-aware abstention or stronger source surfaces

The learned semantic-predicate CPU decoder is now also negative on holdout:

- new analyzer: `scripts/analyze_svamp_source_semantic_predicate_decoder.py`
- strict harm20 gate: live `25/70`, clean source-necessary `3`, accepted harm
  `0`, control clean union `0`
- holdout: `9/70`, clean source-necessary `0`, accepted harm `0`, control
  clean union `0`
- decision: target-safe live recovery is possible, but it does not transfer;
  prune generated-source-trace semantic predicate decoding on current
  Qwen2.5-Math -> Qwen3 SVAMP artifacts

The CPU target-likelihood receiver follow-up is also negative on live:

- scorer: `Qwen/Qwen3-0.6B` on CPU over target/text/source normalized answer
  candidates
- target-alone/text/source candidate correctness: `21/70`, `22/70`, `13/70`
- top-likelihood selection: `14/70`, with source selected on `64/70`
  examples
- accept-all source-top recovers all `6` clean live source-only IDs, but harms
  `16` target-correct examples
- simple no-harm live thresholds recover at most `1` clean source-only ID and
  remain around `22-23/70`, below the `25/70` live gate
- decision: prune this target-likelihood receiver variant before holdout; a
  future receiver-gate claim needs true condition-specific rescored controls
  rather than sketch shuffling or forced target fallback

The SVAMP70 exact-ID overlap audit rules out another threshold sweep on the
current canonical surface:

- canonical live has `6` clean source-only IDs, and all have been recovered by
  at least one audited branch, but the reusable recoveries cluster on a few
  live examples and come from branches that either fail holdout or harm
  target-correct cases
- canonical holdout has only `2` clean source-only IDs; only
  `daea537474de16ac` is recovered, and only by the trace-router family that
  fails the full gate
- adjacent scout positives are not canonical holdout evidence and usually come
  with target-self harm
- decision: stop CPU threshold/router sweeps on current SVAMP70 artifacts; next
  CPU work should be a true condition-specific receiver-control harness, while
  source-surface/interface reset waits for the MPS blocker to clear

The condition-specific receiver-control harness is now implemented and tested:

- new analyzer: `scripts/analyze_condition_likelihood_receiver_gate.py`
- focused tests: `tests/test_analyze_condition_likelihood_receiver_gate.py`
- verification: `12` likelihood/receiver tests passed; py_compile passed
- purpose: evaluate target-likelihood receiver gates only when each control has
  its own receiver-scored candidate pool
- status: harness-ready, not evidence; next gate is CPU collection of
  condition-specific sketches if MPS remains blocked

The condition-specific target-likelihood receiver is now killed on the current
SVAMP70 surface before control collection:

- matched-only live CV reaches only `15/70`
- clean source-necessary IDs: `1`
- accepted target-correct harm: `7`
- duplicate-answer clean IDs: `0`
- decision: do not collect remaining controls or holdout for this branch; keep
  the candidate-pool builder and duplicate-answer de-dup harness for a stronger
  source surface

The top-surface cross-attention rescue also fails:

- after consolidated surface reselection, `svamp70_live` and `svamp70_holdout`
  are the strongest source-complementary surfaces
- rerunning the same cross-attention gate on `svamp70_live` gives matched-only
  clean IDs `0/6`, clean control leaks `3/6`, and mean matched-control clean
  margin `-0.443233`
- decision: tiny learned prefix emitters are not the live branch unless a new
  mechanism directly addresses control dominance

The simplest source-only sidecar/router is also negative:

- source-generated numeric residue sidecar with target-side candidate-pool
  decoding
- source numeric coverage: `32/32`
- matched: `4/32`
- target-self preserve: `0/3`
- clean source-necessary IDs: `0/6`
- controls clean union: `0/6`
- decision: kill raw source-generated numeric residue sidecars; the clean
  issue is weak source signal, not target/control leakage

The strongest GSM mechanism clue is `dynalign_module_replace_residrank16`:

- GSM8K32 smoke: `4/32` vs target `2/32`
- GSM8K70 seed 0: `8/70` vs target `4/70`
- seed-0 source controls: zero/shuffle with target fallback retain `0/6` live
  wins
- blocker: seed stability fails; seed 3 is `2/70`, seed 4 is finite but only
  `4/70` with paired `3W/3L/64T`, and seeds 1/2 hit nonfinite checkpoint
  failures

## What Is Done

- Paper/evidence ledgers exist and are useful:
  - `paper/experiment_ledger_20260421.md`
  - `paper/benchmark_expansion_order_20260422.md`
  - `paper/reviewer_feedback.md`
  - SVAMP/GSM per-run memos under `paper/`
- Evaluation machinery is broad:
  - frozen-slice materialization and exact-ID utilities
  - paired/bootstrap uncertainty and oracle/headroom analyzers
  - source controls: zero source, shuffled source, target-only, slots-only, and
    same-norm/noise controls where applicable
  - GSM residual campaign/sweep wrappers with checkpoint health checks
  - SVAMP source/oracle/syndrome analyzers
  - C2C, KVComm, KVPress, and LatentMAS wrapper or matrix support
- Tests are healthy for unit/schema/tooling coverage:
  - `./venv_arm64/bin/python -m pytest -q`
  - result on this review: `668 passed in 26.77s`

## Saturated Or Weakened Branches

- Raw dynalign residual is a mechanism probe, not a paper method, because the
  finite repeat is target-negative and two repeat seeds are nonfinite.
- Selector-gap accept/fallback is killed as a method: fresh zero/shuffle controls
  retain gated wins, so the score is not source-specific.
- Further simple whitening/conditioning is low priority: it can prevent nonfinite
  failures but trades away the seed-0 ceiling.
- Query-pool and ID-weighted SVAMP variants are saturated below the clean gate:
  best rows recover only `1/6` clean residual IDs.
- Perceiver-query, delta-memory, contrastive delta-memory, answer-teacher
  microfit, pooled source-hidden syndrome, and learned source-token syndrome
  probes all fail the source-derived clean gate.
- C2C prefill scalar/residual summary syndrome probes fail the strict SVAMP32
  gate and remain below the target-only decoder floor.
- Perceiver-query answer-teacher plus source-control contrast fails the
  teacher-forced pre-generation gate; target/control memory still explains the
  apparent clean-ID signal.
- Scaling the same Perceiver answer-teacher connector to SVAMP70 also fails
  the teacher-forced pre-generation gate.
- Adding anti-memory target-only/slots-only training controls to that Perceiver
  branch also fails the SVAMP32 teacher-forced pre-generation gate.
- Source-generated numeric residue sidecar/router is killed: it avoids control
  leakage but has no clean source-necessary recovery.
- Source final-answer copying and stronger-source source-margin escalation are
  killed for the current frozen SVAMP32 clean IDs.
- Direct source-hidden syndrome readout is killed, including all-layer pooled
  features: all-layer Qwen2.5 features reach only `9/32`, underperform the
  `14/32` target-only floor, preserve only `2/3` target-self rows, and recover
  `0/6` clean source-necessary IDs.
- The first learned query-bottleneck residue predictor over all-layer summary
  tokens is also negative: it matches the all-layer ridge gate at `9/32`,
  preserves only `2/3` target-self rows, and recovers `0/6` clean
  source-necessary IDs.
- The full all-layer source-token learned syndrome probe is negative:
  matched `7/32`, target-only `14/32`, target-self `2/3`, and clean
  source-necessary `0/6`.
- The process-repair / selector stack is negative on the strict SVAMP32
  source-control surface: process repair selected route reaches only `10/32`
  versus `14/32` target-self repair, recovers `1/6` clean residual IDs, loses
  `2/3` target-self repair wins, and selects the target candidate on `32/32`
  examples.
- The target-safe output-aware dynalign selector/repair branch over existing
  SVAMP32 candidates is now killed by an oracle replay. Even an oracle over
  target_self_repair, dynalign salt 1, dynalign salt 2, and query-pool transport
  reaches only `1/6` clean residual IDs, while the matching zero/shuffle control
  oracle also reaches `18/32` and recovers `1/6` clean residual IDs. Another
  selector over these rows is not worth running.
- The cached Qwen2.5-Math-Instruct source variant is also not a rescue surface:
  on frozen SVAMP32 it reaches source `3/32`, target `8/32`, text relay
  `4/32`, source-only over target `2`, and clean source-only after text
  exclusion `2`. This fails the pre-C2C source-surface gate and should not be
  scaled.
- The source-conditioned summary soft-prefix branch is now killed on the
  Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom surface. After calibration to
  source-only matched prefixes, numeric-only distractors, length-normalized
  continuation logprob, and fold-local standardization, it recovers only `1/6`
  clean IDs and has `4/6` clean control leaks.
- The first token-local source cross-attention prefix branch is also negative:
  it recovers `0/6` clean IDs, has `4/6` clean control leaks, and remains
  dominated by label-shuffled, shuffled-source, and target-only controls.
- Surface reselection after these failures ranks `svamp70_live` and
  `svamp70_holdout` highest, but the same cross-attention prefix gate also
  fails on `svamp70_live` with `0/6` matched-only clean IDs and `3/6` clean
  control leaks.
- Fixed source-quality guarded sidecars are also killed by holdout controls:
  the finalish-short-numeric guard reaches `9/70` with clean source-necessary
  `0/2` and clean control union `2/2`.
- Source-trace self-consistency routing is also killed as the next fixed
  sidecar rescue: live CV reaches only `1` clean source-necessary ID with `2`
  accepted harms, and the single holdout clean win survives equation-result
  permutation.

## Main Gaps

1. No deployable positive method.
   The only clean-passing SVAMP object is the C2C-derived syndrome bound.

2. No stable larger-slice positive row.
   GSM seed 0 is source-dependent, but the method is not seed-stable.

3. Exact reproducibility is incomplete from tracked files alone.
   `results/`, `.debug/`, checkpoint tensors, and external repos are ignored;
   many decisive artifacts are local-only and referenced by hashes or commands.

4. Integration testing is thin.
   The unit suite is strong, but `transformers` and vendor integrations are
   mostly stubbed. Real cached-model integration should be a separate marked
   test lane.

5. Cross-family evidence is intentionally blocked.
   The benchmark order correctly says not to widen until the same-family gate is
   cleared.

## Highest-Priority Next Gate

The current live branch is no longer adjacent source-surface scouting, shallow
source-readout tuning, target-safe selector replay, tiny learned prefix
emitters, or another Perceiver/query-memory checkpoint. Those gates have now
failed or become control-explained.

Latest gate update: KVComm/C2C-style cache communication is now the top
baseline branch because it has a strict source-control harness. It is not a
promoted method branch until matched-source performance on a real decision
slice beats zero-source, shuffled-source, and target-only controls or gives a
clear systems tradeoff at comparable accuracy.

The highest-priority next gate is a source-interface reset on the only
remaining strong reusable surface, or a new source/target scout after the stuck
MPS process is cleared. The existing-artifact re-scan ranks:

- `svamp70_live_source`: target `21/70`, source `13/70`, source-only `9`,
  oracle `30/70`
- `svamp70_holdout_source`: target `8/70`, source `8/70`, source-only `6`,
  oracle `14/70`
- adjacent SVAMP70 scouts, GSM70, and SVAMP32 remain below threshold

Do not spend more compute on fixed decoded guards, shallow source-text routers,
tiny prefix emitters, source-token residue readouts, or Perceiver target-memory
checkpoints. The next method branch must be a materially different rate-capped
source interface on `svamp70_live_source` with immediate
`svamp70_holdout_source` validation, or a fresh source/target scout if a
stronger cached source is available. Process repair should remain a target-side
baseline and confound unless a separate source-derived route signal exists.

Current hard blocker: PID `31103` is an orphaned MPS calibration process in
`STAT=UE` and ignores `SIGKILL`. Do not start more MPS jobs until it is cleared.
The source-surface re-scan command that produced the current branch selection
was:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface svamp70_live_source=target_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_live_source \
  --surface svamp70_holdout_source=target_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_holdout_source \
  --surface svamp70_chal171_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal171_240_source \
  --surface svamp70_chal241_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal241_310_source \
  --surface svamp70_chal311_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal311_380_source \
  --surface gsm70_math_source=target_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_gsm70_source \
  --surface svamp32_math_chat_source=target_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,source_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp32_chat_source \
  --min-source-only 6 \
  --output-json results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.json \
  --output-md results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.md
```

Latest cycle update: the Qwen2.5 -> OPT-350m byte-span module-replace proxy is
killed as a decision surface. It had tokenizer mismatch (`shared decoded =
0.9047`, `boundary F1 = 0.9434`) and the harness now supports OPT-style decoder
layers and projected output rows, but the GSM30 surface was too weak:
target-alone `0/30`, source-alone `0/30`, text relay `3/30`, and byte-span
rotalign proxy `0/30` at `525562.6` bytes/example. This is not a decisive kill
of the sequence-aligned sidecar hypothesis; it is a surface failure. The next
attempt must start from a target/text baseline with nonzero headroom before any
source controls are worth running.

Follow-up surface scout: Phi-3 has only weak headroom on GSM30 (`3/30` target,
`1/30` text relay) and SVAMP30 (`5/30` target, `2/30` text relay). TinyLlama is
dead on SVAMP30 (`0/30` target, `0/30` text relay). Do not spend large compute
on cross-family GQA repairs for these exact surfaces unless a stronger baseline
slice is found first.

DeepSeek-R1-Distill-Qwen-1.5B -> Qwen3-0.6B on frozen SVAMP32 is also weak as
an immediate surface. Target-alone reaches `8/32`, while source-alone and text
relay each reach only `5/32`; text adds `2` target-missed IDs, and the
target/text oracle is only `10/32`. C2C is unavailable because no published C2C
artifact is registered for this pair. Do not spend connector or source-control
compute on this pair in the current loop.

Qwen2.5-Math-1.5B -> Qwen3-0.6B is now the strongest same-family decision
surface, but only with chat-template prompting. A no-chat SVAMP16 probe
produced an artificially weak target floor (`0/16`), so it is not
claim-worthy. With chat templates, frozen SVAMP32 reaches target `8/32`, source
`6/32`, text relay `8/32`, and C2C `15/32`. C2C adds `9` target-missed IDs and
the target/C2C oracle reaches `17/32`, versus target/text oracle `11/32`.

The clean C2C-headroom target set is now explicit: source-alone explains `3`
of the `9` C2C-only wins, text relay explains `0`, leaving `6` clean
C2C-headroom targets and `2` target-only-vs-C2C rows to preserve. This is the
current strict-small decision surface. The first deployable probes are
negative: source-only numeric sidecars recover `0/6` clean IDs with only
`26/32` source numeric coverage, and source-hidden ridge probes over last-layer
or all-layer summaries also recover `0/6` clean IDs. Do not widen to larger
slices until a deployable source-derived method clears this exact surface with
full source-destroying controls.

Same-family fallback update: a richer C2C prefill residual projection probe
does not rescue the C2C-mechanism distillation branch. Signed residual
projections reach matched `13/32`, but zero-source, label-shuffle, and
target-only controls reach `14/32`, and clean source-necessary recovery remains
`0/6`. Do not scale C2C summary/projection features without a new token/layer
local objective and anti-cache control.

Latest source-surface update: `Qwen/Qwen2.5-Math-1.5B-Instruct ->
Qwen/Qwen3-0.6B` is weaker than the non-instruct Math source on the frozen
SVAMP32 exact-ID slice. It has only `2` source-only-over-target IDs and should
not receive C2C/sidecar spend. The next gate should not be another adjacent
prompt/source variant; it should be a materially different rate-capped source
interface, such as the smallest real-model sequence-aligned sparse/anchor
sidecar smoke inspired by the quotient/GPA toy results, with zero-source,
shuffle, target-only, and slots-only controls from the start.

Sparse-anchor sidecar update: the first real-model smoke for that branch is
negative. A random sparse anchor projection plus tokenizer-boundary sidecar
reaches `9/32` with `0/6` clean C2C-headroom IDs and one slots-only clean
control hit at an estimated `34` bytes/example. A constrained `14`
bytes/example variant reaches only `7/32`, below the target floor, with `0/6`
clean recoveries. Do not tune this exact projection/top-k implementation
further; the branch only remains live if the feature extractor changes to
fold-local token/span sparse dictionaries or an existing real SAE-adapter lane
is evaluated under the same clean target set and controls.

Target-safe selector update: the dynalign/query-pool selector branch is now
killed on the strict SVAMP32 gate. The target-safe candidate oracle reaches
`18/32` but only `1/6` clean residual IDs, below the required `2/6`, and the
matching source-destroying control oracle also reaches `18/32` with `1/6` clean
residual IDs. The next live branch should be a genuinely learned communication
protocol, starting with a minimal target-conditioned soft-token or learned-query
connector trained against the C2C-over-target_self residual surface with zero,
shuffle, target-only, and slots-only controls from the start.

Qwen2.5-Math learned-connector update: the first target-conditioned
Perceiver/query-innovation checkpoint on the current Qwen2.5-Math -> Qwen3
SVAMP32 C2C-headroom surface is also negative before generation. Calibration
completed with answer-teacher and zero/shuffle plus target/slots anti-memory
controls, but teacher-forced diagnostics at gates `0.125`, `0.15`, and `0.20`
all recover `0/6` matched-only clean residual IDs. Two clean IDs have positive
matched margins, but both are explained by shuffled-source controls. Do not
tune fixed gate, positive weight, answer-teacher weight, or anti-memory weight
on this exact Perceiver memory architecture. The next learned branch needs a
materially different target-query-to-source bottleneck with target-only
learned-prefix and slots-only prefix controls at matched byte/query budgets.

Target-query source-bottleneck update: the first implemented version of that
materially different branch is also negative. The cross-fitted diagnostic uses
target prompt states as queries over source token states and adds zero-source,
shuffled-source, label-shuffle, same-norm-noise, target-only-prefix,
projected-soft-prompt, target-only, and slots-only controls. It reaches only
matched `7/32` versus target-only `8/32`, recovers `0/6` clean residual IDs,
and has no clean source-necessary wins. This kills residue-classifier/readout
variants on the current SVAMP32 C2C-headroom surface. A future learned branch
must train a true source-conditioned soft-prefix or gated cross-attention
objective directly on gold-vs-distractor logprob, not another residue
classifier.

Qwen2.5-Math source-token query-bottleneck update: the non-duplicative all-layer
token bottleneck on the current Math SVAMP32 clean C2C-headroom surface also
fails. It reaches matched `8/32`, exactly the target floor, recovers `0/6`
clean C2C-headroom IDs, and a slots-only control recovers one clean ID. This
kills shallow source-token/source-summary residue prediction on the current
surface unless the feature extractor or objective changes materially.

Fold-local token/span dictionary update: the stricter dictionary version also
fails on the same surface. It has healthy codebook telemetry, with dead atom
rate `0.0000` and mean perplexity `28.5363`, but reaches only `7/32`, below
the `8/32` target floor, and recovers `0/6` clean C2C-headroom IDs. This kills
the current source-readout / sparse-dictionary family on this surface. The next
live branch should move to target-safe output-aware dynalign selector or repair,
not more dictionary/top-k/byte-budget tuning.

Qwen-Math token/layer local follow-up: the new C2C tail-token local residual
query-bottleneck gate also fails. It records per-projector key/value `source`,
`target`, `output`, and `delta` tail tensors, reshaped as `224` tokens of width
`1024`, but matched remains `8/32`, target-only is `8/32`, clean
source-necessary recovery is `0/6`, and slots-only controls recover one clean
ID. This kills C2C summary/projection/tail-local mechanism readouts as a live
branch on this surface unless the supervision objective changes.

Positive strict-small update: switching from C2C-only targets to
source-contrastive targets produces the first live positive source-derived row.
Qwen2.5-Math source-alone has `5` source-only wins over target and `4` clean
source-only wins after excluding text relay. A target/text agreement guard plus
1-byte source residue sidecar reaches `11/32` versus target `8/32`, text relay
`8/32`, and source-alone `6/32`, with `3/4` clean source-necessary recoveries
and `0/4` clean control leakage under zero-source, shuffled-source,
label-shuffle, same-norm noise, target-only, and slots-only controls. This is
not ICLR-ready: it is one 32-example slice, uses text relay as a preservation
guard, lacks paired uncertainty/seed stability, and remains below C2C `15/32`.
Promote to SVAMP70 medium confirmation before any broader claims.

SVAMP70 medium update: the same source-contrastive sidecar stack remains
positive against target/text but is not a headline result. Baselines are target
`21/70`, text relay `22/70`, source-alone `13/70`, and C2C `31/70`. The
guarded sidecar reaches `25/70`, recovers `4/6` clean source-only IDs, and has
`0/6` clean control leakage. Paired deltas are `+0.0571` versus target and
`+0.0429` versus text, but both bootstrap intervals cross zero; versus C2C it
is `-0.0857`. A naive C2C-fallback stack fails (`23/70` with `4/6` clean
control leakage). Do not widen this row to 500 examples until the method either
gets closer to C2C or replaces the text-relay guard with a cheaper preservation
signal that creates a clear systems tradeoff.

Textless guard update: replacing the text-relay agreement guard with a
source/target-only decoded-length guard improves the medium row to `26/70`:
apply the 1-byte source residue sidecar only when the source produces a numeric
prediction and its decoded output is shorter than the target output. This keeps
`4/6` clean source-necessary IDs and `0/6` clean control leakage. Paired delta
versus target is `+0.0714` with bootstrap `[+0.0000, +0.1429]`; versus text it
is `+0.0571` with bootstrap crossing zero; versus C2C it is `-0.0714`. This is
the better systems branch because it removes text relay, but it needs
replication because the length guard is brittle.

Holdout replication update: the fixed length-ratio guard fails on a disjoint
SVAMP70 slice (`chal-101` through `chal-170`). Baselines are source `8/70`,
target `8/70`, text relay `18/70`, and C2C `37/70`. The parameterized
`source_target_len_ratio <= 1.0` sidecar reaches only `10/70`, with clean
source-necessary `0/2` and clean control union `2/2`. This weakens the fixed
hand guard as a live method. Do not scale it directly to 500 examples; the next
live branch must use a learned or cross-validated router, or first discover a
source surface with more clean source-only IDs.

Finalish guard holdout update: the alternative fixed source-quality guard
`finalish_short_numeric` also fails on the same disjoint holdout. Its best
1-byte sidecar row reaches only `9/70`, with clean source-necessary `0/2` and
clean control union `2/2`. This prunes fixed source-quality guarded source
sidecars as the live method family; do not tune thresholds or moduli without a
new router feature family and a frozen holdout gate.

CV router update: a 5-fold decision-stump router over existing source/target
JSONL features can reproduce the original SVAMP70 sidecar row (`25/70`,
`4/6` clean source-necessary, `0/6` clean control union), but it fails the
same disjoint holdout (`6/70`, `0/2` clean source-necessary). This weakens
shallow decoded-feature routers as a robust method and points the next live
branch toward source-surface discovery or stronger source-derived signals.

Source-trace router update: the richer live-CV source-trace router over valid
equations, prompt-number coverage, and source-answer reuse also fails. It has
no standard clean-control leakage, but live CV reaches only `20/70` with `1`
clean source-necessary ID and `2` accepted harms; frozen holdout reaches
`10/70` with `1` clean source-necessary ID, and that ID survives
equation-result permutation. This prunes shallow source-text quality features
as the next rescue for the sidecar branch.

Source-internal diagnostics update: a new sidecar collector can rerun source
generation only and record greedy-generation confidence features, including
chosen-token logprob, entropy, top-1 probability, and top-1/top-2 logit margin.
The two-example MPS smoke passed outside the sandbox with offline caches. This
is the next router feature family to test before another decoded-text guard.

Source-internal confidence router update: live SVAMP70 confidence routing is
clean but too weak (`24/70`, `2` clean source-necessary, `0` clean control
union), and the frozen full-live rule fails the disjoint holdout (`7/70`, `0`
clean source-necessary, `1` accepted harm). This prunes the current confidence
router on the old source-sidecar surface; the next gate is disjoint source
surface discovery, not multi-feature tuning on this slice.

Surface scout update: SVAMP `chal-171` through `chal-240` is not a useful
sidecar decision surface. Source is `8/70`, target is `22/70`, text relay is
`24/70`, source-only over target is only `2`, and clean source-only after text
exclusion is only `1`. Do not spend C2C on this slice for the current branch.

Second surface scout update: SVAMP `chal-241` through `chal-310` has nonzero
clean source mass but still does not clear the predefined surface gate. Source
is `5/70`, target is `10/70`, text relay is `14/70`, source-only over target is
`4`, and clean source-only after text exclusion is `4`. Because raw source-only
is below the `>=6/70` gate and text is much stronger than source, do not spend
C2C here. The cheap sidecar gate now confirms the rejection: the text-relay
agreement guard reaches only `9/70` with clean control leakage, and the
textless shorter-than-target guard reaches only `11/70` with `1/4` clean
source-necessary and `1/4` clean control leakage. The next live gate is GSM70
Math source-surface discovery, while the highest-value method branch after a
surface clears is a rate-capped query/resampler or shared sparse source sidecar
rather than another shallow decoded-feature router.

GSM70 source-surface update: Qwen2.5-Math -> Qwen3 on `data/gsm8k_eval_70.jsonl`
also fails the surface gate. Source is `3/70`, target is `4/70`, text relay is
`6/70`, source-only over target is `3`, and clean source-only after text
exclusion is only `2`. Do not spend C2C or sidecar compute on this slice. The
current live branch should move from more same-pair surface scouting to the
smallest stronger-interface smoke on an existing exact-ID SVAMP surface, or to
a different source/target pair only if a cheap source/target/text scout clears.

Third adjacent SVAMP surface scout update: SVAMP `chal-311` through `chal-380`
also fails the surface gate. Source is `8/70`, target is `21/70`, text relay is
`19/70`, source-only over target is only `3`, and clean source-only after text
exclusion is only `2`. This makes three adjacent same-pair SVAMP70 scouts with
insufficient clean source mass (`chal171-240`, `chal241-310`, and
`chal311-380`). Stop adjacent SVAMP range scouting for Qwen2.5-Math -> Qwen3
unless a new source encoder or prompting hypothesis changes the surface.

Process-repair source-control update: old held-out process-repair rows were
re-audited because they were the strongest historical positive-looking result.
On SVAMP70, matched process repair reaches `38/70`, target self-repair reaches
`35/70`, and matched has `3` wins over target self-repair. The zero-source K/V
control reaches `35/70` and overlaps `1/3` of those matched-only IDs. The
shuffled-source prompt control reaches `37/70` and overlaps `3/3`. The combined
gate therefore has `0` source-specific matched-only IDs after controls. Kill
process-repair selected routes as a source-communication method on this
surface; keep it only as a target-side repair/candidate-diversity baseline. The
next live method branch should be a true source-conditioned soft-prefix or
gated cross-attention logprob objective with matched target-only-prefix,
slots-only, projected-soft-prompt, zero-source, and shuffled-source controls
before generation.

Target-CE prefix-generation update: the proposed true continuation-loss rescue
of that soft-prefix/cross-attention family has now been run and failed. On the
SVAMP32 C2C-headroom surface it gives `0/6` matched-only clean IDs in logprob
and matched generation is weaker than every decoded source-destroying or
target-only control on the six clean IDs. Do not continue tiny prefix-emitter
tuning. The next exact gate is a reusable `latent_bridge` query-innovation
resampler audit for whether true LM CE and generation scoring can be attached
to a larger source-memory interface without a high-risk translator refactor.

Historical source-contrastive promotion rule:

- matched `>=9/32`
- clean source-necessary `>=2/4`
- source numeric coverage `>=26/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, same-norm noise, target-only,
  and slots-only controls have clean union `0/4`

The branch cleared this strict-small source-control surface, but later medium
and disjoint-surface gates showed that the row is unstable. Do not widen it to
500 examples or cross-family benchmarks unless a stronger source surface or
router changes the holdout behavior.

## Engineering Follow-Ups

- Add `pytest -m integration` for tiny cached HF runs covering calibration,
  evaluation, C2C, and KVComm wrappers.
- Replace first-N slices with manifest-backed exact-ID slices including input
  hash, ordered IDs, command, seed, model revision, and output hashes.
- Make the source-control matrix mandatory for every promoted positive row.
- Unify root/package requirements and default local commands around
  `venv_arm64` on this machine.
- Build one tracked artifact manifest for decisive predictions/checkpoints,
  including external repo commits and model revisions.

## Review Inputs

- Local ledger and memos in `paper/`
- Implementation stack in `latent_bridge/`, `scripts/`, and `tests/`
- Experiment artifacts under `results/`, `checkpoints/`, and `.debug/`
- Reference state under `references/`
- Three folder-level subagent audits: paper/story, code/eval, and artifacts

## 2026-04-27 Readiness Update

Current readiness: not ICLR-ready. Estimated distance remains substantial:
the project has headroom surfaces and strong negative controls, but still no
positive communication method that survives disjoint source-destroying controls.

Current live branch: `source_likelihood_sketch` on
`svamp70_live_source` with `svamp70_holdout_source` as frozen validation.

Why this branch is live:

- it is materially different from the killed decoded-feature routers,
  process-repair rows, tiny source-prefix emitters, query-memory rows, and
  Perceiver target-memory rows
- it treats the target candidate pool as decoder side information and sends
  only a rate-capped source-model likelihood preference
- it has a crisp source-control gate: zero-source, shuffled-source,
  label-shuffle, target-only, and slots-only controls must recover zero clean
  source-only IDs

Submission blocker:

- no scientific result yet for this branch because the machine still has an
  orphaned MPS `scripts/calibrate.py` process, PID `31103`, with `STAT=UE`
- do not launch more MPS work until PID `31103` is cleared

Next exact gate:

- run the live and holdout sketch collection plus frozen analyzer commands in
  `paper/svamp70_source_likelihood_sketch_20260427.md`
- promote only if live CV and frozen holdout both clear the predefined pass
  rule; otherwise weaken or kill the branch and move to the next source-surface
  or stronger-interface candidate

## 2026-04-27 Historical Positive Audit Update

The old `rotalign`, `latent_bridge`, and results-folder positives were
re-audited before changing branch priority. The audit is recorded in
`paper/historical_positive_branch_audit_20260427.md`.

Conclusion:

- raw GSM70 dynalign remains a real mechanism clue but is killed as the live
  method because finite repeat seeds do not preserve the seed0 lift
- query-memory and Perceiver target-memory checkpoints stay killed because the
  clean-ID answer-likelihood expansion fails against source-destroying and
  target/slots controls
- process repair remains a target-side baseline/confound, not communication
- the strongest historical direction is the side-information family:
  source-contrastive sidecar plus the C2C-derived syndrome bound

This audit supports keeping `source_likelihood_sketch` as the top branch. It
is the smallest non-duplicative test of whether a source-derived, rate-capped
candidate preference can keep the SVAMP70 live signal while avoiding the
holdout leakage that killed fixed decoded guards.

## 2026-04-27 Collector Hardening Update

The `source_likelihood_sketch` branch remains the live branch. The collector
now supports `--limit` and `--resume` and records command, commit, input
hashes, ordered IDs, ordered-ID hash, and output hash in its markdown readout.

Next gate after PID `31103` is cleared:

1. run the two-example `--limit 2` smoke in
   `paper/svamp70_source_likelihood_sketch_20260427.md`
2. if finite, run the full live and holdout collection commands with `--resume`
3. run the frozen live-to-holdout analyzer

This does not change readiness: still not ICLR-ready until the scientific
live/holdout gate clears.

## 2026-04-27 CPU Smoke Under MPS Blocker

The `source_likelihood_sketch` collector passed a CPU-only two-example smoke
while PID `31103` continued blocking MPS:

- output JSONL:
  `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.jsonl`
- output JSONL sha256:
  `863254ecc5110eab3e62efb65ddb31e9472be42513bce6ce1ab44842e1057e9d`
- rows: `2`
- elapsed: `96.06s`
- top labels: `text`, `text`

Readiness impact:

- Tooling risk is lower because the collector can load the source model,
  score continuations, append JSONL rows, and emit provenance/hashes.
- Scientific readiness is unchanged. This is only a micro smoke, and the full
  live/holdout gate still needs MPS after PID `31103` is cleared.

## 2026-04-27 Source Likelihood Sketch Kill

Readiness remains not ICLR-ready.

The `source_likelihood_sketch` live branch is killed on the Qwen2.5-Math ->
Qwen3 SVAMP70 live/holdout surface:

- bare normalized answer mean/sum variants fail live and holdout
- formatted `Answer: {text}` mean logprob has an interesting holdout pass
  (`10/70`, clean source-necessary `2`, control union `0`) but fails live CV
  (`20/70`, clean source-necessary `0`, control union `1`)
- formatted sum-logprob fails live and holdout

The next selected branch is not another likelihood sketch. Post-kill syndrome
bound replays now show that a richer predictor is not justified on this exact
SVAMP70 live/holdout surface: C2C-teacher residues have live headroom but fail
holdout controls, while source-teacher residues recover live clean IDs only by
destroying target-self preservation.

The source-trace router scout also failed and should not be promoted.

Current live branch: none. The next branch is source-surface discovery for a
stronger surface, followed by a bound replay before implementing another
predictor. Stop MPS execution until PID `31103` is cleared; CPU is acceptable
only for tiny smoke/debug work.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

## 2026-04-27 Durable Source-Surface Ranking

Readiness remains not ICLR-ready. Current live method branch: none.

The source-surface selection gate is now durable:

- New ranker: `scripts/rank_source_contrastive_target_sets.py`
- Focused tests: `tests/test_rank_source_contrastive_target_sets.py`
- Output: `results/durable_source_surface_ranking_20260427/source_surface_ranking.json`
- Focused memo: `paper/durable_source_surface_ranking_20260427.md`

The ranker consumes existing `source_contrastive_target_set.json` artifacts and
ranks by clean source-only IDs after controls/baselines, not raw source-only
counts.

Decision:

- `svamp70_live` is the primary next method surface: clean source-only `6/70`,
  raw source-only `9/70`, target/source oracle gain `9/70`.
- `svamp70_holdout` remains the canonical replay gate despite only `2/70`
  clean source-only IDs.
- `svamp70_chal241_310` is only an adjacent falsifier with clean `4/70`.

Recent latent-agent communication references were added in
`references/469_recent_latent_agent_communication_refs.md`. They raise the
baseline bar: the next learned branch should include fixed-budget latent or
activation communication baselines and systems metrics, not only text relay.

Updated next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source MPS scout from
`paper/postkill_historical_cpu_audit_20260427.md`. If it reaches at least six
clean source-only IDs and target/source oracle gain of at least six, run a
zero-init gated latent side-information smoke on `svamp70_live` with
source-destroying controls and activation/latent baselines.

## 2026-04-27 Source-Hidden Query And KVComm Smoke

Readiness remains not ICLR-ready. Current live method branch: none.

CPU evidence weakened direct source-hidden query bottlenecks:

- Command: `scripts/analyze_svamp32_source_latent_syndrome_probe.py` with
  `--probe-model query_bottleneck`, `--query-epochs 2`, `--query-slots 4`,
  `--feature-layers last`, and `--device cpu`.
- Result: `source_latent_syndrome_probe_fails_gate`.
- Matched: `11/32`.
- Zero-source/shuffled-source/label-shuffled/target-only: `14/32`.
- Clean source-necessary IDs: `0`.

CPU tooling smoke for KVComm passed via module invocation:

- Command form: `./venv_arm64/bin/python -m latent_bridge.kvcomm_eval ...`
- One-example CPU smoke wrote `.debug/kvcomm_cpu_smoke_20260427/`.
- Direct script invocation initially failed with `ModuleNotFoundError`; fixed
  `latent_bridge/kvcomm_eval.py` to bootstrap the repo root onto `sys.path`.

Reference update:

- Added `references/470_kv_cache_latent_communication_baselines_refs.md`.
- C2C/KVComm now define the next baseline contract for fixed-budget,
  target-preserving cache communication.

Readiness impact:

- Weakened: direct source-hidden query-bottleneck syndrome readouts.
- Promoted: fixed-budget KV/cache communication baseline as the next executable
  MPS branch after PID `31103` clears.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If clear, run a one-example MPS KVComm smoke or the stronger-source MPS scout,
then scale only if exact ID parity, numeric coverage, and source-destroying
controls are preserved.

If clear, do not run the old `chal311_380` scout recorded in
`paper/svamp70_syndrome_bounds_after_sketch_kill_20260427.md`; those artifacts
already exist and fail the surface gate.

## 2026-04-27 Post-Kill Historical And CPU Audit

Readiness remains not ICLR-ready. Current live branch: none.

The historical positive audit was extended after the `source_likelihood_sketch`
and post-sketch syndrome-bound kills:

- `dynalign_module_replace_residrank16` remains a mechanism clue only; seed
  stability fails and finite repeats do not preserve the seed-0 lift.
- ID-weighted query innovation remains a useful single-ID clue, but still
  recovers only `1/6` clean IDs and does not preserve target-self repair.
- Perceiver/query-memory checkpoints remain killed after six-clean-ID source
  controls.
- Source-contrastive sidecar remains the best historical formulation clue, but
  shallow source-text feature routing does not rescue weak adjacent surfaces.

New CPU-only evidence:

- chal241-310 post-kill source-sidecar CV router fails: best row matches
  `10/70`, clean source-necessary `1`, control clean union `0`, accepted harm
  `1`.
- consolidated existing-surface scan shows `chal311_380` is already available
  and weak: target `21/70`, source `8/70`, source-only `3`, oracle `24/70`.
- existing-artifact CPU mining is exhausted; no remaining CPU-only command can
  promote a positive method.

Focused memo:

- `paper/postkill_historical_cpu_audit_20260427.md`

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, the next MPS command should be a genuinely new
stronger-source scout:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Only run C2C or learned connector work if that scout has ordered ID parity,
high numeric coverage, source-only over target at least `6/70`, and
target-or-source oracle at least target plus `6/70`.

## 2026-04-27 Creative Reference Synthesis

Readiness remains not ICLR-ready. Current live branch: none.

The reference corpus was converted to markdown under
`references/pdf_markdown/` so subagents can inspect the PDF library directly.
The literature sweep changes the next branch priority:

- Candidate-syndrome decoding is promoted as the top CPU-feasible branch:
  source emits a tiny code over target-side candidates, and the target decodes
  against its own side information.
- Zero-init gated query bottlenecks are promoted as the next learned branch
  after MPS clears, because target-self preservation must be built into the
  interface.
- RotAlign/latent-bridge ideas are revived only under anchor-relative sparse
  difference atoms with explicit source-difference zeroing controls.
- Protected-tail quantized residuals are deferred until there is a real
  source-necessary signal worth compressing.

Exact blocker remains a missing positive method plus the local MPS blocker
PID `31103`.

CPU gate result:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir results/candidate_syndrome_decoder_20260427 \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --run-date 2026-04-27
```

Status: `candidate_syndrome_decoder_fails_smoke`.

- Live matched clean source-necessary `1`, target-self harms `17`, control
  clean union `0`.
- Holdout matched clean source-necessary `4`, target-self harms `14`, control
  clean union `0`.

Decision: do not promote the numeric hash-syndrome artifact probe. The next
highest-value branch is zero-init gated query bottlenecks, gated by MPS cleanup
and/or a stronger source surface.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

## 2026-04-27 Byte-Efficient Side-Information Audit

Readiness remains not ICLR-ready. Current live branch: none.

The latest historical/result audit demotes several positive-looking rows to
mechanism clues:

- Sparse K-only and cosine transport showed narrow GSM positives but failed
  later seed stability.
- Raw RotAlign/DynAlign remains killed as a method because seed stability and
  nonfinite checkpoint issues persist.
- Source-contrastive sidecar and candidate-pool syndrome probes remain the best
  low-byte clues, but the old guards/residues are not deployable source
  methods.
- Perceiver/query-memory, shallow source-likelihood, semantic-predicate, and
  numeric hash-syndrome variants are killed on current surfaces.

Updated top branch:

- Learned source-derived syndrome/innovation sidecar decoded against target
  candidate/cache side information.

Required baselines/controls:

- target-alone, source-alone, text/token relay, C2C, KVComm, Q-KVComm-style
  quantized KV where feasible, DroidSpeak-style same-architecture cache reuse
  as a threat model, and target self-repair.
- zero-source, shuffled-source, random same-byte sidecar, target-only,
  slots-only, source-answer-overlap checks, exact-ID parity, numeric coverage,
  paired uncertainty, bytes, latency, generated tokens, and TTFT where
  practical.

New memos:

- `paper/byte_efficient_sideinfo_branch_audit_20260427.md`
- `references/471_byte_efficient_source_sideinfo_refs.md`

Hard blocker:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

PID `31103` still remains present as a `STAT=UE`, `PPID=1` MPS
`scripts/calibrate.py` process even after user-side kill attempts. Do not start
MPS runs until it clears.

Next exact gate after PID clears: run the stronger-source MPS surface scout
recorded in `paper/postkill_historical_cpu_audit_20260427.md`; only build the
learned syndrome/innovation sidecar if that scout has ordered ID parity, high
numeric coverage, source-only over target by at least `6/70`, and
target-or-source oracle at least target plus `6/70`.
