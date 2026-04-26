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

Pivot to source-surface discovery rather than more SVAMP32 source-state
residue predictors or repair-stack tuning. The next live branch is to convert
the strongest toy interface clue, quotient/GPA sparse dictionaries with
sequence-aligned byte sidecars, into a real cross-family tokenizer/interface
stress gate.

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

CV router update: a 5-fold decision-stump router over existing source/target
JSONL features can reproduce the original SVAMP70 sidecar row (`25/70`,
`4/6` clean source-necessary, `0/6` clean control union), but it fails the
same disjoint holdout (`6/70`, `0/2` clean source-necessary). This weakens
shallow decoded-feature routers as a robust method and points the next live
branch toward source-surface discovery or stronger source-derived signals.

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

Current source-contrastive promotion rule:

- matched `>=9/32`
- clean source-necessary `>=2/4`
- source numeric coverage `>=26/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, same-norm noise, target-only,
  and slots-only controls have clean union `0/4`

The branch cleared this strict-small source-control surface. Widen only to the
medium SVAMP70 confirmation rung next; do not move to cross-family or
long-context benchmarks until medium confirmation, paired uncertainty, and
source-control replication are available.

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
