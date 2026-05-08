# Residual Migration Phase 1 Diagnostic

Date: 2026-05-08

Killed entry: `residual_migration_phase1`

Decision: `KILL_RM_PHASE1_FAILED_AT_SCALE`

Result packet:
`experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z`

## Decision

The formal gate result is `KILL_RM_PHASE1_FAILED_AT_SCALE`.

The preregistered replication path required AIME-2025 accuracy drop `<0.015`
with CI upper bound `<0.015`. The observed full-ablation drop was
`0.08333333333333333`, with bootstrap CI95
`[0.0, 0.20833333333333334]`, so the checker correctly killed the branch.

Primary artifacts:

- `experimental/residual_migration/phase1/preregister_rm_phase1.md`
- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/checker_result.json`
- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/metrics.json`
- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/stratified_metrics.json`
- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/headroom_diagnostics.json`
- `experimental/residual_migration/phase1/results/rm_phase1_20260508T204839Z/artifact_check.json`

## Proximate Failure Classification

Classification: mixed, but primarily experimental decision-surface
insufficiency for mechanistic interpretation, with formal hypothesis failure
for the preregistered gate.

This is not an infrastructure issue. `artifact_check.json` reports
`artifact_complete: true`, required files are present, and the checker produced
the expected decision string.

This is not preregistration ambiguity. The preregistered kill rule is explicit
for the replication path: drop `>=1.5%` kills the branch. The measured drop is
`8.333333333333333%`.

However, the packet is weak evidence about the deeper residual-migration
hypothesis because baseline headroom is low:

- baseline accuracy: `2/24 = 0.08333333333333333`
- full ablation accuracy: `0/24`
- lenient oracle accuracy: `3/24 = 0.125`
- extractor failures: `3`
- headroom status: `LOW_BASELINE_HEADROOM`

The formal gate is dead, but the mechanism is underdetermined. Losing 2 correct
answers out of 24 violates the preregistered threshold, yet the target model
barely solves the slice. Therefore this packet cannot support a strong claim
that residual migration fails in general; it supports the narrower claim that
this Phase 1 gate did not replicate a robust low-drop effect at scale.

## Packet Patterns

The most important pattern is uniformity across ablations. Every
layer-stratified condition reports the same drop:

- `full_ablation`: `0.08333333333333333`
- `first_half`: `0.08333333333333333`
- `second_half`: `0.08333333333333333`
- `attention_only`: `0.08333333333333333`
- `mamba_only`: `0.08333333333333333`

This argues against a clean layer-local attribution story. The packet does not
identify a first-half, second-half, attention, or Mamba locus. Instead, every
tested clipping subset destroys the same two baseline-correct items:

- `opencompass_AIME2025_I_0`: baseline `70`, canonical `70`, ablated
  incorrect/null
- `opencompass_AIME2025_II_1`: baseline `49`, canonical `49`, ablated
  incorrect/null

A third baseline generation mentions the canonical answer under the lenient
oracle but extracts incorrectly:

- `opencompass_AIME2025_I_13`: canonical `60`, extracted `59`, generated text
  contains canonical

That pattern suggests the ablation may be perturbing answer emission or
extraction-relevant finalization, not isolating reusable cross-model residual
content. Because all ablated extracted answers are null for the
baseline-correct cases, this looks more like brittle decode/output degradation
than a clean test of transferable reasoning state.

## Setup Versus Hypothesis

For the preregistered Phase 1 claim, the hypothesis is wrong: the method did
not preserve accuracy under the fixed 95th-percentile clipping gate.

For scientific diagnosis, the setup is insufficient to decide the broader
mechanism. A 24-prompt AIME slice with only 2 baseline-correct examples has too
little positive support to distinguish:

- true residual dependence of rare successful solutions,
- generic decode destabilization from clipping,
- answer-format or extractor collapse,
- task/model mismatch where Granite-4.0-H-Small has inadequate AIME headroom.

The conservative conclusion is: kill Residual Migration Phase 1 as a
positive-method branch under this preregistration; do not claim a general
negative theorem about residual transfer.

## Ruled Out, Weakened, Still Alive

Ruled out:

- The preregistered claim that the Phase 0 low-drop effect replicates at scale
  on this AIME-2025/Granite-4.0-H-Small gate.
- Layer-stratified attribution from this packet; all stratified drops are
  identical.

Weakened:

- A broad residual-clipping robustness story. The only baseline-correct prompts
  both fail under ablation.

Still alive:

- Positive methods that avoid destructive clipping and instead use
  residual-derived signals as selective communication, routing, stabilization,
  or verification.
- Methods tested on a decision surface with enough baseline headroom to
  separate communication from floor effects.

## Fresh Positive-Method Pivot Hypotheses

### 1. Critical-Coordinate Residual Signaling

Hypothesis: successful target reasoning depends on a sparse set of residual
coordinates that are fragile under clipping. Instead of clipping or migrating
full residual outliers, transmit a compact critical-coordinate mask or low-rank
salience sketch from source to target and use it to preserve or boost only those
coordinates during decode.

Plausibility: medium. The uniform loss of the two correct items under every
ablation suggests rare successful trajectories may rely on sparse fragile
channels. A method that preserves rather than removes those channels is a
cleaner positive-method pivot.

Potential paper claim: compact residual salience signals can improve
cross-model reasoning preservation with fewer bytes than full hidden-state
transfer.

COLM competitiveness: medium if it shows reproducible gains, seed stability,
and cross-family transfer. Not enough if it only recovers the same two AIME
examples.

### 2. Answer-Finalization Residual Bridge

Hypothesis: the main failure mode is not reasoning transfer but final answer
stabilization. A source or baseline pass can provide a small residual-space
anchor at answer-finalization positions, improving exact-answer extraction
without transferring full chains of thought.

Plausibility: medium-low to medium. The packet shows ablated outputs lose
extractable answers, and one baseline case contains the canonical answer but
extracts the wrong value. This points to finalization brittleness. The risk is
that the method becomes an output-format patch rather than a deep latent-transfer
contribution.

Potential paper claim: cross-model residual anchors at finalization positions
improve answer faithfulness or extraction under compact latent communication.

COLM competitiveness: medium only if framed rigorously as latent finalization
transfer and tested against strong text-hint, verifier, and self-consistency
baselines.

### 3. Headroom-Gated Latent Transfer Benchmark

Hypothesis: residual transfer methods should activate only when target baseline
has demonstrable local capability headroom. A learned gate predicts when latent
intervention helps versus when the target is below floor, preventing destructive
interventions on unsolved prompts.

Plausibility: medium. The low baseline headroom is the dominant interpretability
problem in this packet. A headroom-gated method directly addresses the
floor-effect failure and could separate genuine communication from target
incapability.

Potential paper claim: capability-aware latent transfer improves robustness by
refusing interventions when the target model lacks sufficient local solvability.

COLM competitiveness: medium-high if the gate is learned from preregistered
diagnostics, improves accuracy/latency/bytes, and survives strict same-family
and cross-family controls. It must not become post-hoc prompt filtering.

## Next Exact Gate

Do not rerun this Phase 1 gate with looser thresholds, different seeds,
cherry-picked prompts, or selected layers.

The next defensible Residual-derived gate would require a fresh preregistered
pivot with:

- enough baseline headroom to interpret gains,
- a strict no-cherry-picking prompt split,
- same-family and cross-family separation,
- paired uncertainty,
- explicit controls distinguishing source communication from target-cache or
  formatting effects.
