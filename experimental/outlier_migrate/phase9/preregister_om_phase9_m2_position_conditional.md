# Phase 9 M2 Position-Conditional Union Preregistration

**Frozen on**: 2026-05-12
**Frozen by**: Codex GPU swarm, under `swarm/goal.md` Phase 9 authorization
**Status**: Frozen after Phase 9 Step 9.0 and before any M2 quantized
scoring run.

## Terminology

This preregistration uses **decode-position channel drift** or
**long-decode channel drift** for the measured phenomenon. The phrase
disallowed by `swarm/goal.md` is intentionally not used in new Phase 9
text.

## Phase 9 Step 9.0 precondition

M2 is authorized because Step 9.0 passed the premise gate:

- Granite-4-H-Small strict set-leaving: `0.566234756098`
- Nemotron-3-Nano strict set-leaving: `0.533713200380`
- DeepSeek-R1-Distill-Qwen-1.5B strict set-leaving: `0.670572916667`

All three are above the required `0.50` threshold. Adjacent-bin top-1%
Jaccard overlap averaged `0.745999647119`, which keeps position-binned
methods in scope.

## Central hypothesis

A fixed protected channel set is insufficient in long-decode reasoning
because top-channel identities drift with absolute decode position.
However, the Step 9.0 bin-overlap result indicates that adjacent decode
regions have enough continuity for a small number of position-conditioned
protected channel sets to recover a measurable fraction of the
BF16-vs-static-protection gap.

## Prior-art differentiation

PMPD (Progressive Mixed-Precision Decoding, arXiv 2410.13461) states:
"PMPD builds upon the observation that the prefill phase and the initial
tokens of the decoding phase are more sensitive to approximations than
later tokens."

M2 tests the opposite pressure in long reasoning traces: later decode
positions can require different protected channels because the top-channel
set leaves its initial membership. PMPD gradually lowers precision deeper
in the generated sequence; M2 keeps the same per-position budget but
switches which channels are protected as decode position changes.

KL Lens studies layer-stratified mixed precision in hybrid
SSM-Transformer models. M2 is channel-level and position-conditioned, not
layer-level and sensitivity-static.

TTQ performs prompt-level online activation-aware quantization. M2 is not
once-per-prompt calibration: it uses preregistered absolute decode-position
bins and switches protected channel sets during decode.

DecDEC is not implemented as a baseline in Phase 9. Per `swarm/goal.md`,
the paper will use a top-K-by-magnitude reactive BF16 oracle as a
strictly stronger upper bound on DecDEC-style reactive selection.

## Models

Run M2 in this priority order:

1. `ibm-granite/granite-4.0-h-small`
2. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
4. `tiiuae/Falcon-H1-0.5B-Instruct` if Phase 7 landed before M2 execution

If Granite-4-H-Small returns a kill, still run Nemotron-3-Nano once before
deciding whether to skip remaining models. A two-model kill is more useful
than a single-model kill.

If M2 passes on at least two of the first three models, run the
reviewer-objection-preempting larger Transformer:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

The 7B run requires a separate preregistration before execution.

## Trace set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic prompt indices `0-23`
- Prompt SHA-256:
  `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`

## Quantization and scoring

- Weight quantization: simple symmetric per-channel INT4.
- Activations: FP16.
- Protected channels: BF16.
- AWQ-style activation-aware scaling is not used.
- Decode is deterministic greedy decoding.
- Scoring window follows the existing Phase 4 intervention packet for
  Granite-4-H-Small and the analogous fixed-window scoring protocol for
  other models.

## M2 protected sets

M2 uses three protected channel sets selected from calibration activations:

- Set A: union of top-1% channels at positions `{100, 200, 500}`
- Set B: union of top-1% channels at positions `{1000, 2000, 5000}`
- Set C: union of top-1% channels at positions `{7000, 10000, 15000}`

Decode switching rule:

- Use Set A through decode position `799`
- Switch A -> B at decode position `800`
- Switch B -> C at decode position `6000`

Each active set has approximately a top-1% per-layer channel budget. Across
the decode trajectory, the union may contain up to approximately 3%, but the
active protected set at any decode position remains a single position-bin set.

If a calibration position is unavailable from an existing characterization
packet, the M2 runner must generate calibration activations for that position
before quantized scoring and record them in the result packet. It may not
substitute different positions after observing M2 recovery values.

## Regimes

Each model packet must evaluate:

1. BF16 baseline
2. Static-1% baseline from position `100`
3. M2 position-conditional union
4. Static-3% matched-cost control
5. Random-bin assignment negative control

The random-bin control uses the same three sets as M2 but permutes the
decode-position assignment using bootstrap seed `20260513`; it tests whether
position-conditioned assignment matters beyond set size.

## Metric

For each trace:

`recovery = 1 - (perplexity_method - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)`

Only traces with a positive recoverable static gap are included in the
primary recovery median. The packet must also report:

- count and fraction of no-recoverable-static-gap traces
- recovery for every trace
- median recovery for M2, static-3%, and random-bin control
- bootstrap 95% CI over traces with seed `20260513`

## Decision rule

### PASS_M2_POSITION_CONDITIONAL

Return PASS if all of the following hold:

1. M2 median recovery is at least `0.40`.
2. M2 bootstrap CI lower bound is greater than `0.20`.
3. M2 beats static-1% by at least `0.20` median recovery.
4. M2 beats static-3% matched-cost control by at least `0.10` median
   recovery.
5. M2 beats random-bin assignment control by at least `0.15` median
   recovery.

### KILL_M2_NO_IMPROVEMENT

Return this kill if M2 median recovery is within `0.05` of static-1%.

### KILL_M2_RANDOM_CONTROL_BEATS

Return this kill and stop the method on the current model if the
random-bin negative control beats M2 by more than `0.10` median recovery.

### KILL_M2_AMBIGUOUS

Return ambiguous kill for middle outcomes with overlapping confidence
intervals or insufficient separation from controls.

### FAIL_INFRA_M2

Return infrastructure failure for model load failure, incomplete packet,
OOM that cannot be fixed by reducing batch size, missing required
artifacts, or checker failure that prevents applying the decision rule.

## Required artifacts

Each M2 result packet must contain:

- environment snapshot
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- calibration activation manifest
- protected-set manifest for Sets A/B/C, static-1%, static-3%, and
  random-bin assignment
- per-trace perplexity table for all regimes
- per-trace recovery table
- metrics JSON with bootstrap CIs
- checker result and artifact check
- artifact hashes

## Forbidden actions

- Modifying prior preregistration files.
- Adjusting M2 thresholds after observing M2 data.
- Replacing the fixed A/B/C position grids after observing M2 data.
- Omitting static-3% matched-cost control.
- Omitting random-bin negative control.
- Reporting a positive result without all matched-cost and random controls.
- Using SGLang for Phase 9.
- Modifying model source code.

## Paper integration rule

If M2 passes, the paper may frame position-conditioned protection as the
first positive Phase 9 method and proceed to M10 as an orthogonal decision
surface. If M2 kills on two models, the paper should report that simple
position-conditioned set switching is insufficient and use M10/M9 to test
whether scale tables or learned prediction capture what M2 misses.
