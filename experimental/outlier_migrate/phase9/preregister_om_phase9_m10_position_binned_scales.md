# Phase 9 M10 Position-Binned Scales Preregistration

**Frozen on**: 2026-05-14
**Frozen by**: Codex GPU swarm, under `swarm/goal.md` Phase 9 authorization
**Status**: Frozen after M2 landed on Granite-4-H-Small and before any M10
scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. New Phase 9 files
must not use the terminology that `swarm/goal.md` reserves for MoBiQuant's
different precision-sensitivity phenomenon.

## Preconditions

M10 is authorized because:

1. Phase 9 Step 9.0 replicated strict set-leaving above `0.50` on
   Granite-4-H-Small, Nemotron-3-Nano, and DeepSeek-R1-Distill-Qwen-1.5B.
2. Step 9.0 found overall adjacent-bin top-1% Jaccard overlap
   `0.745999647119`, above the `0.40` threshold that keeps M10 in scope.
3. M2 landed on Granite-4-H-Small with
   `KILL_M2_RANDOM_CONTROL_BEATS`, satisfying the M10 precondition that M2
   has landed with any outcome.

## Central hypothesis

M2 showed that switching protected channel membership by decode-position bin
is insufficient. M10 tests a different decision surface: channel identities
may drift, but adjacent decode-position bins have enough continuity that
bin-specific quantization scale tables can recover a measurable fraction of
the BF16-vs-static-SmoothQuant gap without online computation.

## Prior-art differentiation

PMPD (Progressive Mixed-Precision Decoding, arXiv 2410.13461) states:
"PMPD builds upon the observation that the prefill phase and the initial
tokens of the decoding phase are more sensitive to approximations than
later tokens."

M10 tests the opposite pressure in long reasoning traces. If later decode
positions need different channel scale tables because the high-magnitude
channel set changes with position, then lowering precision monotonically with
decode depth is not sufficient for this regime. PMPD changes precision as the
sequence progresses; M10 keeps the quantization format fixed and changes the
scale table keyed by absolute decode-position bin.

KL Lens studies layer-stratified mixed precision in hybrid SSM-Transformer
models. M10 is channel-scale and decode-position-conditioned, not layer-level
and sensitivity-static.

TTQ performs prompt-level online activation-aware quantization. M10 is not
prompt-level once-per-prompt calibration and does not recompute scales online:
it precomputes a small fixed table per decode-position bin.

DecDEC is not implemented as a baseline in Phase 9. Per `swarm/goal.md`, the
paper will use a top-K-by-magnitude reactive BF16 oracle as a stricter
upper-bound comparison rather than attempting to implement DecDEC.

## Models

Run M10 in this priority order:

1. `ibm-granite/granite-4.0-h-small`
2. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
4. `tiiuae/Falcon-H1-0.5B-Instruct`

Vacation-mode priority is Granite-4-H-Small first. Additional models run only
after Granite-Small lands and the paper Draft 0 remains current.

## Trace set

- Source: AIME-2025
- Primary count: 24 traces
- Selection: deterministic prompt indices `0-23`
- Prompt SHA-256:
  `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`

If throughput is more than 2x slower than estimated or the model approaches
OOM under the default configuration, vacation mode permits a deterministic
12-trace revision using prompt indices `0-11`. The decision thresholds are not
changed; the packet must label the result as a reduced-slice vacation run.

## Position bins

M10 uses the five Step 9.0 bins:

1. `(0, 500]`
2. `(500, 2000]`
3. `(2000, 5000]`
4. `(5000, 10000]`
5. `(10000, 20000]`

At decode time, use the scale table corresponding to the current absolute
decode position. There is no online scale recomputation.

## Scale formula

For each model layer `l`, channel `c`, and bin `b`, compute:

- `A[l,b,c]`: maximum absolute activation magnitude observed for channel `c`
  over all calibration rows in bin `b`
- `W[l,c]`: maximum absolute weight magnitude over tensors in layer `l` whose
  hidden dimension corresponds to channel `c`

M10's bin-specific SmoothQuant-style scale is:

`S[l,b,c] = clamp((A[l,b,c] ** alpha) / (W[l,c] ** (1 - alpha)), 0.01, 100.0)`

with:

- `alpha = 0.5`
- `A` and `W` both clamped below by `1e-8` before exponentiation
- each layer/bin scale vector divided by its median so median scale is `1.0`

The runner may implement this by folding scales into dequantized INT4 weights
and corresponding activation hooks, or by an algebraically equivalent
framework-compatible transformation. It must record the implementation route
and exact scale tensors or hashes in the packet.

## Quantization and scoring

- Weight quantization: simple symmetric per-channel INT4 after applying the
  relevant scale table.
- Activations: FP16.
- Decode is deterministic greedy decoding.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap seed: `20260514`.

## Regimes

Each model packet must evaluate:

1. BF16 baseline
2. Static SmoothQuant scales from position `100`
3. M10 position-binned scales
4. Midpoint matched-cost control
5. Random-bin scale assignment negative control

The midpoint matched-cost control uses the same five-bin table structure as
M10, but each bin table is calibrated only from the bin's nearest recorded
midpoint position instead of all calibration positions in the bin. This tests
whether the M10 result comes from bin aggregation rather than merely storing
multiple tables.

The random-bin control uses the same five scale tables as M10 but permutes the
decode-position table assignment with seed `20260514`. This tests whether the
scale table must match decode position.

## Metric

For each trace:

`recovery = 1 - (perplexity_method - perplexity_BF16) / (perplexity_static_smoothquant - perplexity_BF16)`

Only traces with a positive recoverable static SmoothQuant gap are included in
the primary recovery median. The packet must also report:

- count and fraction of no-recoverable-static-gap traces
- per-trace perplexity for all regimes
- per-trace recovery for M10, midpoint control, and random-bin control
- median recovery and bootstrap 95% CI for each method/control

## Decision rule

### PASS_M10_POSITION_BINNED

Return PASS if all of the following hold:

1. M10 median recovery is at least `0.40`.
2. M10 bootstrap CI lower bound is greater than `0.20`.
3. M10 beats static SmoothQuant by at least `0.20` median recovery.
4. M10 beats midpoint matched-cost control by at least `0.10` median recovery.
5. M10 beats random-bin assignment by at least `0.15` median recovery.

### KILL_M10_NO_IMPROVEMENT

Return this kill if M10 median recovery is within `0.05` of static
SmoothQuant.

### KILL_M10_RANDOM_CONTROL_BEATS

Return this kill if the random-bin scale assignment beats M10 by more than
`0.10` median recovery.

### KILL_M10_AMBIGUOUS

Return ambiguous kill for middle outcomes with overlapping confidence
intervals or insufficient separation from controls.

### FAIL_INFRA_M10

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by reducing batch size, missing required artifacts, or
checker failure that prevents applying the decision rule.

## Required artifacts

Each M10 result packet must contain:

- environment snapshot
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- calibration activation manifest
- scale table manifest with SHA-256 hashes
- quantization configuration and implementation route
- per-trace perplexity table for all regimes
- per-trace recovery table
- metrics JSON with bootstrap CIs
- checker result and artifact check
- artifact hashes

## Forbidden actions

- Modifying prior preregistration files.
- Adjusting M10 thresholds after observing M10 data.
- Replacing the fixed five Step 9.0 bins after observing M10 data.
- Omitting midpoint matched-cost control.
- Omitting random-bin negative control.
- Reporting a positive result without all matched-cost and random controls.
- Using SGLang for Phase 9.
- Modifying model source code.

## Paper integration rule

If M10 passes, the paper may frame position-binned scales as the first
positive Phase 9 method and keep M2 as a failed membership-only ablation. If
M10 kills, the paper follows Outcome Path C: both M2 and M10 fail, so the
paper remains a characterization/mechanism paper unless M9 produces a
publishable predictability result.
