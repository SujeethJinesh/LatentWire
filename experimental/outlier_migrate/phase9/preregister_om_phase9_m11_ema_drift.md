# Phase 9 M11 EMA-Smoothed Drift Protection Preregistration

**Frozen on**: 2026-05-15
**Frozen by**: Codex GPU swarm, under vacation-mode 48-hour sprint
authorization
**Status**: Frozen after M2 landed and while hard-binned M10 was still
running; before any M11 scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined context that distinguishes this channel
set shift across decode positions from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Motivation

M2 tested position-conditional protected-set switching. On Granite-4-H-Small,
M2 landed with `KILL_M2_RANDOM_CONTROL_BEATS`: the random-bin assignment beat
the intended assignment by `0.6675482901676153` median recovery. That result
suggests that abrupt boundary-discontinuous switching may itself be harmful,
not merely that the protected set was poorly chosen.

M11 tests the next structurally different decision surface: update the
protected set smoothly with an exponential moving average (EMA) over the
top-channel indicator vector. The hypothesis is that temporal smoothing can
track decode-position channel drift while avoiding hard bin boundaries.

## Prior-art differentiation

DecDEC (Park, Hyun, Kim, and Lee; OSDI 2025) is the closest inference-time
dynamic-channel neighbor. Its stated design is per-step reactive selection:
"Salient channels are identified dynamically at each decoding step by
analyzing the input activations." M11 is different in the decision surface:
it explicitly smooths the indicator vector across decode steps and caps
membership turnover, rather than re-selecting independently at each decoding
step.

PMPD (Progressive Mixed-Precision Decoding, arXiv 2410.13461) states:
"PMPD builds upon the observation that the prefill phase and the initial
tokens of the decoding phase are more sensitive to approximations than later
tokens." M11 does not allocate lower precision monotonically with decode
position. It keeps the quantization format fixed and updates which channels
are protected as the top-channel set drifts.

MoBiQuant uses "outlier migration" for token-level sensitivity shifts when
bit-width changes. M11 does not route tokens to bit slices and does not target
elastic bit-width switching; it targets channel membership drift at fixed
W4A16-style inference precision.

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

Additional models are not part of this primary gate. If M11 passes on
Granite-Small, the 48-hour sprint decision tree authorizes separate
replication runs on Nemotron-3-Nano and DeepSeek-R1-Distill-Qwen-1.5B.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Prompt payload SHA-256 over indices `0-11`: computed by the runner and
  written to the result packet before analysis.

The 12-trace slice is chosen before M11 data exists and matches the
vacation-mode slice used by M2/M10 on Granite-Small. Decision thresholds are
not relaxed for the smaller slice.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: simple symmetric per-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16.
- Protected channels remain in the unquantized/dequantized high-precision
  path as in prior Phase 4/Phase 9 protected-channel runners.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260525`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, and any
post-hoc threshold tuning are forbidden for M11.

## M11 Protected-Set Update Rule

For each layer independently, maintain a protected-set score vector
`p_l(t)` with one score per channel.

Initialization:

- `p_l(0)` is the calibration top-1% indicator at decode position `100`.
- This is not a cold start.

At each decode position that is a multiple of `100`:

`p_l(t+1) = alpha * current_top_1pct_indicator_l(t) + (1 - alpha) * p_l(t)`

Protected channels at position `t`:

- Select channels where `p_l(t) > threshold`.
- If this exceeds a top-3% absolute cap, evict channels by
  longest-protected duration until the cap is met.
- If fewer than top-1% channels satisfy the threshold, fill by highest
  `p_l(t)` score until top-1% is reached.

The primary threshold is `0.5`. The runner must record the effective
protected count per layer and update step. The top-3% cap is fixed and must
not be changed after observing M11 data.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static-1% protected set from position `100`.
3. M11 EMA-smoothed drift protection with `alpha = 0.1`.
4. M11 EMA-smoothed drift protection with `alpha = 0.3`.
5. M11 EMA-smoothed drift protection with `alpha = 0.5`.
6. Random-walk-protection control: matched 100-position update cadence and
   matched protected-channel counts, but channels are selected by a seeded
   random walk independent of activation magnitudes.

The user-facing sprint prompt listed five regimes and named alpha `0.1` and
`0.3`; it also required sweeping `alpha in {0.1, 0.3, 0.5}`. This
preregistration resolves the count mismatch by including all three required
alpha values plus the mandatory baselines/control. This adds a stricter arm;
it does not relax the gate.

## Metric

For each trace and M11/control regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)`

Only traces with a positive recoverable static-1% gap are included in the
primary recovery median. The packet must still report the count and fraction
of no-recoverable-static-gap traces.

For each alpha, report:

- per-trace perplexity for all regimes
- per-trace recovery
- median recovery
- bootstrap 95% CI for the median
- median separation from static-1%
- median separation from random-walk control
- protected-set count statistics
- strict set-leaving rate after applying the smoothed protected set

## Decision Rule

### PASS_M11_EMA_DRIFT

Return pass if at least one alpha in `{0.1, 0.3, 0.5}` satisfies all of:

1. median recovery is at least `0.30`;
2. bootstrap 95% CI lower bound is greater than `0.10`;
3. median recovery beats static-1% by at least `0.15`;
4. median recovery beats random-walk-protection control by at least `0.20`.

### KILL_M11_NO_IMPROVEMENT

Return this kill if every alpha is within `0.05` median recovery of static-1%.

### KILL_M11_RANDOM_CONTROL_BEATS

Return this kill if the random-walk-protection control beats every M11 alpha
by more than `0.10` median recovery.

### KILL_M11_AMBIGUOUS

Return ambiguous kill for middle outcomes that are neither pass nor the two
specific kill modes, including outcomes where an alpha improves over static
but fails CI or random-walk separation.

### FAIL_INFRA_M11

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by batch-size reduction, hook incompatibility, or checker
failure that prevents applying the mechanical decision rule.

## Required Artifacts

Each M11 packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA)
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- BF16 target traces or cited cache source with SHA-256
- activation/top-channel evidence used for EMA updates
- protected-set trajectory for each alpha and the random-walk control
- quantization configuration
- per-trace perplexity table
- per-trace recovery table
- bootstrap CI table
- checker result and artifact check
- artifact hashes

## Forbidden Actions

- Modifying prior preregistration files.
- Adjusting M11 alpha values, thresholds, caps, or decision thresholds after
  observing M11 data.
- Dropping the random-walk control.
- Replacing the EMA update with hard bin switching.
- Using AWQ-style scaling or SmoothQuant scale folding.
- Selectively reporting only the best alpha without reporting all three.
- Running additional unpreregistered M11 alpha values.
- Modifying model source code.

## Paper Integration Rule

If M11 passes, the paper may frame EMA-smoothed drift protection as a positive
method designed directly in response to the M2 boundary-discontinuity failure.
If M11 kills, Section 6 should treat it as evidence that smoothing set
membership alone is insufficient and proceed to M17 afterglow or M18
cross-tensor coupling per the 48-hour sprint decision tree.
