# Phase 9 KL Accumulation Preregistration

**Frozen on**: 2026-05-19
**Frozen by**: Codex GPU swarm, under revised post-M26 priority order
**Status**: Frozen after M26 landed with `AMBIGUOUS_M26` and before any
KL-accumulation measurement run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined terminology note distinguishing this
channel-across-decode-position usage from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Purpose

Existing Phase 9 method packets score a single 512-token window ending at
decode position `10000`. They support endpoint recovery comparisons, but they
do not identify how quantization error evolves inside a trace. The post-M18
analysis therefore marked per-position recovery curves as not identifiable
from existing packets.

This experiment measures per-position distribution divergence between BF16
and W4A16 regimes along the same long-decode reasoning traces. It tests
mechanism hypothesis 3c: **compound error accumulation**. The result is also
load-bearing for the SLQ counter-result and for any later pulsed-precision
method decision.

## Prior-Art Anchor

Statistically-Lossless Quantization (SLQ; Helcig, Kurtic, and Alistarh,
arXiv:2605.02404, submitted 2026-05-04) studies task-lossless and
distribution-lossless quantization and reports near-lossless 5-6 bpw
distributional fidelity. The arXiv abstract states that SLQ formalizes
distribution-lossless compression through next-token-distribution
indistinguishability and reports efficient statistically-lossless compression
in that higher-bit regime:

- arXiv page: `https://arxiv.org/abs/2605.02404`

The present experiment is not a reproduction of SLQ. It tests a different
regime:

- aggressive W4A16 post-training quantization;
- long reasoning traces up to 20K decode positions;
- small reasoning/hybrid models where previous Phase 9 methods exposed
  recoverable static-protection gaps;
- deterministic shared-prefix distribution comparison rather than task-level
  benchmark accuracy.

The paper must state that agreement or disagreement with SLQ is scoped to this
more aggressive long-decode W4A16 regime.

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

No additional models are part of this Phase 1 KL gate. Cross-model KL curves
are deferred until after M11b replication and reviewer-critical ParoQuant
baseline work.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Target sequence: the deterministic BF16 greedy trace used by M11/M11b/M26
  when available; otherwise the runner must regenerate deterministic BF16
  traces before any KL calculation and record the prompt SHA.

The slice matches M2/M10/M11/M18/DecDEC/M11b/M26 on Granite-Small.

## Quantization Regimes

Compute KL trajectories for:

1. BF16 reference against itself (`KL=0` by definition; stored as a sanity
   row).
2. Static top-1% protection from position `100`.
3. DecDEC algorithmic baseline proxy from the completed DecDEC packet.
4. M11 EMA-smoothed drift protection, using the best preregistered M11 arm
   already evaluated in the M11 packet.

The runner may reuse existing protected-set and score-cache artifacts from
completed M11 and DecDEC packets, but it must record every source artifact
path and SHA-256 in the KL packet.

## KL Measurement

For each trace and decode position `t` from `1` through `20000`, compute:

`KL(BF16 || Q)_t = sum_i p_BF16(i | x_<t) * (log p_BF16(i | x_<t) - log p_Q(i | x_<t))`

where `x_<t` is the same deterministic BF16 target prefix for both models.
This shared-prefix teacher-forced comparison isolates distribution drift from
sampling-path divergence. The runner must compute the KL over the full
vocabulary distribution, not only the observed next token.

If memory or runtime makes full-position evaluation infeasible after an
attempted implementation, the runner may switch to a preregistered dense grid:
all positions `1-512`, every 10th position from `513-2000`, every 50th
position from `2001-10000`, and every 100th position from `10001-20000`.
This fallback must be recorded as `dense_grid_fallback=true`, and the paper
must not describe it as every-position KL. No adaptive position selection is
allowed.

## Growth-Model Fits

For each quantized regime, fit the following models to the per-trace KL
trajectory and to the trace-averaged KL trajectory:

1. Linear: `KL_t = a + b t`.
2. Sublinear square-root: `KL_t = a + b sqrt(t)`.
3. Superlinear power law: `KL_t = a + b t^p`, with `p > 1`.
4. Exponential/AR(1)-style relaxation fit sufficient to estimate an effective
   decay parameter for the M31 pulsed-precision authorization decision.

Report fit parameters, residual sum of squares, AIC where applicable, and the
best-fit class. The experiment is descriptive/mechanistic; it does not have a
positive-method PASS threshold.

## Descriptive Decision Rule

### PASS_KL_ACCUMULATION_REPORTED

Return this decision when:

- the packet is artifact-complete;
- KL trajectories are present for all required regimes;
- growth-model fit outputs are present; and
- the checker can mechanically determine whether the best-fit trajectory is
  more consistent with flat/uniform, linear/sublinear growth, superlinear
  growth, or inconclusive/noisy dynamics.

### FAIL_INFRA_KL_ACCUMULATION

Return infrastructure failure for model load failure, OOM that cannot be
fixed by batch-size reduction or the dense-grid fallback, missing source
artifacts, incomplete KL rows, invalid probabilities, or checker failure that
prevents applying the descriptive rule.

## Required Artifacts

The KL packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA);
- model provenance with HuggingFace snapshot commit;
- prompt manifest and prompt SHA;
- exact command line and stdout/stderr logs;
- source artifact manifest for reused BF16 traces and protected sets;
- quantization/protection configuration for every compared regime;
- per-trace, per-position KL table, stored compressed if needed;
- per-regime summary curves;
- growth-model fit table and residual diagnostics;
- checker result and artifact check;
- artifact hashes.

## Forbidden Actions

- Modifying prior preregistration files.
- Changing the model, trace set, or target sequence after observing KL data.
- Selecting positions adaptively based on preliminary KL values.
- Dropping DecDEC or M11 from the comparison.
- Replacing full-vocabulary KL with top-k-only KL without writing a
  `FAIL_INFRA_KL_ACCUMULATION` packet.
- Claiming contradiction or confirmation of SLQ outside the aggressive
  long-decode W4A16 regime measured here.
- Using this descriptive experiment as a positive-method result.

## Paper Integration Rule

If KL grows superlinearly or shows strong long-range autocorrelation, the
paper should treat compound error accumulation as a live explanation for why
set-level protection methods underperform and should discuss why SLQ's
near-lossless regime may not transfer to aggressive long-decode W4A16.

If KL remains flat or position-uniform, the paper should state that the SLQ
counter-result appears to generalize to this axis and should shift the
mechanism story away from compounding toward signal staleness or budget/set
selection.

If the fit is inconclusive, the paper should report that endpoint recovery
remains interpretable but within-trace error dynamics are not yet resolved.
