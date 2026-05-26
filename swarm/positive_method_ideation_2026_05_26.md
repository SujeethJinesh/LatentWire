# SA2 Positive-Method Ideation, 2026-05-26

## Status Gate

Current paper readiness: **not ICLR-ready as a positive-method paper**.
Estimated distance to ICLR readiness: one successful preregistered method
branch plus a larger frozen slice, seed repeats, paired uncertainty, and at
least one strict same-family/cross-family separation gate.

Current story of the paper: decode-position channel-set drift is real and
well-supported. Strict top-1% set-leaving is Granite `0.566`, Nemotron
`0.534`, DeepSeek `0.671`, and Falcon-H1 `0.674`. Hard switching is harmful,
smoothing alone is insufficient, budget matters, and ParoQuant-style rotation
is currently the strongest Granite baseline.

Exact blocking gap: no deployable positive method yet survives beyond the
small intervention surface. M11b has real signal but is budget/model dependent
(`0.449` Granite top-5 CI `[-1.30, 1.00]`; Nemotron top-10 `0.815` CI
`[0.254, 0.923]`). ParoQuant Granite is stronger (`0.754` CI
`[0.477, 1.00]`) but is a rotation baseline, not the project's channel-set
method. M18 is negative in absolute recovery (`-0.344`) despite strong
signal-vs-random, and M26 is only weakly positive (`0.178`).

Saturated or weakened: static unions, hard position bins, hard scale bins,
top-1% EMA alone, DecDEC-style top-1% reactive protection, stable core alone,
and compound-error explanations. Still alive: rotation+budget composition,
budget-resolution policies, M18b-style cross-tensor protection at larger
budgets, sensitivity-weighted allocation, and receiver-safe online calibration.

## Evidence Anchors

| Evidence | Project number | Implication |
|---|---:|---|
| Four-model strict set-leaving | `0.566/0.534/0.671/0.674` | Static protected sets are not enough. |
| Granite dense FFT | entropy `0.853`, autocorr `100` tokens | Broad, rapidly decorrelating trajectories; pure forecasting is risky. |
| Dense KL | means `0.150/0.133/0.131`; AR decay `0.44-0.51` | Error growth is sublinear; focus on stale selection/budget, not runaway repair. |
| M11b Granite | top-5 `0.449`, CI `[-1.30,1.00]` | Budget helps but Granite evidence is high variance. |
| M11b Nemotron | top-10 `0.815`, CI `[0.254,0.923]` | Strongest channel-set positive clue; budget is load-bearing. |
| ParoQuant Granite | `0.754`, CI `[0.477,1.00]` | Rotation is the strongest immediate baseline and composition target. |
| M26 stable core | `0.178`; stable fractions `0.51-0.62` | Persistent channels exist but core-only coverage is too small. |
| M18 activation+K | `-0.344` but strong signal-vs-random | Cross-tensor signal exists; top-1% budget/selection likely failed. |

## Ranked Summary Table

| Rank | ID | Method | Main mechanism | Cost | PASS probability | Suitability |
|---:|---|---|---|---|---:|---|
| 1 | M37 | Rotation-Budget Co-Design | ParoQuant plus M11b dynamic activation budget | Medium | 28-45% | ICLR if cross-model |
| 2 | M38 | Sensitivity-Weighted Budget Allocator | Allocate protected channels by activation sensitivity and drift | Medium | 24-38% | ICLR/workshop |
| 3 | M39 | Rotated-Basis EMA Protection | Track channel sets after rotation flattens outliers | Medium | 22-35% | ICLR if beats ParoQuant |
| 4 | M40 | Cross-Tensor Top-5 Halo | Reopen M18 only at top-5/top-10 with KV coupling | Medium-High | 18-30% | Workshop to ICLR |
| 5 | M41 | Stable-Core + Migrating-Halo Split | Keep stable core, spend extra budget on EMA halo | Low-Med | 18-29% | Workshop/appendix |
| 6 | M42 | Hysteretic Event Refresh | Smooth change-point refresh with hold bands | Low | 14-24% | Workshop |
| 7 | M43 | Layer-Adaptive Budget Lagrangian | Per-layer budget optimized under byte constraint | Medium | 14-24% | Workshop/ICLR support |
| 8 | M44 | Prompt-Conditional TTQ Warm Start | Prompt-local online calibration seeded by drift stats | Medium | 12-23% | Workshop |
| 9 | M45 | ParoQuant Residual Sideband | Tiny protected sideband for channels rotations do not flatten | Medium | 12-22% | ICLR if additive |
| 10 | M46 | Sigma-Delta Error Feedback | Carry quantization residual as bounded state | Medium | 10-20% | Workshop |
| 11 | M47 | KV-Activation Progressive Mix | PM-KVQ/KIVI-style cache budget tied to activation drift | High | 10-20% | ICLR if systems-strong |
| 12 | M48 | Stable-Core Anchored Transport | Optimal-transport map from stable core to late halo | Medium | 9-18% | Workshop |
| 13 | M49 | Dense-Grid Budget Predictor | Learn budget from early 1K-token trajectory features | Low-Med | 9-17% | Workshop |
| 14 | M50 | Conformal Accept/Defer Controller | Apply method only when uncertainty band predicts benefit | Low | 8-16% | Support/control |
| 15 | M51 | Synthetic Long-Decode Calibration | Generate calibration traces to learn late-channel stats | High | 8-16% | Workshop |
| 16 | M52 | Low-Frequency Subspace Tracker | Streaming PCA over slowly varying drift modes | Medium | 7-14% | Workshop only |
| 17 | M53 | AR-Kalman Channel Forecaster | Predict future protected sets from early trajectory | Medium | 6-13% | Low priority |
| 18 | M54 | Set-Leaving Regularized Rotation | Learn rotations penalized by late set-leaving | High | 6-13% | ICLR only if works |
| 19 | M55 | Temporal Protected-Set Ensemble | Weighted ensemble of scale/protection tables | Low | 5-11% | Appendix |
| 20 | M56 | Trace-Class Mixture Policy | Route trace classes to separate budget policies | Medium | 5-10% | Workshop/control |

## Top-5 Recommendations

1. **Run M37 first.** It is the most direct way to convert the current best
   baseline into a project-owned positive method: preserve ParoQuant's Granite
   recovery while adding M11b's dynamic budget signal.
2. **Pair M37 with M38 as the diagnostic allocator.** If composition helps,
   M38 explains whether the gain is from sensitivity, drift, or raw budget.
3. **Keep M39 as the cleanest ablation.** It tests whether the channel drift
   problem becomes easier in the rotated basis without adding many degrees of
   freedom.
4. **Only reopen M18 through M40.** The old top-1% M18 is killed/ambiguous;
   the live hypothesis is specifically cross-tensor halo at larger budget.
5. **Use M41 as the cheap fallback.** It is not likely to be ICLR-headline by
   itself, but it can validate whether stable cores and migrating halos should
   be separated in every later method.

## Candidate Methods

### M37: Rotation-Budget Co-Design

Mechanism: apply ParoQuant-style pairwise/Givens rotation and channel-wise
scaling first, then run M11b-style EMA protected-channel selection with a
small dynamic budget grid (`top-1/5/10`). The method claim is composition:
rotation reduces outlier dynamic range, while dynamic budget protects residual
late-decode channels that rotation does not eliminate.

Empirical motivation: ParoQuant Granite recovers `0.754` with CI
`[0.477,1.00]`, stronger than M11b Granite top-5 `0.449`; M11b Nemotron
top-10 recovers `0.815` and beats static top-10 by `0.220`. The obvious next
positive row is therefore "preserve ParoQuant and add portable budget."

Cross-domain analogy: image compression often applies a transform before
bit allocation; this is transform coding plus adaptive bit budgeting.

Differentiation: ParoQuant is weight-only rotation/scaling for efficient
reasoning inference, not decode-position activation budget control
(arXiv:2511.10645, OpenReview ICLR 2026). M11b is channel-set protection
without rotation. This method is their explicit interaction test.

Implementation cost: medium. Needs a reproducible ParoQuant runner, M11b
protected-set generation in the rotated basis, matched static top-10, and
ParoQuant-only controls.

Honest PASS probability: 28-45%. Risks: ParoQuant may already remove the
recoverable gap, leaving no denominator; composition may add overhead without
quality lift. Composability with M11b/ParoQuant: maximal. Suitability:
ICLR-worthy if it beats ParoQuant-only on Granite and preserves M11b's Nemotron
top-10 signal under paired uncertainty; workshop if it is only descriptive.

### M38: Sensitivity-Weighted Budget Allocator

Mechanism: replace magnitude-only top-k with a score
`channel_score = sensitivity * drift_probability * EMA_magnitude`, then solve
a per-layer budget allocation under a global byte cap. Sensitivity can be
estimated by loss/logit perturbation on frozen calibration traces.

Empirical motivation: top-1% EMA alone is only `0.048`, but top-10 Nemotron
M11b is `0.815`; the missing variable is not just smoothness, it is where the
extra budget is spent. M26's `0.178` shows stable-core magnitude is not enough.

Cross-domain analogy: unequal error protection in communications, where
limited redundancy is assigned to symbols with high downstream loss.

Differentiation: AWQ uses activation-aware weight quantization; recent
activation-sensitivity theory frames channel perturbation impact on loss
(arXiv:2601.11663). This idea uses sensitivity specifically for long-decode
protected activation-channel budgeting.

Implementation cost: medium. Perturbation scoring is expensive but can be
cached on the same frozen slice.

Honest PASS probability: 24-38%. Risks: sensitivity estimates may be noisy and
overfit to Granite traces; reviewers will demand random and magnitude-only
controls. Composability: high with M11b and ParoQuant. Suitability: ICLR if
cross-family allocation generalizes; workshop if it only explains M11b.

### M39: Rotated-Basis EMA Protection

Mechanism: run EMA protected-set tracking after a fixed rotation/scale
conditioning transform. Unlike M37, the key ablation is not extra budget but
whether the identity of high-value channels becomes more stable after rotation.

Empirical motivation: four-model set-leaving is `0.534-0.674`, and broad FFT
entropy `0.853` argues raw channel identities are noisy. ParoQuant's `0.754`
Granite recovery suggests a better coordinate system may exist.

Cross-domain analogy: tracking a signal after whitening/preconditioning rather
than in raw sensor coordinates.

Differentiation: QuaRot and ParoQuant rotate to reduce quantization error;
this tests rotation as a coordinate system for online protected-set stability.
It differs from DecDEC's per-step reactive top-1% selection.

Implementation cost: medium. Requires exposing rotated activations or an
equivalent transformed scoring path.

Honest PASS probability: 22-35%. Risks: rotation may flatten magnitudes so
top-k becomes less meaningful; any lift may be fully attributable to ParoQuant.
Composability: high with M11b/ParoQuant. Suitability: ICLR only if the rotated
EMA beats both ParoQuant-only and M11b-only at matched bytes.

### M40: Cross-Tensor Top-5 Halo

Mechanism: reopen M18 as a new preregistered branch with a larger halo budget:
protect activation channels plus aligned K/V-cache channels for the top-5 or
top-10 dynamic halo, while keeping a static activation baseline at matched
bytes.

Empirical motivation: M18 absolute recovery is `-0.344`, but it strongly beats
random coupled controls; M11b shows top-1% budget is too small. This promotes
only the cross-tensor signal, not the killed top-1% method.

Cross-domain analogy: joint source-channel coding; protect coupled state
variables together instead of optimizing each tensor independently.

Differentiation: KIVI and PM-KVQ quantize KV caches; M18b would link cache
precision to activation-channel drift rather than cache statistics alone.

Implementation cost: medium-high. Needs reliable K/V hooks, matched cache
budget accounting, and same-family/cross-family controls.

Honest PASS probability: 18-30%. Risks: hook complexity and throughput cost;
KV-only baselines may absorb the gain. Composability: medium with M11b, medium
with ParoQuant. Suitability: workshop first; ICLR if it creates a clean
activation+KV method with systems accounting.

### M41: Stable-Core + Migrating-Halo Split

Mechanism: reserve a small always-protected stable core and spend remaining
budget on an EMA migrating halo. The static core is selected by intersection
across early calibration positions; the halo is dynamic and budgeted.

Empirical motivation: stable-core fractions exist in every model
(`0.51-0.62`), but M26 core-only recovery is only `0.178`. M11b says extra
budget matters; this method separates persistent and migrating roles.

Cross-domain analogy: CPU cache hierarchy: pin hot invariant lines, use the
rest for adaptive replacement.

Differentiation: static union and stable-core-only were already insufficient;
the novelty is an explicitly split budget with different update rules.

Implementation cost: low-medium. Mostly protected-set logic plus existing M11b
runner reuse.

Honest PASS probability: 18-29%. Risks: may be a weaker M11b; stable core may
consume budget better spent dynamically. Composability: high with M11b,
medium with ParoQuant. Suitability: workshop or appendix unless it generalizes
cleanly across Granite/Nemotron.

### M42: Hysteretic Event Refresh

Mechanism: update protected sets only when a smoothed score exits a hold band;
use hysteresis and interpolation to avoid M2/M10-style boundary jumps.

Empirical motivation: hard position switching M2 is `-0.867` and hard scale
bins M10 are beaten by random. EMA helps but top-1% M11 is only `0.048`.
Hysteresis is the minimal mechanism between hard switching and passive EMA.

Cross-domain analogy: thermostat control with dead bands to prevent rapid
oscillation.

Differentiation: DecDEC refreshes reactively each step; TTQ performs online
activation-aware calibration. This is a cheap stability rule for protected
sets, not full test-time requantization.

Implementation cost: low.

Honest PASS probability: 14-24%. Risks: likely too incremental; may reproduce
M11b with more knobs. Composability: high with M11b, low with ParoQuant.
Suitability: workshop.

### M43: Layer-Adaptive Budget Lagrangian

Mechanism: allocate protected budget by layer using a Lagrangian objective:
maximize expected recovery per byte subject to a global budget. Inputs include
layer set-leaving, stable-core fraction, and sensitivity.

Empirical motivation: layer dissection shows drift is not isolated to a single
component class, but layer-level variation exists. Nemotron top-10 passes while
Granite top-5 is noisy, so uniform budget is probably not optimal.

Cross-domain analogy: rate-distortion allocation across frequency bands.

Differentiation: KL Lens and layer-wise PTQ methods study sensitivity by layer;
this is a protected-channel budget allocator for long-decode drift.

Implementation cost: medium.

Honest PASS probability: 14-24%. Risks: can become post-hoc layer picking; must
predefine features and fit only on calibration traces. Composability: high with
M11b and M37. Suitability: ICLR support method if M37 works; workshop alone.

### M44: Prompt-Conditional TTQ Warm Start

Mechanism: perform a short prompt-local online calibration pass to initialize
scales/protected sets, then decode with smoothed updates seeded by that prompt
rather than global calibration.

Empirical motivation: no-gap fractions differ (`0.333` Granite ParoQuant,
`0.167` Nemotron M11b), indicating trace heterogeneity. Prompt-specific
calibration may reduce variance before the long decode.

Cross-domain analogy: adaptive filters that estimate channel conditions from a
preamble.

Differentiation: TTQ adapts quantization at inference time (arXiv:2603.19296).
This idea is narrower: prompt warm-start for W4A16 protected-channel policy,
with strict same-prompt destructive controls.

Implementation cost: medium.

Honest PASS probability: 12-23%. Risks: latency overhead and prior-art
pressure from TTQ; must show long-reasoning drift-specific benefit.
Composability: high with M11b, medium with ParoQuant. Suitability: workshop.

### M45: ParoQuant Residual Sideband

Mechanism: run ParoQuant, then identify residual channels with high
post-rotation quantization error or late sensitivity and preserve a tiny
sideband at higher precision.

Empirical motivation: ParoQuant is strong but not perfect: included-trace
median `0.754`, with one severe negative recovery outlier in the packet. A
sideband targets remaining failures without replacing rotation.

Cross-domain analogy: base-layer/enhancement-layer coding.

Differentiation: KVQuant/KIVI compress cache state; ParoQuant rotates weights.
This method adds activation-side residual protection after rotation.

Implementation cost: medium.

Honest PASS probability: 12-22%. Risks: byte overhead may erase the systems
story; severe outlier may be trace noise rather than fixable residual.
Composability: high with ParoQuant, medium with M11b. Suitability: ICLR if
additive at matched bytes, otherwise workshop.

### M46: Sigma-Delta Error Feedback

Mechanism: maintain a small per-layer residual accumulator for activation
quantization error and feed it back into later scale/protection decisions.

Empirical motivation: KL is sublinear with AR decay `0.44-0.51`, so bounded
residual feedback is plausible; runaway correction is not the right target.

Cross-domain analogy: sigma-delta modulation and error-feedback quantization.

Differentiation: not a static PTQ method; it is decode-stateful error shaping
for activation channels under long reasoning.

Implementation cost: medium.

Honest PASS probability: 10-20%. Risks: state overhead, numerical instability,
and difficult attribution. Composability: medium with M11b, low with ParoQuant.
Suitability: workshop.

### M47: KV-Activation Progressive Mix

Mechanism: combine progressive mixed-precision KV-cache quantization with
activation protected-channel drift; raise precision only where activation drift
and cache sensitivity agree.

Empirical motivation: M18 says activation/K coupling has signal, but top-1%
fails. M11b says larger budgets matter. This is the systems-heavy version of
M40.

Cross-domain analogy: multimodal sensor fusion; require two sensors to agree
before spending extra bandwidth.

Differentiation: PM-KVQ, KIVI, AttentionPredictor, ChanMix, and MixKVQ focus
on KV-cache policies. This adds activation drift as the trigger.

Implementation cost: high.

Honest PASS probability: 10-20%. Risks: too broad before the method gate;
could widen benchmark scope prematurely. Composability: medium with M11b,
medium with ParoQuant. Suitability: ICLR only after M40 passes.

### M48: Stable-Core Anchored Transport

Mechanism: use stable core channels as anchors and learn an optimal-transport
map from early protected sets to late migrating halos. Protect channels that
the transport predicts will become high value.

Empirical motivation: every model has a nonzero stable core, but set-leaving
remains high. The core may be useful as a coordinate system rather than as the
entire protected set.

Cross-domain analogy: landmark-based image registration.

Differentiation: unlike static core M26, the core is used to align moving
channels. Unlike global rotations, transport is local and position-aware.

Implementation cost: medium.

Honest PASS probability: 9-18%. Risks: easy to overfit sparse six-position
packets; dense packets exist only for Granite. Composability: medium with
M11b, low-medium with ParoQuant. Suitability: workshop.

### M49: Dense-Grid Budget Predictor

Mechanism: from the first 1K decode tokens, predict which budget arm
(`top-1/5/10`) should be used for the rest of the trace. Features include
early set-leaving slope, KL proxy, entropy, and stable-core fraction.

Empirical motivation: Granite prefers top-5 signal with wide CI, Nemotron
passes top-10, and no-gap rates vary by trace. A fixed budget is not portable.

Cross-domain analogy: adaptive bitrate streaming: early network measurements
choose later bitrate.

Differentiation: not another protected-set update rule; it is a budget
selection wrapper around M11b with preregistered features.

Implementation cost: low-medium.

Honest PASS probability: 9-17%. Risks: small trace count; needs frozen larger
slice before any claim. Composability: high with M11b and M37. Suitability:
workshop unless validated on large repeats.

### M50: Conformal Accept/Defer Controller

Mechanism: apply a dynamic protection method only when a calibrated uncertainty
band predicts positive recovery; otherwise fall back to static or ParoQuant.

Empirical motivation: ParoQuant and M11b have trace-level heterogeneity, and
Granite has `0.333` no-recoverable-gap traces. Selectivity could improve
median and reduce harms.

Cross-domain analogy: selective classification with abstention.

Differentiation: this is a safety controller, not a core quantizer. It must be
reported as a wrapper unless it unlocks a positive method that would otherwise
be too harmful.

Implementation cost: low.

Honest PASS probability: 8-16%. Risks: reviewer may see cherry-picking; must
use conformal calibration and frozen accept thresholds. Composability: high
with all methods. Suitability: support/control, rarely ICLR-headline.

### M51: Synthetic Long-Decode Calibration

Mechanism: build calibration traces that deliberately induce long reasoning
states, then learn late-channel statistics before deployment.

Empirical motivation: global calibration at position 100 misses 53-67% of
late top-1% channels. The problem may be calibration distribution, not online
policy.

Cross-domain analogy: stress testing hardware under worst-case workloads
rather than average workloads.

Differentiation: SmoothQuant/AWQ use calibration but not long-reasoning
set-leaving objectives. Quamba2's channel persistence assumption is challenged
at our measured block-output surface.

Implementation cost: high because it needs data generation and full
recalibration.

Honest PASS probability: 8-16%. Risks: synthetic traces may not transfer;
could become a dataset paper rather than a method. Composability: medium with
M11b, high with ParoQuant. Suitability: workshop.

### M52: Low-Frequency Subspace Tracker

Mechanism: maintain a streaming low-rank subspace of high-magnitude channel
trajectories and protect channels with high projected future energy.

Empirical motivation: only `0.275` low-frequency power and entropy `0.853`
make this risky, but not impossible; there may be layer-local low-frequency
modes hidden by aggregate entropy.

Cross-domain analogy: tracking a slowly moving target with PCA background
models.

Differentiation: not DecDEC top-k; it protects subspace energy instead of
instantaneous magnitude.

Implementation cost: medium.

Honest PASS probability: 7-14%. Risks: aggregate FFT argues against it; dense
data only exists for Granite. Composability: medium with M11b, low with
ParoQuant. Suitability: workshop only.

### M53: AR-Kalman Channel Forecaster

Mechanism: fit an autoregressive/Kalman predictor for channel magnitudes and
protect predicted late top-k channels.

Empirical motivation: AR decays `0.44-0.51` and autocorr length `100` tokens
mean the forecast horizon is short. This is alive only as a cheap negative
control against fancier predictors.

Cross-domain analogy: Kalman filtering for noisy sensor state.

Differentiation: unlike M11b, it forecasts future channels rather than
smoothing present magnitudes.

Implementation cost: medium.

Honest PASS probability: 6-13%. Risks: likely fails beyond short horizons.
Composability: low-medium with M11b, low with ParoQuant. Suitability: low
priority control.

### M54: Set-Leaving Regularized Rotation

Mechanism: learn rotations with a dual objective: reduce quantization error
and reduce late decode top-k set-leaving in the rotated basis.

Empirical motivation: ParoQuant works well on Granite, but it is optimized for
quantization distortion, not set stability. If rotations can make set-leaving
smaller, dynamic protection becomes easier.

Cross-domain analogy: representation learning with temporal consistency
regularization.

Differentiation: ParoQuant and QuaRot optimize rotation for quantization; this
adds a long-decode stability loss.

Implementation cost: high.

Honest PASS probability: 6-13%. Risks: training complexity, overfitting, and
weak novelty if M39 already answers the question. Composability: high with
ParoQuant, medium with M11b. Suitability: ICLR only after M39 shows promise.

### M55: Temporal Protected-Set Ensemble

Mechanism: precompute several protected sets/scale tables from calibration
positions and mix them continuously by decode position with smooth weights.

Empirical motivation: static unions failed, and hard bins were harmful. A
smooth ensemble is the least disruptive version of multi-position protection.

Cross-domain analogy: mixture of experts with soft routing.

Differentiation: this is not hard position binning; it removes discontinuities
but keeps precomputed tables.

Implementation cost: low.

Honest PASS probability: 5-11%. Risks: likely dominated by M11b; static-union
history is bad. Composability: medium with M11b, low with ParoQuant.
Suitability: appendix/control.

### M56: Trace-Class Mixture Policy

Mechanism: classify prompts/traces into a small number of drift regimes and
apply separate protected-set/budget policies per class.

Empirical motivation: trace heterogeneity is real: no-gap fractions are
`0.333` on Granite ParoQuant and `0.167` on Nemotron M11b. But current slices
are too small for reliable routing.

Cross-domain analogy: mixture-of-experts routing for workload classes.

Differentiation: confidence-only routing has failed in the broader portfolio;
this must route on measurable drift/calibration features, not answer
confidence.

Implementation cost: medium.

Honest PASS probability: 5-10%. Risks: p-hacking and low sample size. It
should not be run before larger frozen slices exist. Composability: high with
M11b, medium with ParoQuant. Suitability: workshop/control.

## Prior-Art Check Sources

- ParoQuant, pairwise rotation quantization for efficient reasoning LLM
  inference: arXiv:2511.10645 and OpenReview ICLR 2026,
  https://arxiv.org/abs/2511.10645,
  https://openreview.net/pdf?id=1USeVjsKau.
- DecDEC, OSDI 2025 dynamic low-bit LLM quantization:
  https://www.usenix.org/conference/osdi25/presentation/park-yeonhong.
- Quamba2, selective state-space model PTQ:
  https://arxiv.org/abs/2503.22879.
- QMamba and OuroMamba, vision-Mamba dynamic/outlier-aware quantization
  context: https://arxiv.org/abs/2501.13624,
  https://arxiv.org/abs/2503.10959.
- SmoothQuant, AWQ, QuaRot, KVQuant, and KIVI as core PTQ/KV baselines:
  https://arxiv.org/abs/2211.10438,
  https://arxiv.org/abs/2306.00978,
  https://arxiv.org/abs/2404.00456,
  https://arxiv.org/abs/2401.18079,
  https://arxiv.org/abs/2402.02750.
- PM-KVQ and AttentionPredictor as recent long-context KV-cache policies:
  https://arxiv.org/abs/2505.18610,
  https://arxiv.org/abs/2502.04077.
- TTQ and activation sensitivity as recent test-time/adaptive PTQ context:
  https://arxiv.org/abs/2603.19296,
  https://arxiv.org/abs/2601.11663.
