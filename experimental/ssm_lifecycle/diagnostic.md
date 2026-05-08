# SSM Lifecycle Phase 0 Diagnostic

## Proximate Failure

Category: (a) hypothesis wrong in its preregistered operational form.

The checker decision is `KILL_SSML_PHASE0_STATE_STABLE` with
`artifact_complete=true`. All 36 Mamba layers had combined KS p-values below
0.01, but 0/36 layers reached the required 2x median drift ratio, so the
failure is not lack of statistical detectability. The data says state
distributions shift, but not with the large magnitude drift needed to justify
age-based aggressive compression as preregistered.

## Fairness Of Setup

The setup was fair for the registered Phase 0 gate: frozen model,
deterministic 12-prompt AIME-2025 slice, all 36 Mamba layers included,
preregistered positions, mechanical checker, and complete artifact packet.

Limitations do not rescue the hypothesis. Some traces reached EOS before
position 10000 and were force-continued per the runner's fixed-position
capture policy; the runner used the naive Mamba path rather than optional fast
kernels, but that affects speed more than captured state values. A fair retest
of natural long traces would require a fresh preregistration with new
prompts/thresholds, not a rerun or relaxed 2x criterion.

## Unexpected Metric Patterns

- KS significance saturated: combined p-values underflow to 0.0 across all
  Mamba layers, so p-value is not a useful effect-size scale here.
- Magnitude drift is tightly centered near 1.0: layer median drift ratio
  median=0.9676, mean=0.9650, min=0.7047, max=1.1584; 0/36 layers reached
  1.25x.
- Direction is mixed, not monotone aging: 13/36 layers increased, 12/36 were
  <=0.9, and the 20-29 layer band had mean drift 0.896 with 7/9 layers <=0.9.
- Trace-level spikes exist but are not stable: only 1/432 trace-layer pairs
  reached >=2x drift, and only 3/432 reached >=1.5x.

## Fresh Positive-Method Hypotheses

1. Shape-conditioned SSM state codec. Old states may differ in distribution
   shape, tails, or quantiles without magnitude growth. A fresh gate would
   test whether age-specific codebooks/normalizers reduce reconstruction error
   and preserve decode quality versus age-agnostic state quantization on a new
   frozen slice.
2. Depth-stratified SSM compression. Coarse depth bands may have different
   attenuation regimes. A fresh gate must predeclare relative depth bins on a
   new model/slice; it cannot target observed layers 20-29 post hoc.
3. Trajectory-conditioned state policy. Drift may be prompt/trajectory
   dependent rather than layer-uniform. A fresh gate would need an online
   predictor from early state statistics and held-out prompts; observed prompt
   IDs cannot define the rule.

## Pivot Decision

Only hypothesis 1 is plausibly authorable now without p-hacking, and only as a
fresh branch with new operational thresholds based on quantization error and
decode-quality preservation. Hypotheses 2 and 3 are currently too post-hoc
from this packet to author safely.

Do not relax or rerun SSML Phase 0. The killed claim remains killed.
