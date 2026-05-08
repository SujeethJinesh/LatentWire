# Cross-Layer Error Diagnostic

Date: 2026-05-08

Killed entry: `cross_layer_error_theoretical`

Decision: `KILL_CLE_BOUND_LOOSE`

Result packet:
`experimental/cross_layer_error/results/cle_theoretical_20260508T191327Z`

## Proximate Failure

The hypothesis was wrong on the preregistered decision surface. This was not an
infrastructure failure and not a preregistration ambiguity.

The first-principles FP4 cross-layer drift bound was artifact-complete but too
loose. The checker killed the branch because predicted/measured ratios exceeded
the preregistered `>5.0` kill threshold at depths 1, 10, and 15:

- depth 1: `5.2348196795166055`
- depth 5: `4.996524162884974`
- depth 10: `5.724405940898007`
- depth 15: `6.922192012041549`

The measured mean L2 drift increased from `252.78462756756122` at depth 1 to
`472.08004427928694` at depth 15, while the predicted bound increased from
`1323.2819430699453` to `3267.8287115543008`. The functional form was not
catastrophically wrong; both curves increase. The failure is scale/calibration:
the Lipschitz/Frobenius-style output scaling and layer-error quadrature
overestimate real logit drift, likely because normalization, residual mixing,
gating, and cancellation damp layerwise weight perturbations.

## Fairness of the Gate

The result packet is artifact-complete:

- `artifact_complete=true`
- `checker_result.json` decision: `KILL_CLE_BOUND_LOOSE`
- prompt set: AIME-2025 deterministic indices 0-11
- prompt SHA: `sha256:2f27c54baa8448e033d6e82f53f775dc6abe38188e4f1e5c0b97e3c74fe7c1dd`
- model: `ibm-granite/granite-4.0-h-tiny`
- quantization format: `nvfp4_e2m1_weight_sim`

One documentation caveat: `derivation.md` prose mentions a 32-element block,
but the executable packet's `quantization_config.json` and runner use
`block_size=16`. This does not change the checker outcome and should not be
mined for a retest.

## Saturated or Ruled Out

The killed claim is: a closed-form upper bound
`F(N, sigma_block, sigma_outlier, depth_pattern)` is tight to within 2x of
measured BF16-vs-FP4 drift at depths `{1, 5, 10, 15}`.

Do not rerun this same gate with altered thresholds, post-hoc depth grids,
different prompt subsets, or cherry-picked layers. Do not write a negative
"loose bound" paper from this kill.

## Useful Signal

The data suggests damping rather than unbounded compounding. Drift grows from
depth 1 to depth 5, then marginal growth is much smaller at depths 10 and 15.
That is useful for positive methods because it suggests that some additional
layers can be quantized without proportional logit movement.

## Positive-Method Pivot Hypotheses

### 1. Damped FP4 Precision Allocator

Hypothesis: a small preregistered calibration pass can estimate layerwise
marginal drift damping, then allocate FP4 to layers whose marginal logit and
argmax risk is low.

Potential paper claim: hybrid models contain measurable damping structure that
can be exploited for a precision-allocation method with lower memory/bytes than
uniform BF16 while preserving task accuracy and logit stability.

Plausibility: moderate. This directly follows the observed saturation pattern,
but it must be tested as a method, not as a repaired theory. A valid gate would
need held-out prompts, non-inferior task accuracy/logit stability, an exact
memory or byte reduction target, and at least one cross-family hybrid/control.

### 2. Argmax-Stable FP4 Depth Expander

Hypothesis: although L2 logit drift is nonzero, top-token argmax stability may
remain high as additional low-risk layers move to FP4. A method can expand the
quantized-depth frontier until a preregistered argmax or task-accuracy budget is
met.

Potential paper claim: the practical serving-relevant failure surface is argmax
or answer stability, not raw L2 drift; a calibrated depth-expansion procedure
can safely quantize more layers than fixed-depth recipes.

Plausibility: moderate but needs a fresh gate. It cannot reuse CLE tightness
ratios as success criteria.

### 3. LM-Head-Calibrated Bound Repair

Hypothesis: replacing raw Frobenius/Lipschitz scaling with a calibration term
derived from LM-head sensitivity could yield a tighter predictor.

Potential paper claim: a calibrated predictor enables better layer precision
allocation. This is lower priority unless paired with a concrete allocator and
efficiency-vs-quality win.

Plausibility: low-to-moderate as a standalone theory paper; stronger only if it
becomes part of a positive compression method.

## Recommended Next Action

If this branch is pivoted, the highest-value fresh preregistration is
`damped_fp4_precision_allocator`. It should be a new branch with fresh
thresholds and held-out data, not a modification of
`preregister_cle_theoretical.md`.
