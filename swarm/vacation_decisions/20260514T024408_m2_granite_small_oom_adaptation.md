# Vacation Decision: M2 Granite-Small OOM Adaptation

## Situation

The Phase 9 M2 Granite-Small run
`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z`
completed BF16 scoring, static-1% scoring, and static-3% matched-cost
scoring for 24 traces, then failed with `FAIL_INFRA_M2` while entering
`m2_position_conditional`.

The root cause was a CUDA OOM during the initial M2 dynamic forward pass:
the runner entered the dynamic regime without `torch.inference_mode()` on
the initial cache-building forward, and it had no safe offload path for
cache tensors across segment-specific model reloads.

## Options Considered

1. Rerun the original 24-trace packet after the inference-mode fix.
   - Most faithful to the original M2 preregistration.
   - Estimated to take another full day or more before any paper-facing
     result lands.

2. Skip M2 as `FAIL_INFRA_M2` and proceed to M10.
   - Fastest path to another method attempt.
   - Leaves the highest-priority Phase 9 method untested after a fixable
     runner issue.

3. Run a vacation-mode 12-trace deterministic revision on prompts 0-11.
   - Authorized by vacation mode V2/V4 because the 24-trace packet was
     slower than planned and failed from an implementation/memory issue.
   - Keeps the same model, method, controls, thresholds, prompt order,
     scoring window, quantization scheme, and decision logic.
   - Lands a method result sooner with wider uncertainty and explicit
     documentation.

## Decision

Choose option 3.

I patched the runner to use inference mode for scoring forwards, to offload
dynamic cache state to CPU across M2 segment-specific model reloads, and to
write a `vacation_adaptation.json` file when running the fixed 12-trace
slice. The checker now accepts this adaptation only when the packet declares
Vacation mode V2/V4 authority and uses deterministic prompt indices 0-11
with at least 12 traces.

The failed 24-trace packet is preserved as `FAIL_INFRA_M2`. The next run
will use:

```bash
--trace-count 12
--reuse-activation-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z
--reuse-trace-run-dir experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_20260513T042800Z
```

## What Would Invalidate This Decision

If the human requires strict interpretation of the original 24-trace M2
preregistration before any M2 result can influence the paper, then this
12-trace vacation-mode result should be treated as exploratory and the
24-trace packet should be rerun later.
