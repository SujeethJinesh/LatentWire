# Vacation Decision - M2 Dynamic Cache Memory Cleanup

## Situation

The vacation-mode 12-trace Granite-Small M2 run
`experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_20260514T025000Z`
completed BF16, `static_1pct`, and `static_3pct`, then failed with CUDA OOM
while loading the next model instance for `m2_position_conditional`.

The traceback points to `shared.load_model_and_tokenizer(...).model.to(device)`
inside `score_dynamic_segments`, after the first dynamic segment had already
entered the segment-switch path. This is an implementation/runtime-memory issue,
not an M2 scientific decision.

## Options Considered

1. Treat the run as `FAIL_INFRA_M2` and move immediately to paper drafting.
2. Reduce trace count below 12.
3. Patch the runner to free the closure-held model references in dynamic
   segment switching, reuse the completed score caches in a fresh run
   directory, and resume the same 12-trace adapted protocol.

## Decision

Proceed with option 3.

The adaptation preserves the scientific question: same model, same 12 traces,
same scoring window, same M2 segments, same static-3% control, same random-bin
negative control, and same checker thresholds. The patch only fixes memory
lifetime in the runner and avoids recomputing already-completed deterministic
score caches.

## What Would Invalidate This Decision

If the human requires the original 24-trace packet or disallows score-cache
reuse after an infrastructure failure, the resumed 12-trace packet should be
treated as an exploratory vacation-mode packet rather than the formal M2 result.
