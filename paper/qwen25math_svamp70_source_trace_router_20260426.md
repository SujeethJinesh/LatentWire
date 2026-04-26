# Qwen2.5-Math -> Qwen3 SVAMP70 Source-Trace Router

- date: `2026-04-26`
- status: `fails_live_holdout_gate`
- scale-up rung: medium live-CV plus frozen holdout falsification
- live branch entering run: source-trace self-consistency router over the
  strongest Qwen2.5-Math -> Qwen3 SVAMP70 source surfaces

## Question

The fixed source-quality sidecar guards failed holdout because their apparent
clean wins were explained by controls. This run asks whether a richer
source-derived text-trace router can preserve the 1-byte source-sidecar idea:
accept the source sidecar only when the source answer is supported by simple
trace evidence such as valid equations, prompt-number coverage, and reuse of
the final answer in the trace.

The key new control is `equation_permuted`: keep the source text identity and
answer fixed, but rotate equation result numbers before feature extraction. A
defensible trace router should lose source-necessary clean wins under this
control.

## Command

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_trace_router_gate.py \
  --live-target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --live-source source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --live-candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --live-eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --holdout-target target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --holdout-source source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --holdout-candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-eval-file results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/_artifacts/svamp_chal101_170_70.jsonl \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --outer-folds 5 \
  --accept-penalty 0.10 \
  --output-json results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json \
  --output-md results/qwen25math_svamp70_source_trace_router_20260426/trace_router.md
```

## Result

Frozen full-live rule:

- feature: `source_answer_reused_in_trace`
- direction: `ge`
- threshold: `0.5`
- train help: `3`
- train harm: `0`
- train accept: `6`

Live CV:

- matched correct: `20/70`
- clean source-necessary: `1`
- clean control union: `0`
- accepted harm: `2`
- equation-permuted retained source-necessary: `0`
- failing criteria: `min_correct`, `min_clean_source_necessary`,
  `max_accepted_harm`

Holdout frozen:

- matched correct: `10/70`
- clean source-necessary: `1`
- clean control union: `0`
- accepted harm: `0`
- equation-permuted retained source-necessary: `1`
- failing criteria: `min_clean_source_necessary`,
  `equation_permutation_loses_half`

## Decision

Kill the source-trace self-consistency router as the next rescue for the
fixed source-sidecar branch. The clean standard-control behavior is useful, but
the row is too weak and the holdout clean win does not depend on the validity
of the equation trace. It is not enough evidence for source communication.

This also weakens the broader hypothesis that shallow text-level source-quality
features can recover the original SVAMP70 sidecar positive. The next
highest-value branch is source-surface discovery or hidden-state/source-logit
confidence capture during generation, not another fixed text guard.

## Artifacts

- result JSON:
  - `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json`
  - sha256: `e4e5600e139efbf7bc068ff2117e172cba9f87055e9477f51839a90175c54c03`
- readout:
  - `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.md`
  - sha256: `e439be6bf00ec99ccf9f63aa72ca0b3a47f423d4a950155846d7417322e99a6c`
- analyzer:
  - `scripts/analyze_svamp_source_trace_router_gate.py`
  - sha256: `92828099f0ccc3188fb49f8171f1e0d0ce4260a27e1780459a8789a3f31e03e5`
- tests:
  - `tests/test_analyze_svamp_source_trace_router_gate.py`
  - sha256: `223a2369f79856847816c5b0963e0fd48d620fdc565bd15e588647071664e05b`

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_trace_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_trace_router_gate.py`

## Next Exact Gate

Run a new source-surface discovery pass that explicitly searches for stronger
source-only over target/text examples, or add a generation-time source
confidence/logit artifact so the next router is based on model-internal
source evidence rather than shallow source text.
