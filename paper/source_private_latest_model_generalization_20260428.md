# Source-Private Latest-Model Generalization Scout

- date: `2026-04-28`
- gate: `latest_model_generalization_scout_20260428`
- status: model matrix added; latest-model claim remains unproven

## Current Readiness

The existing ICLR package is ready for the scoped claim. This scout is a
post-package strengthening gate for model-family breadth.

## Answer

The method should plausibly generalize to MoE models because the source-side
task is not architectural: the source reads a private diagnostic line and emits
a compact packet. A dense model, sparse MoE model, or quantized MoE deployment
can all succeed if they preserve exact instruction following and short-code
copying.

But we do not yet have evidence for Qwen3.5/Qwen3.6 or MoE rows. The paper
should not claim MoE/latest-model generalization until those rows pass the same
source-destroying controls.

## What I Checked

Official Hugging Face model cards/API identify these candidate rows:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.6-27B`

The Qwen3.6 `35B-A3B` row is the key MoE test: `35B` total parameters with `3B`
activated parameters. The FP8 row tests whether quantized deployment preserves
packet emission.

## Local Smoke Attempt

I attempted a local `n=16` smoke for `Qwen/Qwen3.5-0.8B`:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16 \
  --model Qwen/Qwen3.5-0.8B \
  --device mps \
  --dtype float32 \
  --limit 16 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

It failed before generation because the repo-local venv has
`transformers==4.51.0`, while the cached Qwen3.5 config uses
`model_type: qwen3_5` and declares `transformers_version: 4.57.0.dev0`.
Therefore this is a dependency/harness compatibility blocker, not evidence that
the method fails on Qwen3.5.

## Added Matrix

Artifacts:

- `results/source_private_latest_model_matrix_20260428/latest_model_matrix.md`
- `results/source_private_latest_model_matrix_20260428/latest_model_matrix.json`
- `results/source_private_latest_model_matrix_20260428/manifest.md`

Code:

- `scripts/build_source_private_latest_model_matrix.py`
- `tests/test_build_source_private_latest_model_matrix.py`

## Recommended Next Gate

1. Upgrade the repo-local `venv_arm64` Transformers dependency to a version that
   supports `qwen3_5`.
2. Run `Qwen/Qwen3.5-0.8B` `n=16` and `Qwen/Qwen3.5-2B` `n=16`.
3. If both pass, run `Qwen/Qwen3.5-4B` and widen the best small row to `n=64`.
4. Run `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.6-35B-A3B-FP8` off-machine at
   `n=32`, then `n=500` only if controls hold.

Pass rule remains unchanged: matched packets must beat target/no-source by at
least `15` points, and source-destroying controls must stay within `2` points
of target-only.

## Paper Impact

If Qwen3.5 small and Qwen3.6 MoE rows pass, the paper can strengthen its
external-validity claim from "works across Qwen3/Phi-3/Qwen2.5-era source
emitters" to "also transfers to latest small hybrid and sparse MoE source
emitters." Until then, keep the current scoped wording.
