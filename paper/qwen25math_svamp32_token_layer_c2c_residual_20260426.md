# Qwen2.5-Math -> Qwen3 Token-Layer C2C Residual Probe

- date: `2026-04-26`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- readiness: not ICLR-ready
- commit at run time: `5bc3d04fc2950888d22be1ec8f98a68096fbe80d`

## Start Status

- current paper story: Qwen2.5-Math -> Qwen3 has a strong C2C surface on
  SVAMP32, but deployable source-derived probes recover `0/6` clean C2C-only
  IDs.
- exact blocker: determine whether C2C's headroom is carried by local
  projector residual tensors that could motivate a later source-derived
  distillation objective.
- live branch: C2C mechanism distillation diagnostic.
- scale-up rung: strict small diagnostic gate.

## Implementation

Added token/layer-local C2C feature extraction:

- `latent_bridge/c2c_eval.py` now records tail-token local tensors for each C2C
  projector and stream: `source`, `target`, `output`, and `delta`.
- variable-width projector tensors are padded to a common hidden width and
  recorded with `raw_hidden_dims`.
- `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py` now supports
  `--feature-family token_layer_tail_residual` and `--probe-model
  query_bottleneck`.
- the existing source-latent LOOCV evaluator now accepts metadata-backed token
  shapes via `feature_token_shape`.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --probe-model query_bottleneck \
  --feature-family token_layer_tail_residual \
  --shuffle-offset 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --device mps \
  --max-new-tokens 1 \
  --query-slots 8 \
  --query-epochs 80 \
  --query-lr 0.01 \
  --query-weight-decay 0.001 \
  --query-seed 0 \
  --output-json results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.json \
  --output-md results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.md
```

## Result

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 8 | 0 | 0 |
| zero_source | 8 | 0 | 0 |
| shuffled_source | 8 | 0 | 0 |
| label_shuffled | 8 | 0 | 0 |
| target_only | 8 | 0 | 0 |
| slots_only | 7 | 1 | 0 |

- feature shape: `[32, 229376]`
- token shape: `[224, 1024]`
- clean source-necessary IDs: `0/6`
- control clean union: `1/6`
- failing criteria: `min_correct`, `min_clean_source_necessary`,
  `control_clean_union_empty`

## Decision

Kill C2C summary/projection and tail-token local-residual mechanism probes as
live method-distillation branches on this surface. The C2C headroom remains a
useful upper bound and decision surface, but not a readable communication
signal under these probes.

Next exact gate:

- stop tuning C2C trace readouts unless a new objective changes the source of
  supervision
- select the next branch from either source-surface discovery or a deployable
  source-side method that does not depend on decoding C2C mechanism traces

## Artifacts

- result JSON:
  - `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.json`
  - sha256: `b2bfb8605b07c7a9f9d98d31fb35091e06457b42580e884f440be1684fba0b6e`
- result markdown:
  - `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.md`
  - sha256: `83ba897e191dd62b51706c2859a443cf2760a6fd6230a8ea7998d374c6c5b440`
- raw run log:
  - `.debug/qwen25math_svamp32_token_layer_c2c_residual_20260426/logs/probe_rerun.log`
  - sha256: `4fa03704c7c5a83ce60124bd99d0b8e54e5ebb003646522ba32d8b3c6c97bd98`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q
./venv_arm64/bin/python -m py_compile latent_bridge/c2c_eval.py scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py scripts/analyze_svamp32_source_latent_syndrome_probe.py
```

