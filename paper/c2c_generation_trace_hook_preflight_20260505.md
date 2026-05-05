# C2C Generation Trace Hook Preflight

- date: `2026-05-05`
- status: `trace_collector_implemented_local_generation_blocked`
- COLM_v2 readiness impact: strengthens reproducibility and explains why C2C
  generation-time comparisons remain claim-bounded
- ICLR readiness impact: next positive-method gate now needs a C2C-compatible
  generation environment or a native CUDA run

## Purpose

The previous SVAMP32 C2C-teacher sparse-packet preflight showed an oracle
1-byte C2C residue sidecar but no deployable source-derived predictor. The
next gate is to collect richer generation-time dense-teacher traces, then train
a source-causal sparse residual packet from those traces.

This preflight implements the trace collection path without changing C2C
generation behavior.

## Implementation

Updated `latent_bridge/c2c_eval.py`:

- added projector trace-history capture across C2C calls;
- added generation logit-history summaries from the unmodified C2C
  `generate(..., return_dict_in_generate=True, output_scores=True)` path;
- added `extract_c2c_generation_trace_features`, which concatenates projector
  history and target-logit history into one feature vector.

Updated `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`:

- added `--feature-family generation_summary_trace`;
- routes generation traces through the existing strict residue-syndrome
  decoder.

Added `scripts/run_c2c_generation_trace_smoke.py`:

- one-row smoke collector for `.debug/` trace validation;
- records feature shape, feature hash, decoded prediction, projector history
  lengths, and target-logit step count.

Updated `tests/test_c2c_mechanism_trace.py` with generation-history and
logit-history schema tests.

## Local Smoke Results

The code-level trace schema tests pass, but local generation is blocked by the
current Mac/C2C runtime:

- unmodified C2C generation on MPS fails with an Apple MPS matmul incompatible
  dimensions error:
  `tensor<1x16x60x128xbf16>` by `tensor<1x8x128x60xbf16>`;
- trace-enabled generation on MPS fails the same way, so this is not caused by
  the new trace collector;
- trace-enabled generation on CPU reaches the vendored C2C wrapper, then fails
  because the wrapper expects the older `DynamicCache.key_cache/value_cache`
  API while the current installed Transformers `DynamicCache` exposes
  layer-based cache objects.

## Commands Run

Code checks:

```bash
./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py -q
./venv_arm64/bin/python -m py_compile \
  latent_bridge/c2c_eval.py \
  scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  scripts/run_c2c_generation_trace_smoke.py
git diff --check
```

Trace smoke attempted:

```bash
PYTHONUNBUFFERED=1 HF_HOME=.hf_home HUGGINGFACE_HUB_CACHE=.hf_home/hub \
  HF_DATASETS_CACHE=.hf_home/datasets TRANSFORMERS_CACHE=.hf_home/hub \
  TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_c2c_generation_trace_smoke.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device mps \
  --max-new-tokens 8 \
  --limit 1 \
  --residual-projection-dim 4 \
  --output-json .debug/c2c_generation_trace_smoke.json
```

Unmodified generation check attempted:

```bash
PYTHONUNBUFFERED=1 HF_HOME=.hf_home HUGGINGFACE_HUB_CACHE=.hf_home/hub \
  HF_DATASETS_CACHE=.hf_home/datasets TRANSFORMERS_CACHE=.hf_home/hub \
  TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device mps \
  --max-new-tokens 4 \
  --limit 1 \
  --prediction-output .debug/c2c_generation_unmodified_mps.jsonl
```

## NVIDIA / Compatible Runtime Runbook

On a machine where the published C2C artifact generates successfully, run:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --candidate text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --feature-family generation_summary_trace \
  --residual-projection-dim 4 \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --device cuda \
  --max-new-tokens 16 \
  --date 2026-05-05 \
  --output-json results/svamp32_c2c_generation_trace_syndrome_probe_20260505/generation_trace_probe.json \
  --output-md results/svamp32_c2c_generation_trace_syndrome_probe_20260505/generation_trace_probe.md
```

Pass bar: matched `>=14/32`, target-self wins preserved, at least `2/6` clean
C2C residual IDs recovered, and zero-source/source-shuffle/label-shuffle/
target-only/slots-only controls recover no clean residual IDs.

## Decision

The trace collector is ready, but the Mac runtime cannot currently execute
C2C generation. The next ICLR gate requires either:

1. a compatible CUDA/NVIDIA C2C run using the command above; or
2. a local compatibility fix for the vendored C2C wrapper and current
   Transformers cache API, followed by the same one-row smoke and full n32
   generation-trace syndrome probe.

Lay explanation: we added the recording equipment, but the local C2C engine
cannot currently drive on this Mac runtime. The next step is to run the same
recorder in an environment where C2C generation works, then see whether the
new trace can teach a tiny packet to mimic C2C's useful corrections.

## 2026-05-05 Local Cache-Compatibility Update

Added a contained compatibility shim in `latent_bridge/c2c_eval.py` so the
vendored C2C wrapper can read current Transformers `DynamicCache.layers`
through the older `key_cache` / `value_cache` interface. This avoids modifying
the reference C2C clone and lets the CPU generation path run far enough to
collect traces.

Validation:

- shim unit coverage in `tests/test_c2c_mechanism_trace.py`;
- `tests/test_c2c_mechanism_trace.py tests/test_c2c_eval.py`: `9 passed`;
- one-row CPU generation-trace smoke: pass, producing a
  `10080`-dimensional `c2c_generation_projector_and_logit_trace_history`
  feature vector with 28 projector histories and 4 target-logit steps;
- full SVAMP32 CPU generation-trace syndrome probe:
  `results/svamp32_c2c_generation_trace_syndrome_probe_20260505/generation_trace_probe.json`.

Full probe result:

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 12/32 | 0 | 3 |
| zero_source | 14/32 | 0 | 3 |
| shuffled_source | 9/32 | 0 | 1 |
| label_shuffled | 13/32 | 0 | 3 |
| target_only | 14/32 | 0 | 3 |
| slots_only | 8/32 | 0 | 0 |

The gate fails: generation-summary traces recover `0` clean C2C-residual IDs
and underperform zero-source/target-only controls.

Important runtime caveat: a separate local CPU C2C generation smoke with the
shim runs but degenerates into repeated Korean glyph tokens on the first four
SVAMP rows (`0/4`, while the archived MPS C2C teacher had `1/4` on those
rows). The cache shim therefore unblocks instrumentation, but current Mac CPU
C2C generation should not be treated as faithful native C2C teacher evidence.

Decision: mark the current generation-summary trace branch as weakened. The
next ICLR gate should either use a native C2C-compatible runtime for teacher
traces or move to a teacher-logit/KV-delta distillation artifact captured under
a runtime that reproduces the archived C2C teacher behavior. Do not claim C2C
distillation from the local CPU traces.
