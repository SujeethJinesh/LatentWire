# LatentWire Dense Baseline Smoke, 2026-05-05

This artifact records the local work done to turn the C2C/KVComm caveat into
an attempted matched-baseline gate rather than a purely textual limitation.

## Converted MCQA Generation Rows

`scripts/materialize_mcqa_generation_eval.py` converts ARC-Challenge and
OpenBookQA rows into letter-only generation prompts so generation-oriented C2C
and KVComm evaluators can run against the same public multiple-choice surfaces.

Generated smoke files:

- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_train_generation_smoke8.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_test_generation_smoke4.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_train_generation_smoke8.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_test_generation_smoke4.jsonl`

## C2C Smoke

Command family:

```bash
./venv_arm64/bin/python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/dense_baseline_mcqa_smoke_20260505/openbookqa_test_generation_smoke4.jsonl \
  --device mps \
  --max-new-tokens 4 \
  --limit 4 \
  --prediction-output results/dense_baseline_mcqa_smoke_20260505/c2c_openbookqa_smoke4.jsonl
```

Local result:

| Task | rows | exact generation accuracy | parsed letter accuracy | mean latency |
|---|---:|---:|---:|---:|
| OpenBookQA smoke | 4 | 0.000 | 0.250 | 1.300s |
| ARC-Challenge smoke | 4 | 0.000 | 0.750 | 2.756s |

The exact generation matcher is too strict for MCQA because C2C often emits a
letter plus candidate text, so `scripts/summarize_mcqa_generation_smoke.py`
adds an answer-letter parser for this smoke gate. These rows prove that the
local C2C runtime path can execute on converted matched-task prompts; they are
not a full C2C head-to-head baseline.

## KVComm Smoke

KVComm was attempted on the same OpenBookQA smoke rows. Local fixes added in
this pass:

- eager attention selection via `--attn-implementation eager`;
- current Transformers `BatchEncoding` handling for Qwen chat templates;
- `DynamicCache.key_cache` / `value_cache` compatibility shim reuse.

The smoke still fails inside the ported Qwen3 communicator with a cache/mask
length mismatch:

```text
RuntimeError: The size of tensor a (73) must match the size of tensor b (146)
at non-singleton dimension 3
```

This is now a concrete compatibility blocker rather than an unattempted
baseline. The next dense-baseline gate is to fix KVComm's Qwen3 cache-position
and attention-mask construction, then scale both C2C and KVComm from smoke rows
to matched ARC/OpenBookQA slices with parsed-letter scoring and target-only
controls.
