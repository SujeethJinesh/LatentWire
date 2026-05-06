# SinkKV Preregistered Mac Gate

- date: 2026-05-06
- status: preregistered before measurement
- branch: sink-protected mixed-precision KV cache

## Hypothesis

Uniform FP4-style KV quantization harms quality partly because sink K/V vectors
are high-leverage outlier positions. Keeping the first 1--4 sink positions at
BF16 or FP8 while quantizing all non-sink positions to simulated MXFP4 should
recover at least half of the quality degradation caused by uniform MXFP4.

## Rows

For every model/surface/length row:

- `bf16_kv`: unquantized reference K/V.
- `uniform_mxfp4_kv`: all K/V positions passed through simulated MXFP4.
- `sink_protected_mxfp4_kv`: sink positions restored from BF16, all other K/V
  positions simulated MXFP4.

## Initial Surfaces

Mac gate surfaces, in priority order:

1. small cached Q/K/V tensors from `distilgpt2` or Qwen-family proxy;
2. synthetic long-context K/V tensors with controlled sink mass;
3. larger model cached traces only if the first two surfaces show a signal.

Target lengths:

- `seq_len in {2048, 8192, 32768}` when cached tensors are available;
- smaller synthetic lengths are allowed only for code validation, not promotion.

## Metrics

Lower is better:

- continuation NLL when logits are available;
- attention-output relative L2 when only Q/K/V are available;
- sink-mass drift;
- optional task accuracy for cached reasoning traces.

## Promotion Rule

Promote to GPU implementation only if all are true:

- `(sink_FP4 - BF16) / (uniform_FP4 - BF16) <= 0.5` on at least two
  model/length surfaces;
- the minimum-row confidence interval excludes 1.0;
- sink-protected quality is not worse than uniform low precision on any
  measured row;
- the recipe fixes sink count, quantization format, block size, and exception
  rule before GPU work.

## Kill Rule

Kill if sink protection recovers less than 25% of the uniform-FP4 quality gap
on the first two real/proxy surfaces, or if the effect only appears on synthetic
tensors.

## Forbidden Retuning

Do not change sink count, block size, protected-position rule, or evaluation
surface after seeing gate results unless a new preregistration is written.

## Output Packet

Write results under:

```text
experimental/sinkkv/results/sinkkv_gate_<YYYYMMDD>_<model_slug>_<surface_slug>/
```

Required files:

- `config.json`
- `raw_rows.jsonl`
- `summary.json`
- `summary.md`
- `decision.md`
