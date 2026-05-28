# DriftRot Residual Cache: Granite Tight Clip

Run: `artifacts/rot_resid_correction/residual_cache_granite_tight_20260528T2025Z`

Status: `RESIDUAL_CACHE_READY_ACTIVATION_EMA_NEEDED`.

This cache measures per-input-column residual energy after applying the tight ParoQuant-style clip `[0.5, 2.0]`. It does not yet include long-decode activation EMA, so it is a residual candidate pool, not a runnable residual-correction policy.

## Key Numbers

- measured tensors: 289
- skipped tensors: 154
- top-25 FP16 working set at 64 columns/tensor: 336.25 MiB
- pJ/token: not measured; requires profiler-backed reference or kernel path

## Top Residual-Energy Tensors

| Rank | Tensor | Relative residual energy | Top-128 residual fraction | FP16 working set MiB @64 cols |
|---:|---|---:|---:|---:|
| 1 | `model.embed_tokens` | 0.009974 | 0.0402 | 12.25 |
| 2 | `model.layers.2.block_sparse_moe.input_linear` | 0.010103 | 0.0379 | 13.50 |
| 3 | `model.layers.39.block_sparse_moe.input_linear` | 0.010142 | 0.0325 | 13.50 |
| 4 | `model.layers.38.block_sparse_moe.input_linear` | 0.010138 | 0.0318 | 13.50 |
| 5 | `model.layers.24.block_sparse_moe.input_linear` | 0.010153 | 0.0323 | 13.50 |
| 6 | `model.layers.37.block_sparse_moe.input_linear` | 0.010140 | 0.0318 | 13.50 |
| 7 | `model.layers.14.block_sparse_moe.input_linear` | 0.010124 | 0.0323 | 13.50 |
| 8 | `model.layers.15.block_sparse_moe.input_linear` | 0.010129 | 0.0322 | 13.50 |
| 9 | `model.layers.28.block_sparse_moe.input_linear` | 0.010146 | 0.0321 | 13.50 |
| 10 | `model.layers.35.block_sparse_moe.input_linear` | 0.010132 | 0.0318 | 13.50 |

## Interpretation

The cache unlocks the next residual-correction step: combine post-ParoQuant residual column energy with long-decode activation EMA on the same tensor input columns. Residual energy alone is not enough for a method claim because it ignores which columns are active on tail traces.

The top residual tensors are dominated by MoE input linears plus the embedding matrix. A practical correction cannot correct every listed tensor without reporting HBM bytes/token and latency/energy overhead. The first smoke should use a small top-tensor/top-column budget on Granite tail trace 4 and representative traces 7/9/10.

## Next Gate

Collect or approximate activation EMA for the top residual tensors on Granite tail/representative traces, then build a protected-column pool using:

`score_i = EMA(x_i^2) * ||(W_fp - W_pq)[:, i]||_2^2`

Do not run residual-correction scoring until this activation factor exists.
