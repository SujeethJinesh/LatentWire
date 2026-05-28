# Residual Candidate Pool: Granite Tail Trace 4

Residual cache: `artifacts/rot_resid_correction/residual_cache_granite_tight_20260528T2025Z/residual_column_cache.json`
Activation EMA: `artifacts/rot_resid_correction/activation_ema_granite_tail4_top8_20260528T2030Z/activation_ema.json`

Status: `CANDIDATE_POOL_READY_SMOKE_RUNNER_NEEDED`.

Score formula:

`score_i = EMA(x_i^2) * ||(W_fp - W_pq)[:, i]||_2^2`

## Top Modules / Columns

| Module | Hook count | Top score column | Top score | Activation EMA sq | Delta norm sq |
|---|---:|---:|---:|---:|---:|
| `model.layers.14.block_sparse_moe.input_linear` | 5110 | 3720 | 2.61701 | 3.14026 | 0.833373 |
| `model.layers.15.block_sparse_moe.input_linear` | 5110 | 3219 | 2.179 | 2.88355 | 0.755668 |
| `model.layers.2.block_sparse_moe.input_linear` | 5110 | 573 | 3.58959 | 4.1005 | 0.875403 |
| `model.layers.24.block_sparse_moe.input_linear` | 5110 | 990 | 1.34062 | 1.74201 | 0.769582 |
| `model.layers.28.block_sparse_moe.input_linear` | 5110 | 2053 | 3.39332 | 4.51064 | 0.752292 |
| `model.layers.37.block_sparse_moe.input_linear` | 5110 | 3979 | 17.7199 | 23.5872 | 0.751254 |
| `model.layers.38.block_sparse_moe.input_linear` | 5110 | 3016 | 7.72606 | 10.1426 | 0.761743 |
| `model.layers.39.block_sparse_moe.input_linear` | 5110 | 816 | 33.8029 | 41.4383 | 0.81574 |

## Interpretation

This is the first valid rotated-basis residual-correction candidate pool: it uses post-ParoQuant residual column energy and long-decode activation EMA on the tail trace. It is not yet a recovery result. The next step is a tiny reference implementation smoke on trace 4 that applies residual correction to a small number of modules/columns and compares against tight ParoQuant.

Because kernel/runtime cost matters, the smoke must report working-set bytes and should initially use a small budget such as top 8 modules x 32 columns.
