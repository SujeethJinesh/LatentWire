# C1 K-RES Residual-Headroom Gate

Status: `KILL_CURRENT_TOP8X32_PROXY_NO_GPU`.

This gate used the existing ParoQuant tight-clip residual cache and tail-trace activation EMA. The candidate score is `EMA(x_i^2) * ||DeltaW_i||_2^2` in the post-rotation residual basis.

## Readout

- Measured tensors: 289.
- Median relative residual energy: 0.0102.
- Mean top-128 residual-energy fraction per tensor: 0.08873.
- Top-25 FP16 sidecar working set at 64 columns/tensor: 336.25 MiB.
- Current top-8x32 MoE sidecar working set: 54.00 MiB.
- Valid smoke result: recovery -12.294 versus tight ParoQuant reference 5.940 on Granite tail trace I_4.

## Decision

The current residual proxy is not promoted. It is not enough that residual columns can be identified; the first valid correction reintroduced a large tail loss after tight ParoQuant had already fixed that trace. Residual correction remains a design criterion only if a later CPU gate shows stronger concentration or a KLLOOK/bounded selector changes the candidate set before GPU.

Supporting table: `residual_benefit_summary.csv`.
