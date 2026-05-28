# Protected-Column Residual Correction Spec

Date: 2026-05-28

Scope: CPU design artifact only. This packet does not implement Triton/CUDA,
does not launch GPU jobs, and does not claim latency evidence. It supersedes
the older generic `artifacts/kernel_design/` correction sketch for the
rotation-first sprint by focusing on ParoQuant residual correction:

```text
y = x W_pq^T + x_P DeltaW_P^T
DeltaW = W_fp - W_pq
```

## Status Context

Paper readiness is not ICLR-ready. The current story is rotation-first:
ParoQuant-style rotation is a strong positive baseline on Granite and Nemotron,
but ParoQuant is not our method. The open method question is whether
drift-aware rotation calibration, surface/branch choice, or residual correction
can beat or robustify the ParoQuant baseline on held-out traces.

This artifact supports only the residual-correction branch. It is not a
candidate result by itself. A GPU/kernel implementation should wait until a
residual-correction policy survives smoke and partial evaluation.

## Method

Given a quantized rotated weight matrix `W_pq` and original higher-precision
weight `W_fp`, define:

```text
DeltaW = W_fp - W_pq
```

For a selected protected input-column set `P`, compute:

```text
base_out = x @ W_pq.T
correction = x[:, P] @ DeltaW[:, P].T
y = base_out + correction
```

This is exactly equivalent to multiplying by a mixed matrix where columns in
`P` use `W_fp` and all other columns use `W_pq`.

## Column Selection

The default selector for the residual-correction smoke branch is:

```text
score_i = E_cal[x_i^2] * ||DeltaW[:, i]||_2^2
P = top_k(score)
```

This approximates the oracle objective `h_i(t) * x_i(t)^2` with a cheap
downstream sensitivity proxy derived from the residual norm. It differs from
WJAC because it uses the quantization residual `DeltaW`, not the full weight
column norm. The selector must be fit on calibration traces only; confirmation
traces must be held out.

Optional selectors for later gates:

- KL candidate pool: restrict `P` to columns that appear in an offline KL-look
  pool, then rank by the residual score above.
- Tail-trace pool: fit `P` only on calibration traces identified as ParoQuant
  tail risks, then confirm on held-out tail traces.

## API Sketch

```python
build_residual_sidecar(
    weight_fp: Tensor,          # [out_features, in_features]
    weight_pq: Tensor,          # [out_features, in_features]
    protected_idx: Tensor,      # [p], sorted input-channel indices
) -> ResidualSidecar

select_residual_columns(
    activation_second_moment: Tensor,  # [in_features]
    delta_weight: Tensor,              # [out_features, in_features]
    k: int,
) -> Tensor                            # [k], sorted by descending score

protected_column_correction(
    x: Tensor,                         # [..., in_features]
    weight_pq: Tensor,                 # [out_features, in_features]
    sidecar: ResidualSidecar,
    bias: Tensor | None = None,        # [out_features]
    base_out: Tensor | None = None,    # optional [..., out_features]
) -> Tensor                            # [..., out_features]
```

Production kernels would replace `weight_pq` with the packed ParoQuant/W4A16
representation and consume a precomputed `base_out` from the base kernel.

## Tensor Shapes

| Symbol | Meaning | Shape |
|---|---|---|
| `B` | batch size | scalar |
| `T` | tokens, or decode microbatch length | scalar |
| `M` | flattened rows, usually `B*T` | scalar |
| `K` | input channels | scalar |
| `N` | output channels | scalar |
| `P` | protected columns | scalar |
| `x` | activations | `[M, K]` or `[..., K]` |
| `W_fp` | fp16/bf16 reference weight | `[N, K]` |
| `W_pq` | ParoQuant-dequantized weight | `[N, K]` |
| `DeltaW_P` | sidecar residual columns | `[N, P]` |
| `protected_idx` | selected input columns | `[P]` |
| `y` | output | `[M, N]` or `[..., N]` |

For MoE, build one sidecar per expert or per expert group:

```text
W_fp[e]      : [N_e, K_e]
W_pq[e]      : [N_e, K_e]
DeltaW_P[e] : [N_e, P_e]
```

Routed tokens must use the sidecar for their active expert. This artifact does
not define a fused MoE gather kernel.

## Reference Semantics

The reference implementation must satisfy:

```text
correction_form(x, W_pq, DeltaW_P, P) == explicit_mixed_weight(x, W_mix)
```

where:

```text
W_mix = W_pq
W_mix[:, P] = W_fp[:, P]
```

The PyTorch reference in this directory includes a CPU self-test for:

- correction equals explicit mixed-weight matmul;
- selector returns high residual-impact columns;
- optional `base_out` gives the same result as recomputing the base output;
- overhead estimates are finite and monotonic in `P`.

## Overhead Model

Let `M` be flattened tokens, `K` input channels, `N` output channels, and `P`
protected columns.

Base dense matmul work:

```text
base_flops ~= 2 * M * N * K
```

Correction work:

```text
correction_flops ~= 2 * M * N * P
relative_flop_overhead ~= P / K
```

Sidecar storage:

```text
sidecar_bytes ~= N * P * bytes(delta_dtype) + P * bytes(index_dtype)
```

Extra activation gather/read:

```text
x_P_bytes ~= M * P * bytes(x_dtype)
```

This is acceptable only if the quality gain is real or if the branch cuts the
tail without increasing median loss. At `P/K = 10%`, the correction GEMM can be
large enough that deployment needs a fused or batched implementation; a
separate skinny GEMM is still the right correctness-first prototype.

## Benchmark Plan

No GPU benchmark should run from this artifact. If a residual policy survives
evaluation, benchmark in three gates:

1. CPU/reference:
   - randomized `[M, K, N, P]` shape tests;
   - exact comparison to explicit mixed-weight matmul;
   - selector determinism and calibration/confirmation split audit.
2. GPU microbenchmark:
   - `M in {1, 8, 32, 128}`;
   - model-realistic `K, N`;
   - `P/K in {0.01, 0.03, 0.05, 0.10}`;
   - compare base ParoQuant, base plus separate correction GEMM, and fused
     correction if implemented later.
3. End-to-end:
   - fixed prompt slice, seeds, sidecars, and quantization config;
   - report quality deltas separately from latency;
   - include held-out traces and at least one cross-family model.

Metrics:

- median recovery against static/ParoQuant baseline;
- CI lower bound and CVaR tail;
- p50/p95 latency;
- memory overhead;
- launches per token;
- maximum absolute and relative output error versus explicit mixed reference.

## Risks

- Residual correction may only tune ParoQuant tails on calibration traces and
  fail confirmation.
- The selector can collapse to magnitude if `||DeltaW_i||^2` is nearly flat.
- The sidecar can erase W4A16 memory gains if `P` is large.
- Dynamic `P` can interfere with CUDA graph capture. First implementation
  should use fixed per-layer `P` from calibration.
- The correction covers input-column residuals only. If errors concentrate in
  output rows or rotation surfaces, this branch will underperform.
- A separate skinny GEMM adds launch overhead and extra output traffic.
- This method is not novel if described as "mixed precision columns"; the
  defensible claim is narrower: drift-aware, held-out residual correction after
  rotation in long-reasoning W4A16.
- Kernel speed cannot rescue a method that fails ParoQuant-held-out quality
  gates.

## Gate Recommendation

Do not implement Triton/CUDA yet. Promote to GPU smoke only if C2/C3/C5
identify ParoQuant tail traces or residual-impact columns with a calibration
split and the command uses ParoQuant as the explicit baseline.

Recommended status for this artifact: `SPEC_READY_KERNEL_BLOCKED_ON_METHOD`.
