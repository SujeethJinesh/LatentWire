# Kernel Design Spec: Policy Update and Protected-Column Correction

Date: 2026-05-28

Scope: design only. This artifact does not implement a Triton or CUDA kernel,
does not run GPU jobs, and does not claim throughput evidence.

## Project Status Context

Paper readiness remains below ICLR threshold. The current story is that static
channel protection fails under decode drift, while budgeted adaptive protection
is still the live positive-method direction only if it survives stronger frozen
evaluation. The immediate blocker is not kernel speed. It is whether surviving
methods beat strict baselines on larger frozen slices, seed repeats, paired
uncertainty, and cross-family falsification.

This design supports future finalist methods only after the evaluation gate is
cleared. It is based on the current ledger and decision state, especially:

- `RUN_LEDGER.md`: kernel design is queued as a CPU artifact, with no GPU work.
- `DECISIONS.md`: WJAC is provisionally killed, LAMBDA and HYST remain smoke
  candidates, and budgeted EMA is the live adaptive-protection pattern.
- `experimental/outlier_migrate/phase9/preregister_om_phase9_funnel_smoke.md`:
  HYST means flat top-10 budget with hysteretic protected-set updates.
- `experimental/outlier_migrate/phase4/run_om_phase4_intervention.py`:
  protected channels are currently preserved by restoring rows and columns
  after symmetric INT4 dequantization.

## Kernel 1: Policy Kernel

### Purpose

Maintain protected-channel policy state during long decode:

1. combine the current salience signal with optional WJAC score;
2. update an EMA score per layer and channel;
3. update the protected mask with hysteresis;
4. enforce a per-layer budget.

This kernel should run at policy-update cadence, not on every matmul. Current
experiments use 100-token update cadence and score a final snapshot, so this is
not expected to sit in the hottest W4A16 GEMM path.

### API Sketch

```python
policy_update(
    ema_scores: Tensor,          # [R, C], fp32/fp16
    current_scores: Tensor,      # [R, C], same shape, already calibrated
    prev_mask: Tensor,           # [R, C], bool
    *,
    alpha: float,                # next = alpha * current + (1 - alpha) * ema
    enter_k: int,                # channels entering by top-k rank
    exit_k: int,                 # previous channels retained until below top-k
    max_k: int | None,           # optional hard cap, usually >= enter_k
    wjac_scores: Tensor | None,  # [R, C], optional additive score
    wjac_weight: float,
) -> tuple[Tensor, Tensor]       # next_ema [R, C], next_mask [R, C]
```

`R` is a row axis that can represent layers, flattened layer-expert pairs, or a
padded/ragged layer group. The PyTorch reference uses dense `[R, C]` tensors.
For models with heterogeneous channel counts, production should either launch
one row group per common `C`, or use a metadata table with row offsets.

### Expected Shapes

| Symbol | Meaning | Typical form |
|---|---|---|
| `R` | policy rows, usually transformer layers | 24-80 |
| `C` | hidden channels per row | 2048, 4096, 8192, etc. |
| `ema_scores` | previous EMA salience | `[R, C]`, fp32 preferred |
| `current_scores` | current activation, marginal, or lambda score | `[R, C]` |
| `wjac_scores` | optional WJAC diagnostic score | `[R, C]` |
| `prev_mask` | prior protected set | `[R, C]`, bool |
| `next_mask` | next protected set | `[R, C]`, bool |

For a top-10 policy, `enter_k = max(1, ceil(0.10 * C))`. For HYST top-10,
`exit_k = min(C, 2 * enter_k)`, matching "enter at top-k, exit only below
top-2k". For a top-1 static diagnostic, `enter_k = max(1, ceil(0.01 * C))`
and `exit_k = enter_k`.

### Reference Semantics

The reference implements deterministic rank masks:

```text
combined = current_scores + wjac_weight * wjac_scores  # if WJAC provided
next_ema = alpha * combined + (1 - alpha) * ema_scores
enter = topk(next_ema, enter_k)
retain = topk(next_ema, exit_k)
candidate = enter OR (prev_mask AND retain)
next_mask = topk(next_ema restricted to candidate, max_k) if max_k else candidate
```

Ties are resolved by ascending channel index in the reference. A production
kernel should specify tie handling because nondeterministic mask churn can
invalidate paired comparisons.

### Triton/CUDA Feasibility

The arithmetic is simple and memory-bound. EMA, additive WJAC fusion, and
hysteresis Boolean updates are easy to fuse. The hard part is top-k/rank
selection per row.

Feasible routes:

- For moderate `C`, a Triton block per row can load a full row, update EMA,
  compute approximate or exact top-k, and write the next mask. Triton's
  programming model exposes per-program IDs and block tensor operations, which
  fit fixed-width row kernels.
- For large `C`, ragged `C`, or exact deterministic top-k, CUDA/CUB or a
  framework `topk` call may be more reliable than hand-rolled Triton. The
  top-k step can be two-pass: local block candidates followed by a row reducer.
- WJAC should be treated as an input score. Computing Jacobian or sensitivity
  scores is outside this kernel; otherwise policy overhead can dominate the
  method.

Risk gate: do not move this into the token-critical path unless profiler traces
show policy updates are material. It is initially a state-update utility.

## Kernel 2: Protected-Column Correction Kernel

### Purpose

Replace "restore protected input columns into the full dequantized weight" with
a runtime correction:

```text
y = base_W4A16(x) + x_P @ (W_fp16_P - W_dequant_P)^T
```

Here `P` is the protected input-channel set for the layer. The correction is
exactly equivalent to using a mixed weight matrix where protected columns are
kept in fp16/bf16 and all other columns use the W4A16 dequantized base.

### API Sketch

```python
protected_column_linear(
    x: Tensor,                    # [M, K] or [..., K], fp16/bf16/fp32
    packed_w4: Tensor,            # base W4 storage, backend-specific
    scales: Tensor,               # base W4 dequant scales, backend-specific
    protected_idx: Tensor,        # [P], sorted int32/int64 input channels
    delta_p: Tensor,              # [N, P], W_fp16_P - W_dequant_P
    *,
    bias: Tensor | None,          # [N]
    base_out: Tensor | None,      # optional precomputed base_W4A16(x), [..., N]
) -> Tensor                       # [..., N]
```

The PyTorch reference takes `weight_fp16` and `weight_dequant` instead of
`packed_w4` because it is a correctness oracle, not a packed W4 kernel.

### Expected Shapes

For a normal linear layer:

| Symbol | Meaning | Shape |
|---|---|---|
| `M` | flattened tokens or batch-token rows | `B * T` or decode batch |
| `K` | input hidden size | e.g. 2048-8192 |
| `N` | output hidden/intermediate size | e.g. 2048-28672 |
| `P` | protected input channels | `ceil(fraction * K)` |
| `x` | input activations | `[M, K]` or `[B, T, K]` |
| `W_dequant` | dequantized W4 base, row-major linear weight | `[N, K]` |
| `W_fp16` | original fp16/bf16 weight | `[N, K]` |
| `delta_p` | protected-column delta | `[N, P]` |
| `y` | output | `[M, N]` or `[B, T, N]` |

For MoE expert banks, use either grouped expert batches or flatten the expert
axis into independent rows: `W[e]` has shape `[N, K]`, `delta_p[e]` has shape
`[N, P_e]`, and routed tokens for expert `e` use that expert's protected set.

### Reference Semantics

Let `W_mix = W_dequant` except `W_mix[:, protected_idx] = W_fp16[:, protected_idx]`.
The correction kernel must match:

```text
y = x @ W_mix.T + bias
```

The reference verifies this by comparing the correction form against an explicit
mixed-weight matmul. It supports arbitrary leading dimensions on `x`.

### Triton/CUDA Feasibility

The base W4A16 GEMM should remain the primary kernel. The correction can be:

1. **Second-kernel skinny GEMM plus add**: easiest integration. Launch
   `x_P @ delta_p.T` as an fp16/bf16 GEMM and add to `base_out`. This is
   likely enough for correctness and first microbenchmarks, but adds a launch
   and rereads output.
2. **Fused epilogue inside W4A16 GEMM**: fastest route if the W4A16 backend
   exposes the input tile and output accumulator. Each output tile accumulates
   base W4 contributions, then loops over protected columns and accumulates
   `x[:, P] * delta_p[:, P]` before store. This avoids an extra launch but
   complicates backend integration.
3. **Packed protected-column sidecar**: store `protected_idx` and `delta_p`
   contiguously per layer. Sorting `protected_idx` improves input gather
   locality. If `P` is large, materializing `x_P` once per decode step may beat
   repeated irregular gathers inside every output tile.

CUTLASS is a plausible CUDA backend because it provides GEMM abstractions for
mixed precision and narrow integer types. Triton is plausible for the second
kernel or a custom fused prototype. Nsight Compute should be used later to
measure launch count, DRAM throughput, tensor-core utilization, and correction
kernel occupancy.

### Benchmark Plan

No benchmark in this artifact should be interpreted as GPU evidence. Future
benchmarking should proceed in gates:

1. **CPU/reference gate**
   - randomized shapes for `[M, K, N, P]`;
   - compare correction output to explicit mixed-weight matmul;
   - verify policy invariants: mask count cap, enter/retain hysteresis, no
     accidental WJAC effect when `wjac_weight = 0`;
   - include tie cases for deterministic channel selection.
2. **GPU microbenchmark gate**
   - shapes: `M in {1, 8, 32, 128}`, `K,N` from target model layers,
     `P/K in {0.01, 0.03, 0.10}`;
   - baselines: base W4A16 only, current restored-column full dequantized
     scoring path, base W4A16 plus separate correction GEMM, and fused
     correction if implemented;
   - metrics: p50/p95 latency, launches, bytes read/written, achieved
     bandwidth, achieved tensor-core utilization where applicable, max
     absolute/relative error against fp16 mixed reference.
3. **End-to-end finalist gate**
   - run only after the method survives frozen evaluation;
   - hold prompt slice, seed, protected sets, and quantization config fixed;
   - report paired quality deltas separately from latency;
   - include same-family and cross-family rows so kernel speed does not mask
     method failure.

### Risks

- The policy kernel may be irrelevant to latency because updates occur every
  100 tokens and top-k selection is not in the main GEMM path.
- WJAC score computation can dwarf the policy update if it is not cached or
  approximated.
- Hysteresis can reduce churn but lag real drift; this is an evaluation risk,
  not just a kernel risk.
- Dynamic masks can interfere with CUDA graph capture or require recapture if
  protected sets change during serving.
- For `P/K = 0.10`, the correction GEMM may be too large to beat simply storing
  restored mixed weights or using a backend-supported mixed-precision path.
- Irregular protected-column gathers may destroy memory coalescing unless
  indices are sorted, grouped, or materialized.
- The correction formula covers input-column protection only. Existing
  experiments also restore output rows when a hidden output axis is protected.
  A row-correction design would need a separate epilogue or output overwrite.
- Delta sidecar storage can erode the W4 memory win if many layers carry large
  protected sets.
- Kernel success does not rescue a method that fails source-index, random,
  same-budget, or cross-family controls.

## Sources

- Repo ledger and decisions: `RUN_LEDGER.md`, `DECISIONS.md`.
- Current HYST smoke preregistration:
  `experimental/outlier_migrate/phase9/preregister_om_phase9_funnel_smoke.md`.
- Current protected-row/column implementation:
  `experimental/outlier_migrate/phase4/run_om_phase4_intervention.py`.
- Current EMA-style implementation:
  `experimental/outlier_migrate/phase9/run_om_phase9_m11_ema_drift.py`.
- Triton language programming model:
  <https://triton-lang.org/main/python-api/triton.language.html>.
- NVIDIA CUTLASS overview:
  <https://docs.nvidia.com/cutlass/latest/overview.html>.
- NVIDIA Nsight Compute CLI:
  <https://docs.nvidia.com/nsight-compute/NsightComputeCli/>.
