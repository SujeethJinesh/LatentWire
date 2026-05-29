# C6 Residual Reference + Systems Spec

Status: `SPEC_READY_KERNEL_BLOCKED_ON_METHOD_PASS`.

The PyTorch reference and systems cost model define:

`y = x W_pq^T + x_P (W_fp - W_pq)_P^T`

Current sidecar sizes from the prior Granite top-8x32 proxy are 54.0 MiB FP16, with 336.25 MiB for the top-25 tensor / 64-column sketch. pJ/token is not measured; it requires a kernel/profiler path after a method passes quality.
