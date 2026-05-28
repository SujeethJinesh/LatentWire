# C12 Kernel Spec Provenance

Date: 2026-05-28

Working directory: `/workspace/LatentWire`

Current commit when authored:

```text
321b690b049267fd016c5ed6615fc6e331ac9450
```

Branch:

```text
main
```

Inputs read:

- `artifacts/kernel_design/spec.md`
- `artifacts/kernel_design/pytorch_reference.py`
- User task: C12 protected-column correction spec for rotation-first sprint.

Files intentionally not edited:

- `RUN_LEDGER.md`
- `DECISIONS.md`
- `artifacts/kernel_design/`

GPU work:

```text
none
```

Triton/CUDA work:

```text
none
```

Scope:

This artifact defines the PyTorch reference and design gate for:

```text
y = x W_pq^T + x_P (W_fp[:, P] - W_pq[:, P])^T
```

It does not claim that residual correction beats ParoQuant. That requires a
future calibration/confirmation evaluation.
