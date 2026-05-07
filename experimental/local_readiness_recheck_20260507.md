# Local Readiness Recheck

- date: 2026-05-07
- scope: HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8
- purpose: verify that the Mac-side packet validators, paper-boundary tests,
  and Triton-interpreter correctness gates are clean before interpreting any
  native 5090 evidence.

## Result

```text
291 passed, 1 skipped, 2 warnings in 11.63s
```

The skipped test is the opt-in non-interpreter Triton CPU-backend check for the
HybridKernel toy primitive:

```text
experimental/hybridkernel/phase4/tests/test_boundary_triton_cpu_backend.py
set HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 to run this opt-in gate
```

The warnings are import-time `SwigPyPacked` / `SwigPyObject` deprecation
warnings and do not affect the project gates.

## Command

Run from the repository root:

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/tests \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests \
  experimental/ssq_lr/phase2/tests \
  experimental/horn/phase2/tests \
  experimental/hbsm/phase2/tests \
  experimental/shared/tests \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -q -rs
```

## Interpretation

This recheck is not model evidence. It only says the local validators, packet
contracts, paper-boundary tests, and Triton-interpreter correctness checks are
ready to evaluate the next admissible artifacts:

After the paper edits, the HybridKernel, SSQ-LR, and ThoughtFlow-FP8 PDFs were rebuilt with
`latexmk -pdf -interaction=nonstopmode -halt-on-error`.

- HybridKernel: native 5090 Nsight/vLLM full-matrix packet.
- SSQ-LR: a newly preregistered Mac rescue recipe/layer rule, if pursued.
- HORN: no work unless a new full H2/H3 preregistration exists.
- HBSM: no work unless a new narrower mechanism preregistration exists.
- ThoughtFlow-FP8: paper-only copyedit/human review.
