# ThoughtFlow Triton Interpreter Note

This note keeps the Triton-related evidence inside the ThoughtFlow artifact
set. It is a correctness note only, not a native FP8, CUDA, latency,
throughput, HBM, or GPU-serving claim.

The Phase 4 tests exercise the anchor/phase int8 primitive through the local
Triton interpreter path and compare it against the CPU reference. The stable
local command is:

```bash
TRITON_CPU_BACKEND=1 \
TRITON_INTERPRET=1 \
TRITON_HOME="$PWD/.debug/triton_home" \
./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -q -rs
```

Interpretation:

- Passing tests support only indexing and numerical parity for the small
  anchor/phase reference primitive.
- They do not validate native CUDA execution, production FP8, real serving
  latency, or any sparse-cache method claim.
- The current paper uses the primitive as a bounded artifact in the
  falsification ladder, not as headline evidence.
