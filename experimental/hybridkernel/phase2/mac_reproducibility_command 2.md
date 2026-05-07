# HybridKernel Mac Reproducibility Command

- date: 2026-05-06
- scope: owned Mac-local checks only; excludes external/vendor and GPU-only
  tests

## Stable Command

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/hybridkernel/phase0/tests \
  experimental/hybridkernel/phase2/tests \
  experimental/hybridkernel/phase3/tests \
  experimental/hybridkernel/phase4/tests -rs
```

This validates the local preflight, profiler parser/checker/generator tests,
the CPU reference, and Triton interpreter correctness. It intentionally leaves
the non-interpreter CPU-backend test skipped unless explicitly opted in.

## CPU-Backend Caveat

The experimental Triton CPU backend is useful for plumbing checks, but it is
not a stable paper gate on this Mac. In fresh shells:

- `/usr/bin/gcc` fails to link the kernel with `ld: library 'gcc' not found`.
- `CC=/opt/homebrew/bin/gcc-14` reaches a different Darwin linker failure:
  `ld: library not found for -lSystem`.

Therefore the camera-ready paper should cite Triton interpreter correctness as
the stable Mac gate and describe CPU-backend execution as an optional
environment-dependent diagnostic, not a required result.
