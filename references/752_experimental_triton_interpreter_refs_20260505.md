# Experimental Triton Interpreter References

This memo records the source used for the Macbook-side Triton interpreter gate
added to the three experimental systems projects.

## Primary Source

1. Triton documentation, "Debugging Triton".
   - URL: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
   - Relevance: documents interpreter mode via `TRITON_INTERPRET=1`, CPU
     simulation of Triton kernels, and limitations. Used to scope the
     experimental Phase 0-4 kernel tests as correctness/debugging checks only,
     not GPU performance evidence.

## Local Boundary

The local `./venv_arm64` environment could not install a compatible `triton`
package on this Mac. The experimental tests therefore collect and skip
interpreter execution locally while keeping CPU references passing. A future
machine with an importable Triton package should run:

```bash
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest experimental/*/phase4/tests -rs
```

Do not report throughput, HBM movement, energy, or native latency from these
interpreter tests.
