# SinkAware Approximate-Attention Triton Gate

Status: **local interpreter correctness passes in `venv_arm64`**.

The new Phase 4 scaffold targets the actual approximate SinkAware operator, not
the killed exact-static branch. It computes exact tail logits, substitutes
predicted fixed-sink logits, and checks that exact predicted sink logits recover
exact scalar attention.

Triton's debugging guide describes interpreter mode as a CPU simulation path
enabled by `TRITON_INTERPRET=1`, with compilation bypassed for debugging:
<https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>.

Current local readiness check:

- `triton==3.7.0+git270e696d` is importable from the local `triton-cpu`
  source build;
- `TRITON_INTERPRET=1` was set for the readiness command;
- `torch.cuda.is_available()` is false on this Mac;
- execution tests pass in interpreter mode;
- no CUDA, GPU, or speed claim is made.

2026-05-06 source install: after checking the official Triton install docs and
`triton-lang/triton-cpu`, source installation succeeded from `triton-cpu`
revision `270e696` after initializing the `third_party/sleef` submodule. The
shared install note is `experimental/triton_cpu_source_install_20260506.md`.

Next gate:

```bash
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest \
  experimental/sinkaware/phase4/tests/test_approx_sink_attention_triton_interpret.py -rs
```

Promotion requires interpreter correctness against the Phase 3 CPU reference
before any native GPU timing. This gate is now cleared locally; promotion still
requires downstream quality controls and native timing evidence.
