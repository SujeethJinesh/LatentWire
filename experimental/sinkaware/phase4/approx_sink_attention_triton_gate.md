# SinkAware Approximate-Attention Triton Gate

Status: **scaffolded; local execution waits on `triton` in `venv_arm64`**.

The new Phase 4 scaffold targets the actual approximate SinkAware operator, not
the killed exact-static branch. It computes exact tail logits, substitutes
predicted fixed-sink logits, and checks that exact predicted sink logits recover
exact scalar attention.

Current local environment:

- `triton` is not importable in `./venv_arm64`;
- tests are written with `pytest.importorskip("triton")`;
- no GPU or speed claim is made.

Next gate:

```bash
TRITON_INTERPRET=1 ./venv_arm64/bin/python -m pytest \
  experimental/sinkaware/phase4/tests/test_approx_sink_attention_triton_interpret.py -rs
```

Promotion requires interpreter correctness against the Phase 3 CPU reference
before any native GPU timing.
