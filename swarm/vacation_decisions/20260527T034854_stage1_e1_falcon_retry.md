# Stage 1 E1 Falcon-H1 Retry Decision

E1 first attempt completed DeepSeek-R1-Distill-Qwen-1.5B and Falcon-H1 BF16
reference generation, then failed during Falcon-H1 `static_1pct` quantized KL
collection.

Failure signature:

`RuntimeError: Expected conv_state.scalar_type() == input_type to be true`

Diagnosis: Falcon-H1 uses dict-backed hybrid Mamba cache states. The shared
cache helper aligned list-backed caches to the FP16 autocast dtype, but did not
align dict-backed `conv_states` / `ssm_states`. Quantized KL runs under FP16
autocast, while Falcon-H1 initialized its cache in model dtype (`bfloat16`),
triggering a causal-conv1d update mismatch.

Decision: patch the shared cache helper to align dict-backed cache states, then
rerun E1 with `--resume`. This is a first infrastructure failure, not a
scientific KILL. It does not trigger the two-failure pause condition.

Verification before rerun:

- `python -m py_compile experimental/outlier_migrate/phase4/run_om_phase4_intervention.py experimental/outlier_migrate/phase9/run_om_stage1_e1_cross_model_kl_fft.py`
- `PYTHONPATH=. pytest -q experimental/outlier_migrate/phase4/tests/test_phase4_intervention_gate.py experimental/outlier_migrate/phase9/tests/test_m11b_budget_scaling.py`

Patch commit: `ff424658`.

Follow-up: a read-only diagnostic confirmed that dict cache alignment alone was
not enough because Falcon-H1's fused fast path still rejects the BF16/FP16
mixed state. Added a bounded fast-path disable around autocast-sensitive
quantized scoring/KL collection and added E1 resume reuse for existing Falcon
BF16/activation artifacts. This avoids repeating the completed 12-trace,
20K-token Falcon reference pass.

Follow-up patch commit: `2e78bfbe`.
