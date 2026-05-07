# HybridKernel Native Run Packet

Status: skeleton only; not native profiler evidence.

Fill this directory on the NVIDIA host and return the whole directory. Do not
return screenshots, notebooks, or client-only traces.

Required final command:

```bash
python "$HWK_ROOT/phase2/check_profiler_run_artifacts.py" \
  --run-dir "$HWK_RUN" \
  --require-full-matrix \
  | tee "$HWK_RUN/artifact_check.json"
```

The checker must pass before any HybridKernel result is interpreted. A pass
only means the packet is complete enough for review; it is not a speed claim.

Initial model target: `ibm-granite/granite-4.0-h-tiny`.
