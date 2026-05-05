# SVAMP32 C2C Generation Trace Syndrome Probe Manifest

- date: `2026-05-05`
- status: `c2c_mechanism_syndrome_probe_fails_gate`
- pass gate: `false`

| Artifact | SHA256 |
|---|---|
| `generation_trace_probe.json` | `68e03029aa09c38fc0bbc836a5a0c6b280a210bbe19547229cc22afe8f64a454` |
| `generation_trace_probe.md` | `9e242d7966e5b75b4d632a4c778ca27d8ddba73fc993100947a5db6c2e51d01c` |

## Summary

The current Transformers cache compatibility shim unblocks CPU execution of
the vendored C2C generation path, but the generation-time trace syndrome probe
does not recover clean C2C-residual rows:

- matched: `12/32`;
- zero-source: `14/32`;
- target-only: `14/32`;
- clean source-necessary IDs: `0`;
- feature matrix: `32 x 10080`, `float32`.

This weakens the current generation-summary trace branch as a deployable
sparse-packet path. A separate local smoke also shows the current shimmed CPU
C2C generation degenerates into repeated Korean glyph tokens on the first four
SVAMP rows, so current Mac CPU C2C traces should be treated as runtime
diagnostics rather than native C2C teacher evidence.
