# Qwen-to-Phi Target-Error Repair Audit

- audit only: `True`
- eval rows: `768`
- Phi target-only accuracy: `0.263021`
- fixed Qwen-hybrid packet accuracy: `0.467448`
- Qwen source top-2 oracle accuracy: `0.675781`
- fixed-hybrid or Qwen top-2 oracle accuracy: `0.694010`
- fixed-hybrid or Phi top-2 oracle accuracy: `0.727865`
- fixed-hybrid errors with source-unique top-2 repair: `90`
- target-error branch alive: `True`

## Interpretation

The useful next branch is not another unconditioned score receiver. The audit asks whether a tiny source
packet can act like an error-correcting syndrome for the subset of rows where the target-side decision
is wrong and source top-2 still contains the gold candidate. A future method must convert this oracle
headroom into held-out overrides while beating Phi-local top-2 and source-destroying controls.
