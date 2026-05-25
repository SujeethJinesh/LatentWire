# Audit Summary

Date: 2026-05-25

The requested subagent audits were completed locally because all audit
subagents failed with usage-limit errors. Findings are severity categorized.

| Audit | Findings | Severity | Status |
|---|---|---|---|
| Code correctness | Full mode returned frozen claims instead of real GPU execution. | CRITICAL | Fixed. Full mode now raises clearly. |
| Code correctness | Release helpers are minimal and not the archived GPU packet pipeline. | SUBSTANTIAL | Documented. |
| Hardcoding/determinism | Frozen claim values are hardcoded for FAST_VERIFY. | SUBSTANTIAL | Intentional and documented. |
| Error handling | No silent broad exception handling found; full mode now fails loudly. | MINOR | Verified. |
| Reproducibility | Clean-clone FAST_VERIFY passed twice; full GPU reproduction not implemented. | SUBSTANTIAL | Documented; not overclaimed. |
| Adversarial review | M11b must remain hedged due Granite CI and Nemotron budget shift. | SUBSTANTIAL | Paper already hedges. |
| Statistical rigor | Positive-gap sample sizes are small. | SUBSTANTIAL | Paper reports included traces and no-gap fractions. |
| Figure/table audit | No KL/FFT plots; evidence is table/text heavy. | SUBSTANTIAL | Acceptable for first-complete draft; polish opportunity. |
| Headline numbers | Packet values match paper/release by rounding. | MINOR | Verified. |
| Cross-document consistency | External collaboration state and final reports are stale/missing. | SUBSTANTIAL | Pending final documentation. |

## Critical Findings

All CRITICAL findings from this audit pass are addressed. The main fix was to
remove the false full-mode pass from release scripts and documentation.

## Substantial Deferred Items

- Full model-inference reproduction is not implemented inside `release/`.
  Because FAST_VERIFY was the verified clean-clone gate and the package now
  discloses the limitation, this is deferred rather than hidden.
- Final documentation still needs to be updated before declaring the sprint
  complete.
