# K-RES Failure Audit

Decision: `VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED`.

The current K-RES proxy was compared against the correct tight ParoQuant baseline on the same prompt/window and used the fixed rotated-basis residual sidecar implementation. No implementation mismatch was found that justifies rerunning this exact proxy.

The result is decisive for the current design: top-8x32 MoE residual correction scored recovery -12.294 on Granite tail trace I_4, while tight ParoQuant reference scored 5.940. The method therefore remains killed unless a new bounded or KLLOOK-gated selector changes the corrected columns.

Files in this audit:
- `per_trace.csv`
- `candidate_columns.csv`
- `residual_energy_distribution.csv`
- `correction_magnitude_stats.csv`
- `before_after_losses.csv`
- `implementation_check.md`
- `tail_trace_debug.md`
