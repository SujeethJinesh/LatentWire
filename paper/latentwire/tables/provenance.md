# LatentWire Provenance Table

| Alias | Numeric claims supported | n / split | Caveat | Source artifact |
| --- | --- | --- | --- | --- |
| LW-HO source-index null | WZ delta `-0.023622`, CI `[-0.060367, 0.013123]`; target `0.194226`; source-index+confidence `0.204724`; WZ `0.181102`; upper bound `0.314961` | `381` held-out aggregate rows | Aggregate-only paper use; raw held-out rows are not packaged | safe aggregate dashboard `dashboard/latentwire_terminal_negative_sanitized.md` |
| LW score ladder | score signal `2.098527` bits to `0.281933` bits; gate delta `-0.013928`, CI `[-0.050139, 0.022284]` | `1500` scored dev/gate rows; `359` gate rows | Dev/gate only; not a deployable positive | `results/mac_continue/latentwire_oracle_ladder/summary.json` |
| L-Q1 query packet | delta `-0.022284`, CI `[-0.055710, 0.011142]`; source-copy MI `0.457419` bits | `359` gate rows | Controls did not collapse; ablations explain signal | `results/mac_continue/latentwire_query_packet/summary.json` |
| Exact evidence | visible exact `1.000`, matched packet `0.775`, delta `+0.225`, CI `[0.192188, 0.257812]` | `640` model-helper rows; `160` unique examples | Unique-example floor not met; reported as operational support, not universal proof | `results/escape_tests/20260605_cpu_cached/summary.json` |
| Privacy proxy | packet utility `0.875`, family leakage `1.000`; no matched-utility text comparator | `512` rows per condition | Cached proxy only | `results/escape_tests/20260605_cpu_cached/summary.json` |
| L-IB1 | current utility/leakage `0.875/1.000`; best bottleneck `0.602339/0.602339`; dev/gate `341/171` | `512` matched rows | CPU cached feature-selector gate; no neural IB encoder | `results/escape_tests/L_IB1_privacy_bottleneck/summary.json` |
| Oracle audit | L-PC5 oracle `+0.666` and L-C2 oracle `+0.832` rejected as deployable claims | strict cached screens | Gold-aware or setup-blocked; not promoted | `dashboard/l_pc5_deployable_verifier_plan.md`, `dashboard/l_c2_oracle_decomposition.md` |
