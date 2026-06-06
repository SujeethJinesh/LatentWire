# Channel-Set Provenance Table

| Alias | Numeric claims supported | n / split | Caveat | Source artifact |
| --- | --- | --- | --- | --- |
| CS top-1 drift | strict set-leaving: Granite-Small `0.566235`, Nemotron-3 `0.533713`, DeepSeek-R1-Distill `0.670573`, Falcon-H1 `0.673611` | cached decomposition packets | Measurement/regime evidence only | `experimental/outlier_migrate/phase9/step9_0_decomposition_replication.md`, `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z/migration_decomposition.md` |
| CS threshold sweep | strict set-leaving remains high at 0.5%, 1%, 2%, 5% thresholds for Granite-Small, Nemotron-3, and DeepSeek-R1-Distill | cached post-hoc sensitivity sweep | Falcon-H1 has top-1 decomposition but not the same threshold-sweep artifact | `paper/channel_set/figures/threshold_sensitivity.png` |
| CS within-set shuffle | top-1 within-set shuffling: Granite-Small `0.270935`, Nemotron-3 `0.269082`, DeepSeek-R1-Distill `0.165737`, Falcon-H1 `0.097538` | cached decomposition packets | Secondary to strict set-leaving | `paper/channel_set/figures/within_set_shuffling.png` |
| C-A1 cached screen | C-A1 gate rows `6`; positive medians `5`; nonpositive `1`; min/max `-1.1159`/`1.5666` | cached Stage-1 screen | Not native W4A16; not a claim | `results/overnight/20260603_cpu_only_screening/exp5/summary.json` |
| C-A1 manifest blocker | same-row counts: Granite `0`, DeepSeek `12`, Falcon `12` | cheap cache inventory | Granite member missing/invalid; C-A1 parked | `results/cheap_exhaustion/20260605T180448Z/summary.json` |
| C-U1 blocker | `500` KL rows, but missing paired difficulty/policy-uplift labels | strict Mac cached screen | schema-blocked | `results/mps_first_strict_20260605/C_U1_drift_as_signal_router/summary.json` |
| C-S1 blocker | `198` eligible rows vs `500` floor | strict Mac cached screen | floor not lowered | `results/mps_first_strict_20260605/C_S1_clean_survival_stablecore_denominator/summary.json` |
| C-Y5 blocker | `6` audit checks; native three-model pairing missing | strict Mac cached screen | parked until native pairing | `results/mps_first_strict_20260605/C_Y5_channel_set_defense_bundle/summary.json` |
