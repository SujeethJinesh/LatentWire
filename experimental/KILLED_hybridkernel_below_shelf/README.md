# KILLED: HybridKernel Below Shelf

Date: 2026-05-07

Decision string: `KILL_HYBRIDKERNEL_BELOW_SHELF`

Result packet:

- `experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z/`

Checker:

- Command: `PYTHONPATH="$PWD" ./.venv_gpu/bin/python experimental/hybridkernel/phase2/check_profiler_run_artifacts.py --run-dir experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z --packet-mode no_boundary_signal_kill --require-full-matrix`
- Output: `experimental/hybridkernel/phase2/results/hybridkernel_profiler_gate_20260507T212428Z/artifact_check.json`
- `artifact_check.json` SHA-256: `sha256:a12b41d8412e4b65b7b93c137df1d88f1249a36541e9c791b82aa9b361622451`
- Checker status: `PASS`
- Checker warning: `Nsight Compute artifact is optional in no_boundary_signal_kill mode`

Analysis artifacts:

- `profiler_metrics.json` SHA-256: `sha256:5a629fe9a7f61e84df574b88891c1f3b988231c8792cb5346f0c6487aa44be6e`
- `profiler_analysis_gate.json` SHA-256: `sha256:c1e27fdf98b975ba8ad0f8014325ed3b7bb083ef90db9927a5ae3b9d448ea58c`
- Analysis status: `KILL or shelve: native profiler summaries show less than 1% recoverable gain.`
- Analysis decision: `Do not spend kernel implementation time without a new profiler anomaly.`

Decision metrics:

| Role/config | Rows | Mean recoverable gain upper bound | Bootstrap 95% CI | Primary min gain upper bound |
|---|---:|---:|---:|---:|
| Primary Granite boundary windows | 3 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| Same-family Granite non-boundary controls | 3 | 0.000000 | [0.000000, 0.000000] | 0.000000 |
| Cross-family Nemotron replacement boundary controls | 3 | 0.000000 | [0.000000, 0.000000] | 0.000000 |

Reason classification:

- `below-floor`: yes. Repeated native summaries show less than the 1% kill floor.
- `no-boundary-signal`: yes. Nsight Systems traces did not expose a distinct boundary-local conversion/materialization kernel or layer-boundary NVTX range that could be reduced into a candidate fused operator window.
- `controls-reproduce`: no. Same-family and cross-family controls were below the 3% gate.
- `packet-incomplete`: no. The full-matrix packet passed the checker.
- `substitution-violation`: no. The cross-family replacement was committed pre-profile in `metadata/cross_family_control_replacement_template.json`.
- `prereg-drift`: no known drift. This kill did not modify preregistration files.

Row artifact SHAs:

| Run ID | Role | Model | Committed sanitized Nsight SQLite artifact SHA-256 | Raw local Nsight SQLite export SHA-256 |
|---|---|---|---|---|
| `granite_primary_r1` | `primary_hybrid` | `ibm-granite/granite-4.0-h-tiny` | `sha256:16a4ff21cca1bb7aa213ba00f5aa182e837c9c2fcd0ee220dfe3a93aff6ae09b` | `sha256:3ba969e85822aa2445f3c8fa2740d7f55381a4cfe8179a6b5cf76964201779ab` |
| `granite_primary_r2` | `primary_hybrid` | `ibm-granite/granite-4.0-h-tiny` | `sha256:86bca506bf72bb6a64cf7ad2980625b49f9e01c645817d579c9a95238ea1f8a0` | `sha256:35c0694bc4c8ccc301556f3cef42e4c3f1f5234bcd2e1b3fed58eafafbe147ea` |
| `granite_primary_r3` | `primary_hybrid` | `ibm-granite/granite-4.0-h-tiny` | `sha256:746761bd1f702100ac6908e5ade4a93081ce5ea6fa8f93d634f0bd7c8d30c5b8` | `sha256:e40b4706248f9b22f8f2fd676c8b8eff9d1cf86485ec7814968610b7460b8130` |
| `granite_same_family_r1` | `same_family_control` | `ibm-granite/granite-4.0-h-tiny` | `sha256:adad4d880a7f70e8ddc71ee19aad94111e72545fc9e3d8a323fc7eeb9c517a82` | `sha256:563d5c793773a4e7fdca79ca07b6a55eb4f55a924f2957b31db549472dee4b3a` |
| `granite_same_family_r2` | `same_family_control` | `ibm-granite/granite-4.0-h-tiny` | `sha256:77cb3b8e9cd2052b329e447802f512e7a4d8ba2a82d7a7b70c2c73afacc6c9d0` | `sha256:2ee939209d0609734dccde0ae38dcdbce6d5753ce82032dad3166d311327eb84` |
| `granite_same_family_r3` | `same_family_control` | `ibm-granite/granite-4.0-h-tiny` | `sha256:c49d369044e3c0ca0204640a387c3ae3d6e430492997ddbc8c6c346c697597dd` | `sha256:30d9765caf5d0bc19e463a8fd93a56ba47a1b497c8b035f36b1ae7a0832416c4` |
| `cross_family_r1` | `cross_family_falsification` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `sha256:7c27436b821317090f9205ccf236acc0dab4dd3964470d626ec1271157fed1c2` | `sha256:39161ac4612c9f725ef911ba49be077923f77e2d1eddf33d2de8717d49495a79` |
| `cross_family_r2` | `cross_family_falsification` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `sha256:1ebdfe30832be8bbbdca0b26d10980f2868109fd7eba197247b688620058605c` | `sha256:84243b5ea47827fe4fe2c2d3515a1fbe36bf621148a9a9178b594a1bd061653e` |
| `cross_family_r3` | `cross_family_falsification` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `sha256:17ac91ed01be4c87f0812b5055d584ef3144ec3487c68aa55f465ae45c85f12d` | `sha256:4cdf40e5c2ba6b608dcae0d488cb46538baa0279db86b79094afa3ba7109d11d` |

Effect on portfolio:

- The HybridKernel boundary-fusion profiler gate is killed under the frozen Phase 2 rule.
- The paper draft is preserved but must not receive GPU speedup claims from this packet.
- Per `swarm/goal.md`, this kill is data for positive-method diagnostics and pivots, not a negative-result paper draft.

Storage note:

- The committed packet cites sanitized `.sqlite` source artifacts in `profiler_metrics.json` and `metadata/reduction_input_manifest.json`.
- Raw `.nsys-rep` files are intentionally not committed because Nsight captured environment data that triggered GitHub secret scanning. Raw local `.sqlite` exports are preserved in the workspace run directory and recorded by hash in `metadata/reduction_worksheet.tsv`; the committed sanitized SQLite artifacts preserve session start, request-window NVTX, and aggregate kernel summaries used by `metadata/no_boundary_reduction.py`.
