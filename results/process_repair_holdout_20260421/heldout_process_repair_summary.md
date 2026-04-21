# Held-Out Process Repair Summary

Run date: 2026-04-21

Frozen manifest:

- `process_repair_holdout_manifest.json`
- `run_process_repair_holdout.sh`

## Raw Route Pools

| Split | Salt | Target candidate acc. | Selected route acc. | Delta | Method-only | Baseline-only | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM70 | 0 | 0.0571 | 0.0857 | +0.0286 | 4 | 2 | 2 | 62 |
| GSM70 | 1 | 0.0571 | 0.0286 | -0.0286 | 2 | 4 | 0 | 64 |
| GSM70 | 2 | 0.0571 | 0.0571 | +0.0000 | 2 | 2 | 2 | 64 |
| SVAMP70 | 0 | 0.3000 | 0.3000 | +0.0000 | 10 | 10 | 11 | 39 |
| SVAMP70 | 1 | 0.3000 | 0.3000 | +0.0000 | 8 | 8 | 13 | 41 |
| SVAMP70 | 2 | 0.3000 | 0.2571 | -0.0429 | 6 | 9 | 12 | 43 |

Interpretation: raw stochastic route pools are candidate generators, not a
standalone method claim. The held-out gain comes from process repair.

## Process Repair

| Split | Method | Accuracy | Delta vs target | Pre-repair acc. | Method-only | Baseline-only | Both correct | Both wrong | Changed answer | Repair help | Repair harm | Target selected | Full oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM70 | process_repair_selected_route | 0.2000 | +0.1429 | 0.1286 | 10 | 0 | 4 | 56 | 0.4000 | 0.0714 | 0.0000 | 0.6714 | 0.1571 |
| SVAMP70 | process_repair_selected_route | 0.5429 | +0.2429 | 0.3571 | 17 | 0 | 21 | 32 | 0.4000 | 0.1857 | 0.0000 | 0.8429 | 0.5286 |

Interpretation: strict selector plus process repair is now a held-out positive
method candidate across GSM70 and SVAMP70. It should be reported with explicit
target-side repair compute, token, byte, and latency accounting before making an
efficiency claim against text-to-text or direct competitors such as `C2C`.
