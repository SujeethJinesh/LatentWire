# Phase 3 Position Grid Sensitivity

Run dir: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`

| Grid | Positions | Median recovery | 95% CI |
|---|---|---:|---:|
| sparse | [100, 5000, 10000] | 0.000000000000 | [0.000000000000, 0.752281234025] |
| primary | [100, 1000, 5000, 10000] | 0.000000000000 | [0.000000000000, 0.711143244199] |
| dense | [100, 500, 1000, 2000, 5000, 7500, 10000] | 0.000000000000 | [0.000000000000, 0.920007799726] |

CSV: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/grid_sensitivity.csv`
SVG plot: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/grid_sensitivity.svg`

This is a descriptive robustness check, not a separate pass/kill gate.
