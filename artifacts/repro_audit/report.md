# C11 Repro Audit Report

## Status

The V1/Nemotron rotation result is traceable from committed score caches and
the V1 delta pack. This audit records the current final-paper tables needed
after the rotation-first queue change.

## Recorded Source Paths

### V1 ParoQuant-on-Nemotron

- Run dir:
  `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z`
- Checker:
  `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z/checker_result.json`
- Score cache:
  `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z/score_cache/`
- V1 pack:
  `artifacts/external_review_pack/v1_rotation_delta_pack_20260528_1506/`

### ParoQuant Granite Reference

- Run dir:
  `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
- Checker:
  `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z/checker_result.json`

### M11b Nemotron Reference

- Run dir:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`
- Metrics:
  `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/metrics.json`

## Key Repro Decision

V1 uses the same BF16/static reference packet as the Nemotron M11b top-10
salvage run. Two no-gap traces are excluded from recovery aggregation and
listed in `current_method_matrix.csv`/the V1 pack.

## Tables Prepared

- `current_method_matrix.csv`
- `rotation_summary.csv`
- `failed_branch_table.csv`
- `decision_tree.md`

## Remaining Repro Gap

DeepSeek and Falcon ParoQuant smoke results have not been measured. Until
those land, the final paper table must mark them as pending rather than
claiming four-model rotation dominance.

