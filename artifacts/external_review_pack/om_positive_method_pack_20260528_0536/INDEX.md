# OutlierMigrate Positive-Method External Review Pack

Recommended read order:

1. `EXECUTIVE_SUMMARY.md`
2. `tables/experiment_summary.csv`
3. `QUESTIONS_FOR_REVIEWER.md`
4. `experiments/18_wjac_prefilter/summary.md`
5. `experiments/19_lambda_prefilter/summary.md`
6. `experiments/20_hyst_prefilter/summary.md`
7. `IDEATION_MAP.md`

Top-level folders:

- `experiments/`: one folder per branch with summary, metrics, per-trace data, decisions, logs, and compact source files.
- `tables/`: cross-experiment CSVs.
- `plots/`: compact visual summaries generated from tables.
- `code_snippets/`: minimal standalone method/metric sketches.
- `scripts/`: GPU-free helpers for inspecting this pack.
- `provenance/`: git/env/model/artifact provenance.

Large activation tensors are intentionally omitted; see `OMITTED_LARGE_ARTIFACTS.md`.
