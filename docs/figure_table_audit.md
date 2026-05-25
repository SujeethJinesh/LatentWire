# Figure And Table Accuracy Audit

Date: 2026-05-25

Scope: every figure/table in `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`.

## External Figures

No `\includegraphics` commands are present. The only figure environment is a
LaTeX-native table labeled `fig:layer-stratified`. The paper tree contains a
style file and figure audit inventory but no exported plot assets.

## Required Visual/Tabular Claims

| Required item | Present? | Location / note |
|---|---:|---|
| Four-model set-leaving comparison | Yes | Cross-architecture decomposition table. |
| Method results comparison table | Yes | Intervention attempts table. |
| Three-mechanism framework diagram | No | Covered textually; absence is a presentation weakness, not a numerical error. |
| KL trajectory plot | No | KL results are tabular/textual. |
| FFT spectral characterization | Textual | Release reproduces entropy/autocorrelation; no plot. |
| Per-component dissection | Yes | `fig:layer-stratified`. |
| Trace heterogeneity / no-gap rate | Yes | Text and method tables. |
| Structural ceiling pattern | Textual | Not a separate figure. |
| ParoQuant scope contextualization | Yes | Intervention table and mechanism text. |
| M11b cross-model comparison | Yes | Intervention table and budget subsection. |

## Number Checks

Spot checks against packet files and paper text:

- M11b Granite top-5 `0.449284091125` matches packet rounded from
  `0.4492840911245966`.
- M11b Nemotron top-5 `0.456736183270` and top-10 `0.814739798903` match the
  Path C salvage packet.
- ParoQuant `0.753776848891` matches the algorithmic baseline packet within
  rounding.
- M26 `0.177615807648` matches the stable-core packet.
- KL means and AR decay values are described consistently with the KL packet.
- Per-component rows are internally consistent with the paper's scoped
  block-output claim.

## Findings

| Severity | Finding | Status |
|---|---|---|
| SUBSTANTIAL | The paper relies heavily on tables and has no KL/FFT plots. | Acceptable for first-complete draft; could be improved for submission polish. |
| MINOR | `fig:layer-stratified` is a table inside a figure environment. | Acceptable LaTeX-native presentation. |
