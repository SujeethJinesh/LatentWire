# SA4 Figure Audit - 2026-05-23

Scope: read-only outside `experimental/outlier_migrate/paper/figures/`.
No paper text, result data, result scripts, or generated result artifacts were edited.

## Inventory

| Item | Location | Referenced in paper | Editable in scope | Status |
|---|---|---:|---:|---|
| Layer-stratified migration figure | LaTeX-native `figure` table in `../outlier_migrate_colm2026.tex` | yes, `Figure~\\ref{fig:layer-stratified}` | no | Framework-stable layout; data refresh needed if Nemotron integration updates layer-stratified packets. |
| Draft paper PDF | `../outlier_migrate_colm2026.pdf` | no | no | Compiled paper artifact, not a standalone figure. |

Nearby generated figure-like artifacts not under the paper tree and not referenced by
the paper: `../../phase3/results/om_phase3_20260509T212000Z/grid_sensitivity.svg`
and `../../decomposition_analysis/threshold_sensitivity.pdf`.

## Reference Check

- `fig:layer-stratified` is referenced before definition in the layer-stratified
  migration section.
- `fig:layer-stratified` is also referenced in the reproducibility statement as
  generated from `experimental/outlier_migrate/phase3/results/layer_stratified_migration.json`.
- No `\\includegraphics` commands are present in the paper TeX.

## Styling Audit

- The only referenced figure is a LaTeX table, so matplotlib font sizes, color
  palettes, axis labels, and legend placement do not apply.
- The table has visible column labels and a caption that identifies source data.
- There were no low-DPI paper figure exports, missing plot labels, or debug
  metadata in editable figure assets under this folder.

## Styling Template

`outlier_migrate_matplotlib.mplstyle` defines the default style for future
matplotlib paper figures:

- 300 DPI export target;
- 8 pt base font with smaller tick/legend labels;
- colorblind-safe categorical cycle;
- light gridlines and embedded editable vector text for PDF/SVG exports;
- compact margins suitable for COLM-style single-column figures.

No existing figure source under `figures/` needed restyling because none existed
before this audit.

## Refresh Classification

Needs data refresh during Nemotron integration:

- `fig:layer-stratified`: includes partial Phase 2 Nemotron-3 values and should
  be regenerated if the Nemotron packet is replaced, extended, or reclassified.

Framework-stable:

- The LaTeX-native table layout and caption/source pattern are stable.
- The matplotlib style template is data-independent and ready for future
  exported figures.

## Unresolved Quality Issues

- No editable external figure source exists for the current paper figure, so
  styling can only be standardized for future generated figures unless the paper
  later moves the LaTeX-native table into an external plot.
- Any future paper plot should use the style template here before export.
