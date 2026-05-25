# Headline Number Re-Derivation

Date: 2026-05-25

Scope: headline values in the paper, independently re-read from packet JSON.

## Side-By-Side Checks

| Number | Packet value | Paper/release value | Status |
|---|---:|---:|---|
| M11b Granite top-5 median | 0.4492840911245966 | 0.449284091125 | Match by rounding |
| M11b Granite top-5 CI low | -1.3009000187907436 | -1.300900018791 | Match by rounding |
| M11b Granite top-5 CI high | 1.00079062654749 | 1.000790626547 | Match by rounding |
| M11b Granite top-5 vs static-top10 margin | 0.5010512676801815 | 0.501051267680 | Match by derivation |
| M11b Nemotron top-5 median | 0.4567361832703779 | 0.456736183270 | Match by rounding |
| M11b Nemotron top-10 median | 0.8147397989034302 | 0.814739798903 | Match by rounding |
| M11b Nemotron top-10 vs static-top10 margin | 0.2203560552855075 | 0.220356055286 | Match by derivation |
| ParoQuant Granite median | 0.753776848891178 | 0.753776848891 | Match by rounding |
| M26 stable-core median | 0.17761580764776214 | 0.177615807648 | Match by rounding |
| KL static mean | 0.15030275027584675 | 0.150302750276 | Match by rounding |
| KL DecDEC mean | 0.13270188444570422 | 0.132701884446 | Match by rounding |
| KL M11 mean | 0.13140658538252428 | 0.131406585383 | Match by rounding |

## Command Evidence

Values above were extracted from:

- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/{metrics,bootstrap_ci}.json`
- `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/{metrics,bootstrap_ci}.json`
- `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z/metrics.json`
- `experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z/metrics.json`
- `experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z/kl_summary.json`

No headline-number discrepancy was found in this pass.
