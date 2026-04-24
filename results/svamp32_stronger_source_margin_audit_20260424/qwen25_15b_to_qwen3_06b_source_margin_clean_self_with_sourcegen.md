# SVAMP32 Source Margin Audit

- date: `2026-04-24`
- status: `no_source_margin_signal`
- clean IDs scored: `6`
- source/text final clean correct: `0/6`
- source final unknown: `0/6`
- text final unknown: `6/6`
- source-margin positive clean IDs: `3/6`
- source-margin positive+advantage clean IDs: `1/6`
- mean source margin: `-1.455791`
- mean target margin: `-3.624139`
- mean source-minus-target margin: `2.168349`

## Clean Residual Rows

| Example ID | Gold | Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 13cb77b698eeadb5 | 8142 | 46 | 7902 | n/a | -7.598871 | -13.017838 | 5.418966 | no_source_margin_advantage |
| 1d50b408c8f5cd2c | 949 | 1 | 246 | n/a | -4.470896 | -11.604153 | 7.133257 | no_source_margin_advantage |
| 2de1549556000830 | 39 | 33 | 3 | n/a | -8.765114 | -9.340767 | 0.575653 | no_source_margin_advantage |
| 6e9745b37ab6fc45 | 61 | 600 | 661 | n/a | 2.278078 | -8.293890 | 10.571968 | source_positive_advantage |
| aee922049c757331 | 1 | 17 | 4 | n/a | 7.649824 | 14.522760 | -6.872936 | no_source_margin_advantage |
| e3ab8666238a289e | 1 | 4 | 3 | n/a | 2.172234 | 5.989050 | -3.816816 | no_source_margin_advantage |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---|
| 4c84ebf42812703b | 10 | 2 | -5.021349 | -5.933392 | 0.912043 | no_source_margin_advantage |
| 4d780f825bb8541c | 26 | 1 | -3.889095 | -11.010489 | 7.121393 | no_source_margin_advantage |
| de1bf4d142544e5b | 57 | 2 | -1.329193 | -3.357849 | 2.028656 | no_source_margin_advantage |
