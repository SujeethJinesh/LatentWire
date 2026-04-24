# SVAMP32 Source Margin Audit

- date: `2026-04-24`
- status: `no_source_margin_signal`
- clean IDs scored: `6`
- source/text final clean correct: `0/6`
- source final unknown: `0/6`
- text final unknown: `6/6`
- source-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `1/6`
- mean source margin: `1.343750`
- mean target margin: `-3.619792`
- mean source-minus-target margin: `4.963542`

## Clean Residual Rows

| Example ID | Gold | Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 13cb77b698eeadb5 | 8142 | 46 | 8 | n/a | -5.312500 | -12.984375 | 7.671875 | no_source_margin_advantage |
| 1d50b408c8f5cd2c | 949 | 1 | 246 | n/a | -8.500000 | -11.593750 | 3.093750 | no_source_margin_advantage |
| 2de1549556000830 | 39 | 33 | 3 | n/a | -7.500000 | -9.343750 | 1.843750 | no_source_margin_advantage |
| 6e9745b37ab6fc45 | 61 | 600 | 661 | n/a | 20.312500 | -8.296875 | 28.609375 | source_positive_advantage |
| aee922049c757331 | 1 | 17 | 4 | n/a | 13.437500 | 14.531250 | -1.093750 | no_source_margin_advantage |
| e3ab8666238a289e | 1 | 4 | 4 | n/a | -4.375000 | 5.968750 | -10.343750 | no_source_margin_advantage |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---|
| 4c84ebf42812703b | 10 | 2 | 0.812500 | -5.921875 | 6.734375 | source_positive_advantage |
| 4d780f825bb8541c | 26 | 1 | -1.187500 | -11.015625 | 9.828125 | no_source_margin_advantage |
| de1bf4d142544e5b | 57 | 2 | 2.687500 | -3.328125 | 6.015625 | source_positive_advantage |
