# SVAMP32 Source Margin Audit

- date: `2026-04-24`
- status: `no_source_margin_signal`
- clean IDs scored: `6`
- source/text final clean correct: `0/6`
- source-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `0/6`
- mean source margin: `-3.065174`
- mean target margin: `-3.624139`
- mean source-minus-target margin: `0.558965`

## Clean Residual Rows

| Example ID | Gold | Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 13cb77b698eeadb5 | 8142 | 46 | n/a | 4 | -13.759378 | -13.017838 | -0.741541 | no_source_margin_advantage |
| 1d50b408c8f5cd2c | 949 | 1 | 25 | 25 | -8.573317 | -11.604153 | 3.030836 | no_source_margin_advantage |
| 2de1549556000830 | 39 | 33 | 33 | 33 | -2.990372 | -9.340767 | 6.350395 | no_source_margin_advantage |
| 6e9745b37ab6fc45 | 61 | 600 | 3 | 661 | -1.279598 | -8.293890 | 7.014292 | no_source_margin_advantage |
| aee922049c757331 | 1 | 17 | 5 | 3 | 5.874361 | 14.522760 | -8.648399 | no_source_margin_advantage |
| e3ab8666238a289e | 1 | 4 | 5 | 5 | 2.337257 | 5.989050 | -3.651793 | no_source_margin_advantage |

## Target-Self-Repair Rows

| Example ID | Gold | Distractor | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---|
| 4c84ebf42812703b | 10 | 2 | -3.328716 | -5.933392 | 2.604675 | no_source_margin_advantage |
| 4d780f825bb8541c | 26 | 1 | -3.886646 | -11.010489 | 7.123842 | no_source_margin_advantage |
| de1bf4d142544e5b | 57 | 2 | -0.698309 | -3.357849 | 2.659540 | no_source_margin_advantage |
