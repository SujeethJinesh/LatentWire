# Source-Private Product-Codebook Target-Decoder Smoke

- examples: `8`
- pass gate: `False`
- strict CI pass gate: `False`
- matched minus target: `0.250`
- matched minus best control: `0.000`
- best control condition: `label_shuffled_ridge`
- matched vs target CI95: `[-0.250, 0.625]`
- matched vs best control CI95: `[0.000, 0.000]`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 2/8 | 0.250 | 1.000 | 0.00 | 0.00 | 0.02 |
| zero_source | 2/8 | 0.250 | 1.000 | 0.00 | 0.00 | 0.01 |
| matched_product_codebook | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2471.16 |
| label_shuffled_ridge | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2414.13 |
| constrained_shuffled_source | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2384.05 |
| answer_masked_source | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2332.01 |
| permuted_codes | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2314.92 |
| wrong_codebook_packet | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2335.44 |
| random_same_byte | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2341.88 |
| structured_json_same_byte | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2276.74 |
| structured_free_text_same_byte | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2335.64 |
| target_derived_sidecar | 4/8 | 0.500 | 1.000 | 4.00 | 4.00 | 2360.73 |

Pass rule: matched product-codebook target decoder must beat target-only by >=0.15, every source-destroying or same-byte text/target-derived control must stay within target+0.05, exact-ID parity must hold, and every condition must keep valid prediction rate >=0.95. strict_ci_pass_gate additionally requires paired bootstrap CI95 lower bounds >=+0.10 for matched-vs-target and matched-vs-best-control.
