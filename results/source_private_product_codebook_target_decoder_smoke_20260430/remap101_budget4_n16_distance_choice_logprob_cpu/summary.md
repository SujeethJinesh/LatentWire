# Source-Private Product-Codebook Target-Decoder Smoke

- examples: `16`
- pass gate: `False`
- strict CI pass gate: `False`
- matched minus target: `0.000`
- matched minus best control: `0.000`
- best control condition: `zero_source`
- matched vs target CI95: `[0.000, 0.000]`
- matched vs best control CI95: `[0.000, 0.000]`

| Condition | Correct | Accuracy | Valid prediction | Mean bytes | Mean generated tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 5/16 | 0.312 | 1.000 | 0.00 | 1.00 | 496.89 |
| zero_source | 5/16 | 0.312 | 1.000 | 0.00 | 1.00 | 503.55 |
| matched_product_codebook | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 637.94 |
| label_shuffled_ridge | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 631.10 |
| constrained_shuffled_source | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 604.14 |
| answer_masked_source | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 597.10 |
| permuted_codes | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 591.45 |
| wrong_codebook_packet | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 575.21 |
| random_same_byte | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 578.24 |
| structured_json_same_byte | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 605.86 |
| structured_free_text_same_byte | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 649.46 |
| target_derived_sidecar | 5/16 | 0.312 | 1.000 | 4.00 | 1.00 | 607.91 |

Pass rule: matched product-codebook target decoder must beat target-only by >=0.15, every source-destroying or same-byte text/target-derived control must stay within target+0.05, exact-ID parity must hold, and every condition must keep valid prediction rate >=0.95. strict_ci_pass_gate additionally requires paired bootstrap CI95 lower bounds >=+0.10 for matched-vs-target and matched-vs-best-control.
