# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.281`
- deterministic Hamming matched accuracy: `0.250`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.031`
- learned minus best control: `0.031`
- learned minus Hamming: `0.031`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1095 | 0.0006 |
| matched_consistency_packet | 0.281 | 0.250 | 6.00 | 20.3239 | 0.4915 |
| masked_matched_packet | 0.234 | 0.250 | 6.00 | 20.6242 | 0.4838 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.1024 | 0.0006 |
| shuffled_source | 0.250 | 0.406 | 6.00 | 22.8388 | 0.4896 |
| answer_masked_source | 0.250 | 0.359 | 6.00 | 20.9177 | 0.6979 |
| random_same_byte | 0.250 | 0.281 | 6.00 | 6.8984 | 0.4458 |
| target_derived_sidecar | 0.250 | 0.297 | 6.00 | 14.8676 | 0.4837 |
| answer_only | 0.250 | 0.375 | 6.00 | 7.3198 | 0.5428 |
| structured_text_matched | 0.250 | 0.188 | 6.00 | 6.1002 | 0.4654 |
| wrong_projection_source | 0.188 | 0.375 | 6.00 | 21.1717 | 0.5737 |
| full_diag_oracle | 1.000 | 0.484 | 6.00 | 14.7874 | 1.0498 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
