# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.969`
- deterministic Hamming matched accuracy: `0.969`
- target-only accuracy: `0.250`
- best learned control: `shuffled_source` at `0.281`
- learned minus target: `0.719`
- learned minus best control: `0.688`
- learned minus Hamming: `0.000`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4486 | 0.0006 |
| matched_consistency_packet | 0.969 | 0.969 | 6.00 | 43.3201 | 1.1873 |
| masked_matched_packet | 0.953 | 0.969 | 6.00 | 39.6591 | 0.4913 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4445 | 0.0006 |
| shuffled_source | 0.281 | 0.250 | 6.00 | 40.9093 | 0.4785 |
| answer_masked_source | 0.250 | 0.281 | 6.00 | 41.3634 | 0.4723 |
| random_same_byte | 0.250 | 0.266 | 6.00 | 14.5283 | 0.4493 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 28.8718 | 0.4468 |
| answer_only | 0.250 | 0.375 | 6.00 | 13.7928 | 0.5022 |
| structured_text_matched | 0.250 | 0.188 | 6.00 | 14.3556 | 0.4387 |
| wrong_projection_source | 0.250 | 0.219 | 6.00 | 43.3121 | 0.4466 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 32.6504 | 0.4736 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
