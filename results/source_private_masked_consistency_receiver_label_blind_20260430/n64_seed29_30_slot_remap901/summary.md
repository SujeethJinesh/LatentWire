# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.375`
- deterministic Hamming matched accuracy: `0.344`
- target-only accuracy: `0.250`
- best learned control: `wrong_projection_source` at `0.297`
- learned minus target: `0.125`
- learned minus best control: `0.078`
- learned minus Hamming: `0.031`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1023 | 0.0006 |
| matched_consistency_packet | 0.375 | 0.344 | 6.00 | 109.7013 | 1.1397 |
| masked_matched_packet | 0.406 | 0.344 | 6.00 | 112.6875 | 0.4893 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.1031 | 0.0006 |
| shuffled_source | 0.203 | 0.375 | 6.00 | 105.3781 | 0.5063 |
| answer_masked_source | 0.250 | 0.297 | 6.00 | 115.2330 | 0.5257 |
| random_same_byte | 0.250 | 0.297 | 6.00 | 37.7372 | 0.4822 |
| target_derived_sidecar | 0.250 | 0.375 | 6.00 | 69.5885 | 1.1005 |
| answer_only | 0.250 | 0.438 | 6.00 | 35.0026 | 0.4950 |
| structured_text_matched | 0.250 | 0.156 | 6.00 | 36.9825 | 1.0827 |
| wrong_projection_source | 0.297 | 0.312 | 6.00 | 105.4792 | 0.4736 |
| full_diag_oracle | 1.000 | 0.359 | 6.00 | 76.0255 | 0.4854 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
