# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.914`
- deterministic Hamming matched accuracy: `0.930`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.664`
- learned minus best control: `0.664`
- learned minus Hamming: `-0.016`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4261 | 0.0006 |
| matched_consistency_packet | 0.914 | 0.930 | 6.00 | 17.8151 | 0.4615 |
| masked_matched_packet | 0.871 | 0.930 | 6.00 | 16.6780 | 0.4455 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4305 | 0.0006 |
| shuffled_source | 0.250 | 0.285 | 6.00 | 17.4035 | 0.4422 |
| answer_masked_source | 0.250 | 0.242 | 6.00 | 15.7539 | 0.4409 |
| random_same_byte | 0.250 | 0.273 | 6.00 | 5.2065 | 0.4651 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 11.9768 | 0.4349 |
| answer_only | 0.250 | 0.277 | 6.00 | 4.9684 | 0.4449 |
| structured_text_matched | 0.250 | 0.242 | 6.00 | 5.2413 | 0.4626 |
| wrong_projection_source | 0.250 | 0.316 | 6.00 | 16.9930 | 0.4375 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 11.7760 | 0.4425 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
