# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.957`
- deterministic Hamming matched accuracy: `0.988`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.707`
- learned minus best control: `0.707`
- learned minus Hamming: `-0.031`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4303 | 0.0006 |
| matched_consistency_packet | 0.957 | 0.988 | 6.00 | 16.6538 | 0.4516 |
| masked_matched_packet | 0.918 | 0.988 | 6.00 | 15.3513 | 0.4330 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4268 | 0.0006 |
| shuffled_source | 0.250 | 0.344 | 6.00 | 15.4969 | 0.4670 |
| answer_masked_source | 0.250 | 0.219 | 6.00 | 16.2702 | 0.4615 |
| random_same_byte | 0.250 | 0.277 | 6.00 | 5.3316 | 0.4603 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 12.3635 | 0.4623 |
| answer_only | 0.250 | 0.289 | 6.00 | 5.3946 | 0.4438 |
| structured_text_matched | 0.250 | 0.223 | 6.00 | 5.1412 | 0.4408 |
| wrong_projection_source | 0.250 | 0.273 | 6.00 | 16.4223 | 0.4556 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 12.0504 | 0.4354 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
