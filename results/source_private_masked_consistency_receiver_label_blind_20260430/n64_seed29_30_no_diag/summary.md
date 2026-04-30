# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `1.000`
- deterministic Hamming matched accuracy: `0.172`
- target-only accuracy: `0.250`
- best learned control: `shuffled_source` at `0.312`
- learned minus target: `0.750`
- learned minus best control: `0.688`
- learned minus Hamming: `0.828`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4270 | 0.0006 |
| matched_consistency_packet | 1.000 | 0.172 | 6.00 | 110.4069 | 0.8739 |
| masked_matched_packet | 0.969 | 0.172 | 6.00 | 107.8602 | 0.8141 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4426 | 0.0007 |
| shuffled_source | 0.312 | 0.078 | 6.00 | 108.3297 | 0.4854 |
| answer_masked_source | 0.250 | 0.078 | 6.00 | 106.0749 | 0.4627 |
| random_same_byte | 0.250 | 0.266 | 6.00 | 36.9024 | 0.4976 |
| target_derived_sidecar | 0.250 | 0.219 | 6.00 | 73.1855 | 0.4880 |
| answer_only | 0.250 | 0.375 | 6.00 | 35.4874 | 0.4836 |
| structured_text_matched | 0.250 | 0.188 | 6.00 | 34.4524 | 0.4940 |
| wrong_projection_source | 0.281 | 0.141 | 6.00 | 109.8267 | 0.4998 |
| full_diag_oracle | 1.000 | 0.109 | 6.00 | 78.9349 | 0.4528 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
