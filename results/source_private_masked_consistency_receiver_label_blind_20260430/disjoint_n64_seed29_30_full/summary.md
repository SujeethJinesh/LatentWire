# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.922`
- deterministic Hamming matched accuracy: `0.922`
- target-only accuracy: `0.250`
- best learned control: `shuffled_source` at `0.266`
- learned minus target: `0.672`
- learned minus best control: `0.656`
- learned minus Hamming: `0.000`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4117 | 0.0006 |
| matched_consistency_packet | 0.922 | 0.922 | 6.00 | 10.2857 | 0.4123 |
| masked_matched_packet | 0.828 | 0.922 | 6.00 | 10.1434 | 0.4245 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4111 | 0.0006 |
| shuffled_source | 0.266 | 0.219 | 6.00 | 9.8919 | 0.4204 |
| answer_masked_source | 0.250 | 0.297 | 6.00 | 9.9627 | 0.4537 |
| random_same_byte | 0.250 | 0.281 | 6.00 | 3.8146 | 0.4291 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 6.9248 | 0.4254 |
| answer_only | 0.250 | 0.312 | 6.00 | 3.5583 | 0.4151 |
| structured_text_matched | 0.250 | 0.188 | 6.00 | 3.8224 | 0.4448 |
| wrong_projection_source | 0.266 | 0.281 | 6.00 | 9.6936 | 0.4520 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 7.5070 | 0.4313 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
