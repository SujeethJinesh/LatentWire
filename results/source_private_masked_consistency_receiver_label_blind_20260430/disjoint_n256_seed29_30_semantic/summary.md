# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.996`
- deterministic Hamming matched accuracy: `0.156`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.746`
- learned minus best control: `0.746`
- learned minus Hamming: `0.840`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.3656 | 0.0006 |
| matched_consistency_packet | 0.996 | 0.156 | 6.00 | 7.0945 | 0.4176 |
| masked_matched_packet | 0.973 | 0.156 | 6.00 | 7.3389 | 0.4218 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.3694 | 0.0006 |
| shuffled_source | 0.250 | 0.223 | 6.00 | 6.9360 | 0.4161 |
| answer_masked_source | 0.250 | 0.184 | 6.00 | 7.4759 | 0.4169 |
| random_same_byte | 0.250 | 0.273 | 6.00 | 3.0335 | 0.4226 |
| target_derived_sidecar | 0.250 | 0.094 | 6.00 | 5.8110 | 0.4239 |
| answer_only | 0.250 | 0.277 | 6.00 | 3.1416 | 0.4337 |
| structured_text_matched | 0.250 | 0.242 | 6.00 | 3.0570 | 0.4153 |
| wrong_projection_source | 0.250 | 0.250 | 6.00 | 7.1660 | 0.4272 |
| full_diag_oracle | 1.000 | 0.129 | 6.00 | 5.2148 | 0.4150 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
