# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.957`
- deterministic Hamming matched accuracy: `0.977`
- target-only accuracy: `0.250`
- best learned control: `wrong_projection_source` at `0.281`
- learned minus target: `0.707`
- learned minus best control: `0.676`
- learned minus Hamming: `-0.020`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4352 | 0.0006 |
| matched_consistency_packet | 0.957 | 0.977 | 6.00 | 26.9366 | 0.4754 |
| masked_matched_packet | 0.922 | 0.977 | 6.00 | 28.1278 | 0.5258 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4406 | 0.0006 |
| shuffled_source | 0.250 | 0.340 | 6.00 | 26.6620 | 0.4969 |
| answer_masked_source | 0.250 | 0.203 | 6.00 | 25.8996 | 0.5115 |
| random_same_byte | 0.250 | 0.273 | 6.00 | 9.0582 | 0.4927 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 19.9631 | 0.5699 |
| answer_only | 0.250 | 0.289 | 6.00 | 10.3046 | 0.5213 |
| structured_text_matched | 0.250 | 0.223 | 6.00 | 8.6644 | 0.4914 |
| wrong_projection_source | 0.281 | 0.242 | 6.00 | 27.1590 | 0.4559 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 20.0299 | 0.4703 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
