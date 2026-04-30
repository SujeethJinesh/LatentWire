# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.312`
- deterministic Hamming matched accuracy: `0.188`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.062`
- learned minus best control: `0.062`
- learned minus Hamming: `0.125`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0825 | 0.0005 |
| matched_consistency_packet | 0.312 | 0.188 | 6.00 | 12.2601 | 0.4069 |
| masked_matched_packet | 0.359 | 0.188 | 6.00 | 11.3283 | 0.4028 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0819 | 0.0005 |
| shuffled_source | 0.203 | 0.203 | 6.00 | 12.4612 | 0.4118 |
| answer_masked_source | 0.250 | 0.156 | 6.00 | 14.5778 | 0.4118 |
| random_same_byte | 0.250 | 0.219 | 6.00 | 2.9927 | 0.4149 |
| target_derived_sidecar | 0.250 | 0.281 | 6.00 | 8.5208 | 0.4013 |
| answer_only | 0.250 | 0.312 | 6.00 | 2.8585 | 0.3966 |
| structured_text_matched | 0.250 | 0.281 | 6.00 | 4.6029 | 0.3980 |
| wrong_projection_source | 0.234 | 0.219 | 6.00 | 13.3141 | 0.4101 |
| full_diag_oracle | 1.000 | 0.203 | 6.00 | 9.1829 | 0.4010 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
