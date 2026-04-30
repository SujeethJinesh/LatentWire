# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.234`
- deterministic Hamming matched accuracy: `0.180`
- target-only accuracy: `0.250`
- best learned control: `shuffled_source` at `0.254`
- learned minus target: `-0.016`
- learned minus best control: `-0.020`
- learned minus Hamming: `0.055`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.1036 | 0.0006 |
| matched_consistency_packet | 0.234 | 0.180 | 6.00 | 22.5312 | 0.5320 |
| masked_matched_packet | 0.227 | 0.180 | 6.00 | 20.8592 | 0.4799 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.1036 | 0.0006 |
| shuffled_source | 0.254 | 0.219 | 6.00 | 21.7659 | 0.5178 |
| answer_masked_source | 0.250 | 0.242 | 6.00 | 22.2818 | 0.4906 |
| random_same_byte | 0.250 | 0.289 | 6.00 | 6.8603 | 0.4725 |
| target_derived_sidecar | 0.250 | 0.219 | 6.00 | 13.9509 | 0.5177 |
| answer_only | 0.250 | 0.293 | 6.00 | 6.9030 | 0.4731 |
| structured_text_matched | 0.250 | 0.258 | 6.00 | 6.5707 | 0.4782 |
| wrong_projection_source | 0.234 | 0.199 | 6.00 | 22.0144 | 0.4936 |
| full_diag_oracle | 1.000 | 0.285 | 6.00 | 15.0385 | 0.4930 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
