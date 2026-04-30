# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.977`
- deterministic Hamming matched accuracy: `0.961`
- target-only accuracy: `0.250`
- best learned control: `wrong_projection_source` at `0.258`
- learned minus target: `0.727`
- learned minus best control: `0.719`
- learned minus Hamming: `0.016`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.4380 | 0.0006 |
| matched_consistency_packet | 0.977 | 0.961 | 6.00 | 26.9269 | 0.4637 |
| masked_matched_packet | 0.922 | 0.961 | 6.00 | 28.8519 | 0.4990 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.4378 | 0.0006 |
| shuffled_source | 0.246 | 0.332 | 6.00 | 27.6193 | 0.4740 |
| answer_masked_source | 0.250 | 0.246 | 6.00 | 26.9233 | 0.4779 |
| random_same_byte | 0.250 | 0.227 | 6.00 | 9.0122 | 0.4716 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 19.9895 | 0.5303 |
| answer_only | 0.250 | 0.246 | 6.00 | 9.2999 | 0.5428 |
| structured_text_matched | 0.250 | 0.227 | 6.00 | 8.4759 | 0.4472 |
| wrong_projection_source | 0.258 | 0.328 | 6.00 | 26.4774 | 0.4987 |
| full_diag_oracle | 1.000 | 1.000 | 6.00 | 19.3845 | 0.4884 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
