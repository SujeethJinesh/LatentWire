# Source-Private Masked Consistency Receiver Smoke

- examples: `500`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.366`
- deterministic Hamming matched accuracy: `0.348`
- target-only accuracy: `0.250`
- best learned control: `wrong_projection_source` at `0.254`
- learned minus target: `0.116`
- learned minus best control: `0.112`
- learned minus Hamming: `0.018`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0750 | 0.0005 |
| matched_consistency_packet | 0.366 | 0.348 | 16.00 | 0.3583 | 0.3874 |
| masked_matched_packet | 0.388 | 0.348 | 16.00 | 0.3677 | 0.3859 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0745 | 0.0005 |
| shuffled_source | 0.250 | 0.364 | 16.00 | 0.3574 | 0.3882 |
| answer_masked_source | 0.250 | 0.168 | 16.00 | 0.3544 | 0.3865 |
| random_same_byte | 0.250 | 0.220 | 16.00 | 0.1828 | 0.3869 |
| target_derived_sidecar | 0.250 | 0.362 | 16.00 | 0.2377 | 0.3864 |
| answer_only | 0.250 | 0.304 | 16.00 | 0.1815 | 0.3854 |
| structured_text_matched | 0.250 | 0.362 | 16.00 | 0.1814 | 0.3843 |
| wrong_projection_source | 0.254 | 0.376 | 16.00 | 0.3575 | 0.3870 |
| full_diag_oracle | 1.000 | 0.340 | 16.00 | 0.2377 | 0.3850 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
