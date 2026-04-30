# Source-Private Masked Consistency Receiver Smoke

- examples: `64`
- pass gate: `True`
- source packet pass: `True`
- learned matched accuracy: `0.922`
- deterministic Hamming matched accuracy: `0.422`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.672`
- learned minus best control: `0.672`
- learned minus Hamming: `0.500`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.3891 | 0.0006 |
| matched_consistency_packet | 0.922 | 0.422 | 6.00 | 101.9864 | 0.4939 |
| masked_matched_packet | 0.875 | 0.422 | 6.00 | 110.9893 | 0.4721 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.3918 | 0.0006 |
| shuffled_source | 0.250 | 0.141 | 6.00 | 113.8688 | 0.4789 |
| answer_masked_source | 0.250 | 0.219 | 6.00 | 104.5859 | 0.4621 |
| random_same_byte | 0.250 | 0.266 | 6.00 | 34.7353 | 1.1188 |
| target_derived_sidecar | 0.250 | 0.359 | 6.00 | 84.6540 | 1.0786 |
| answer_only | 0.250 | 0.375 | 6.00 | 37.3976 | 1.1062 |
| structured_text_matched | 0.250 | 0.188 | 6.00 | 34.9928 | 0.5057 |
| wrong_projection_source | 0.250 | 0.172 | 6.00 | 102.5565 | 0.4676 |
| full_diag_oracle | 1.000 | 0.406 | 6.00 | 68.1875 | 0.5160 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
