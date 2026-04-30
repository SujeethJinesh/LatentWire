# Source-Private Masked Consistency Receiver Smoke

- examples: `500`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.336`
- deterministic Hamming matched accuracy: `0.350`
- target-only accuracy: `0.250`
- best learned control: `shuffled_source` at `0.252`
- learned minus target: `0.086`
- learned minus best control: `0.084`
- learned minus Hamming: `-0.014`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0757 | 0.0005 |
| matched_consistency_packet | 0.336 | 0.350 | 6.00 | 0.3413 | 0.3813 |
| masked_matched_packet | 0.356 | 0.350 | 6.00 | 0.3448 | 0.3781 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0752 | 0.0005 |
| shuffled_source | 0.252 | 0.378 | 6.00 | 0.3414 | 0.3802 |
| answer_masked_source | 0.250 | 0.174 | 6.00 | 0.3350 | 0.3783 |
| random_same_byte | 0.250 | 0.238 | 6.00 | 0.1676 | 0.3778 |
| target_derived_sidecar | 0.250 | 0.368 | 6.00 | 0.2184 | 0.3781 |
| answer_only | 0.250 | 0.388 | 6.00 | 0.1661 | 0.3784 |
| structured_text_matched | 0.250 | 0.320 | 6.00 | 0.1662 | 0.3793 |
| wrong_projection_source | 0.242 | 0.360 | 6.00 | 0.3385 | 0.3784 |
| full_diag_oracle | 1.000 | 0.350 | 6.00 | 0.2183 | 0.3769 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
