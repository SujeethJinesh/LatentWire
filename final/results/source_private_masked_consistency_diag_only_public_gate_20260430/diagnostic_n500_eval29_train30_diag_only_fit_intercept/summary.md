# Source-Private Masked Consistency Receiver Smoke

- examples: `500`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.346`
- deterministic Hamming matched accuracy: `0.348`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.096`
- learned minus best control: `0.096`
- learned minus Hamming: `-0.002`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0740 | 0.0005 |
| matched_consistency_packet | 0.346 | 0.348 | 6.00 | 0.3355 | 0.3719 |
| masked_matched_packet | 0.344 | 0.348 | 6.00 | 0.3413 | 0.3705 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0736 | 0.0005 |
| shuffled_source | 0.250 | 0.370 | 6.00 | 0.3363 | 0.3710 |
| answer_masked_source | 0.250 | 0.186 | 6.00 | 0.3314 | 0.3699 |
| random_same_byte | 0.250 | 0.238 | 6.00 | 0.1648 | 0.3689 |
| target_derived_sidecar | 0.250 | 0.368 | 6.00 | 0.2140 | 0.3684 |
| answer_only | 0.250 | 0.388 | 6.00 | 0.1635 | 0.3696 |
| structured_text_matched | 0.250 | 0.320 | 6.00 | 0.1636 | 0.3690 |
| wrong_projection_source | 0.244 | 0.350 | 6.00 | 0.3345 | 0.3688 |
| full_diag_oracle | 1.000 | 0.350 | 6.00 | 0.2149 | 0.3690 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
