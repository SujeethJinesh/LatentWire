# Source-Private Masked Consistency Receiver Smoke

- examples: `500`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.302`
- deterministic Hamming matched accuracy: `0.214`
- target-only accuracy: `0.250`
- best learned control: `zero_source` at `0.250`
- learned minus target: `0.052`
- learned minus best control: `0.052`
- learned minus Hamming: `0.088`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0761 | 0.0005 |
| matched_consistency_packet | 0.302 | 0.214 | 6.00 | 0.3411 | 0.3809 |
| masked_matched_packet | 0.326 | 0.214 | 6.00 | 0.3455 | 0.3787 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0757 | 0.0005 |
| shuffled_source | 0.244 | 0.270 | 6.00 | 0.3415 | 0.3795 |
| answer_masked_source | 0.250 | 0.314 | 6.00 | 0.3359 | 0.3775 |
| random_same_byte | 0.250 | 0.230 | 6.00 | 0.1684 | 0.3774 |
| target_derived_sidecar | 0.250 | 0.216 | 6.00 | 0.2192 | 0.3765 |
| answer_only | 0.250 | 0.156 | 6.00 | 0.1671 | 0.3766 |
| structured_text_matched | 0.250 | 0.196 | 6.00 | 0.1671 | 0.3763 |
| wrong_projection_source | 0.250 | 0.238 | 6.00 | 0.3396 | 0.3776 |
| full_diag_oracle | 1.000 | 0.226 | 6.00 | 0.2191 | 0.3757 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
