# Source-Private Masked Consistency Receiver Smoke

- examples: `256`
- pass gate: `False`
- source packet pass: `False`
- learned matched accuracy: `0.262`
- deterministic Hamming matched accuracy: `0.273`
- target-only accuracy: `0.250`
- best learned control: `wrong_projection_source` at `0.266`
- learned minus target: `0.012`
- learned minus best control: `-0.004`
- learned minus Hamming: `-0.012`

| Condition | Learned acc | Hamming acc | Payload B | Learned p50 ms | Hamming p50 ms |
|---|---:|---:|---:|---:|---:|
| target_only | 0.250 | 0.250 | 0.00 | 0.0971 | 0.0006 |
| matched_consistency_packet | 0.262 | 0.273 | 6.00 | 20.9415 | 0.4865 |
| masked_matched_packet | 0.281 | 0.273 | 6.00 | 20.4082 | 0.4828 |
| zero_source | 0.250 | 0.250 | 0.00 | 0.0994 | 0.0006 |
| shuffled_source | 0.223 | 0.238 | 6.00 | 20.7084 | 0.4927 |
| answer_masked_source | 0.250 | 0.359 | 6.00 | 20.6943 | 0.5015 |
| random_same_byte | 0.250 | 0.297 | 6.00 | 6.6116 | 0.4516 |
| target_derived_sidecar | 0.250 | 0.250 | 6.00 | 13.5278 | 0.4735 |
| answer_only | 0.250 | 0.297 | 6.00 | 6.2026 | 0.4572 |
| structured_text_matched | 0.250 | 0.230 | 6.00 | 6.9693 | 0.4985 |
| wrong_projection_source | 0.266 | 0.258 | 6.00 | 20.3461 | 0.5721 |
| full_diag_oracle | 1.000 | 0.219 | 6.00 | 13.4140 | 0.4764 |

Pass rule: Pass requires exact ID parity; learned matched packet >= target+0.15; learned matched packet >= best destructive control+0.15; all destructive controls <= target+0.05; and learned matched packet within 0.05 accuracy of deterministic Hamming packet decoding.
