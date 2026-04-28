# Source-Private Test-Log Packet Leakage Audit

- examples: `160`
- exact ID SHA256: `fcfd2cfcecfa51f4caae6e5de39cf0632dafb634e4f19db0dfdc12c2ef8dbd2e`
- public target answer-label candidate-pool hits: `160`
- public target private-log hits: `0`
- public target TRACE_SIG hits: `0`

## Over-Budget Counts

| Budget | Condition | Over budget | Parse failures | Answer-label copies | Candidate-label copies | TRACE_SIG field copies |
|---:|---|---:|---:|---:|---:|---:|
| 2 | target_only | 0 | 0 | 0 | 0 | 0 |
| 2 | target_wrapper | 0 | 0 | 0 | 0 | 0 |
| 2 | matched_testlog_packet | 0 | 0 | 0 | 0 | 0 |
| 2 | zero_source | 0 | 0 | 0 | 0 | 0 |
| 2 | shuffled_source | 0 | 0 | 0 | 0 | 0 |
| 2 | random_same_byte | 0 | 0 | 0 | 0 | 0 |
| 2 | answer_only | 0 | 0 | 0 | 0 | 0 |
| 2 | answer_masked | 0 | 0 | 0 | 0 | 0 |
| 2 | target_derived_sidecar | 0 | 0 | 0 | 0 | 0 |
| 2 | structured_text_matched | 0 | 160 | 0 | 0 | 0 |
| 2 | full_structured_log | 160 | 0 | 0 | 0 | 0 |
| 2 | full_signature_text | 160 | 0 | 0 | 0 | 160 |
| 4 | target_only | 0 | 0 | 0 | 0 | 0 |
| 4 | target_wrapper | 0 | 0 | 0 | 0 | 0 |
| 4 | matched_testlog_packet | 0 | 0 | 0 | 0 | 0 |
| 4 | zero_source | 0 | 0 | 0 | 0 | 0 |
| 4 | shuffled_source | 0 | 0 | 0 | 0 | 0 |
| 4 | random_same_byte | 0 | 0 | 0 | 0 | 0 |
| 4 | answer_only | 0 | 0 | 0 | 0 | 0 |
| 4 | answer_masked | 0 | 0 | 0 | 0 | 0 |
| 4 | target_derived_sidecar | 0 | 0 | 0 | 0 | 0 |
| 4 | structured_text_matched | 0 | 160 | 0 | 0 | 0 |
| 4 | full_structured_log | 160 | 0 | 0 | 0 | 0 |
| 4 | full_signature_text | 160 | 0 | 0 | 0 | 160 |
| 8 | target_only | 0 | 0 | 0 | 0 | 0 |
| 8 | target_wrapper | 0 | 0 | 0 | 0 | 0 |
| 8 | matched_testlog_packet | 0 | 0 | 0 | 0 | 0 |
| 8 | zero_source | 0 | 0 | 0 | 0 | 0 |
| 8 | shuffled_source | 0 | 0 | 0 | 0 | 0 |
| 8 | random_same_byte | 0 | 0 | 0 | 0 | 0 |
| 8 | answer_only | 0 | 0 | 0 | 0 | 0 |
| 8 | answer_masked | 0 | 0 | 0 | 0 | 0 |
| 8 | target_derived_sidecar | 0 | 0 | 0 | 0 | 0 |
| 8 | structured_text_matched | 0 | 160 | 0 | 0 | 0 |
| 8 | full_structured_log | 160 | 0 | 0 | 0 | 0 |
| 8 | full_signature_text | 160 | 0 | 0 | 0 | 160 |
| 16 | target_only | 0 | 0 | 0 | 0 | 0 |
| 16 | target_wrapper | 0 | 0 | 0 | 0 | 0 |
| 16 | matched_testlog_packet | 0 | 0 | 0 | 0 | 0 |
| 16 | zero_source | 0 | 0 | 0 | 0 | 0 |
| 16 | shuffled_source | 0 | 0 | 0 | 0 | 0 |
| 16 | random_same_byte | 0 | 0 | 0 | 0 | 0 |
| 16 | answer_only | 0 | 0 | 0 | 0 | 0 |
| 16 | answer_masked | 0 | 0 | 0 | 0 | 0 |
| 16 | target_derived_sidecar | 0 | 0 | 0 | 0 | 0 |
| 16 | structured_text_matched | 0 | 160 | 0 | 0 | 0 |
| 16 | full_structured_log | 160 | 0 | 0 | 0 | 0 |
| 16 | full_signature_text | 0 | 0 | 0 | 0 | 160 |
| 32 | target_only | 0 | 0 | 0 | 0 | 0 |
| 32 | target_wrapper | 0 | 0 | 0 | 0 | 0 |
| 32 | matched_testlog_packet | 0 | 0 | 0 | 0 | 0 |
| 32 | zero_source | 0 | 0 | 0 | 0 | 0 |
| 32 | shuffled_source | 0 | 0 | 0 | 0 | 0 |
| 32 | random_same_byte | 0 | 0 | 0 | 0 | 0 |
| 32 | answer_only | 0 | 0 | 0 | 0 | 0 |
| 32 | answer_masked | 0 | 0 | 0 | 0 | 0 |
| 32 | target_derived_sidecar | 0 | 0 | 0 | 0 | 0 |
| 32 | structured_text_matched | 0 | 160 | 0 | 0 | 0 |
| 32 | full_structured_log | 160 | 0 | 0 | 0 | 0 |
| 32 | full_signature_text | 0 | 0 | 0 | 0 | 160 |

Matched test-log packets intentionally transmit the two-byte trace signature. Leakage counts focus on answer/candidate-label copies and accidental TRACE_SIG field leakage outside the full-log oracle.
