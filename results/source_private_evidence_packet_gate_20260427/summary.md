# Source-Private Evidence Packet Gate

- examples: `128`
- syndrome bytes/example: `2`
- best no-source accuracy: `0.250`
- best source-destroying control accuracy: `0.250`
- matched minus best no-source: `0.750`
- matched minus best control: `0.750`
- matched minus matched-byte text relay: `0.750`
- pass gate: `True`

| Condition | Correct | Accuracy | Mean bytes | p50 latency ms |
|---|---:|---:|---:|---:|
| target_only | 32/128 | 0.250 | 0.00 | 0.0004 |
| target_wrapper | 32/128 | 0.250 | 0.00 | 0.0004 |
| matched_syndrome | 128/128 | 1.000 | 2.00 | 0.0026 |
| zero_source | 32/128 | 0.250 | 0.00 | 0.0004 |
| shuffled_source | 32/128 | 0.250 | 2.00 | 0.0027 |
| random_same_byte | 32/128 | 0.250 | 2.00 | 0.0023 |
| answer_only | 32/128 | 0.250 | 2.00 | 0.0026 |
| answer_masked | 32/128 | 0.250 | 0.00 | 0.0004 |
| target_only_sidecar | 32/128 | 0.250 | 2.00 | 0.0029 |
| structured_text_matched | 32/128 | 0.250 | 2.00 | 0.0007 |
| structured_text_full | 128/128 | 1.000 | 13.00 | 0.0004 |

Pass rule: matched_syndrome beats best no-source by >=0.15, source-destroying controls stay within +0.02 of no-source, and matched-byte structured text relay stays within +0.02 of no-source
