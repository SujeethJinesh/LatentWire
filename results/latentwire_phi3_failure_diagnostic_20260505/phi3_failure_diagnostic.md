# Phi-3 Failure Diagnosis

- date: `2026-05-05`
- interpretation: `source-choice/family boundary, not decoded-packet corruption`

## Source Cache

| Split | n | Phi-3 source acc | Qwen source acc | Phi-3/Qwen choice agree | both correct | Phi-3 only | Qwen only | both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 1172 | 0.246 | 0.346 | 0.289 | 121 | 167 | 284 | 600 |
| validation | 299 | 0.274 | 0.388 | 0.254 | 30 | 52 | 86 | 131 |

## Packet Aggregates

| Surface | Seeds | n/seed | Acc | Follow Phi-3 | Follow Qwen |
|---|---:|---:|---:|---:|---:|
| test:matched_source_private_packet | 10 | 1172 | 0.244 | 0.997 | 0.288 |
| validation:matched_source_private_packet | 10 | 299 | 0.277 | 0.995 | 0.258 |
| test:qwen_substituted_packet | 10 | 833 | 0.340 | 0.002 | 0.996 |
| validation:qwen_substituted_packet | 10 | 223 | 0.383 | 0.001 | 0.997 |

## Conditional Packet Accuracy

| Split | Condition | Source category | Items | Acc | Follow Phi-3 | Follow Qwen |
|---|---|---|---:|---:|---:|---:|
| test | matched_source_private_packet | both_correct | 121 | 0.988 | 0.988 | 0.988 |
| test | matched_source_private_packet | both_wrong | 600 | 0.001 | 0.998 | 0.363 |
| test | matched_source_private_packet | phi3_only_correct | 167 | 0.997 | 0.997 | 0.001 |
| test | matched_source_private_packet | qwen_only_correct | 284 | 0.000 | 0.999 | 0.000 |
| validation | matched_source_private_packet | both_correct | 30 | 0.997 | 0.997 | 0.997 |
| validation | matched_source_private_packet | both_wrong | 131 | 0.000 | 0.999 | 0.350 |
| validation | matched_source_private_packet | phi3_only_correct | 52 | 0.998 | 0.998 | 0.002 |
| validation | matched_source_private_packet | qwen_only_correct | 86 | 0.013 | 0.987 | 0.013 |
| test | qwen_substituted_packet | both_wrong | 382 | 0.002 | 0.001 | 0.996 |
| test | qwen_substituted_packet | phi3_only_correct | 167 | 0.003 | 0.003 | 0.997 |
| test | qwen_substituted_packet | qwen_only_correct | 284 | 0.994 | 0.004 | 0.994 |
| validation | qwen_substituted_packet | both_wrong | 85 | 0.000 | 0.000 | 1.000 |
| validation | qwen_substituted_packet | phi3_only_correct | 52 | 0.000 | 0.000 | 1.000 |
| validation | qwen_substituted_packet | qwen_only_correct | 86 | 0.992 | 0.003 | 0.992 |

## Readout

The Phi-3 failure is primarily a source-choice/family boundary result, not packet corruption: on the ARC test cache Phi-3's source-choice accuracy is 0.246 versus Qwen's 0.346, Phi-3 and Qwen choose the same candidate only 0.289 of the time, and the decoded packet follows the Phi-3 source choice at 0.997. The current packet therefore faithfully transports the alternate source's candidate preference, but that preference is weaker and often different from the same-family Qwen source.
