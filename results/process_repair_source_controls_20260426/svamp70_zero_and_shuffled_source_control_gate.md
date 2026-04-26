# Process Repair Source-Control Gate

- date: `2026-04-26`
- status: `process_repair_source_controls_do_not_clear_gate`
- method: `process_repair_selected_route`
- matched correct: `38/70`
- target correct: `21/70`
- target-self repair correct: `35/70`
- matched-only vs target IDs: `17`
- matched-only vs target-self IDs: `3`
- source-specific vs target-self after controls: `0`

## Controls

| Control | Correct | Overlap With Matched-Only vs Target-Self | Control-Only vs Target-Self |
|---|---:|---:|---:|
| zero_source_kv | 35/70 | 1 | 1 |
| shuffled_source_prompt | 37/70 | 3 | 5 |

## Source-Specific IDs

- matched-only vs target-self: `a4e792e50d68217b`, `b0b71023ae4d233b`, `cfc46d556ee67fd8`
- retained after controls: none

## Decision

all matched-only-vs-target-self IDs must be absent from source-destroying controls; passed: `False`.
