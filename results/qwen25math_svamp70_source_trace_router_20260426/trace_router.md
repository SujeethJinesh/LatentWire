# Source-Trace Self-Consistency Router Gate

- date: `2026-04-26`
- status: `source_trace_router_fails_gate`
- frozen feature: `source_answer_reused_in_trace`
- frozen direction: `ge`
- frozen threshold: `0.5`

### Live CV

- status: `fails`
- failing criteria: `min_correct, min_clean_source_necessary, max_accepted_harm`
- matched correct: `20`
- clean source-necessary: `1`
- clean control union: `0`
- accepted harm: `2`
- equation-permuted retained source-necessary: `0`
- source-necessary IDs: `2de1549556000830`

### Holdout Frozen

- status: `fails`
- failing criteria: `min_clean_source_necessary, equation_permutation_loses_half`
- matched correct: `10`
- clean source-necessary: `1`
- clean control union: `0`
- accepted harm: `0`
- equation-permuted retained source-necessary: `1`
- source-necessary IDs: `daea537474de16ac`

## Features

- `source_final_value_matches_last_equation`
- `source_equation_valid_fraction`
- `prompt_number_coverage`
- `source_answer_reused_in_trace`
- `valid_add_count`
- `valid_sub_count`
- `valid_mul_count`
- `valid_div_count`
