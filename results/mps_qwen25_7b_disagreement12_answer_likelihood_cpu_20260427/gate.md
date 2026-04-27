# Condition-Specific Likelihood Receiver Gate

- date: `2026-04-27`
- status: `condition_likelihood_receiver_fails_gate`
- frozen feature: `margin`
- frozen direction: `ge`
- frozen threshold: `1.4828551919199526`
- conditions: `matched, shuffled_source, target_only, slots_only, answer_only, answer_masked_source`

### Live CV

- status: `fails`
- failing criteria: `min_clean_source_necessary, max_clean_control_union`
- matched correct: `1`
- matched accepted: `7`
- clean source-necessary: `0`
- clean control union: `1`
- duplicate-answer clean IDs: `0`
- accepted harm: `0`
- mean sidecar bits: `8.000`
- source-necessary IDs: none

### Holdout Frozen

- status: `fails`
- failing criteria: `min_clean_source_necessary, max_clean_control_union`
- matched correct: `4`
- matched accepted: `8`
- clean source-necessary: `0`
- clean control union: `2`
- duplicate-answer clean IDs: `0`
- accepted harm: `0`
- mean sidecar bits: `8.000`
- source-necessary IDs: none
