# C2C Teacher Innovation Probe

- date: `2026-05-04`
- gate: `candidate_recovers_teacher_innovations_without_control_explanation`
- target: `8 / 32`
- teacher: `16 / 32`
- teacher-only IDs: `10`

## Row Summary

| Role | Label | Correct | Wins vs target | Teacher-only recovered | Losses vs target | Numeric coverage |
|---|---|---:|---:|---:|---:|---:|
| source | `source_alone` | 5/32 | 3 | 0 | 6 | 31/32 |
| source | `text_to_text` | 2/32 | 1 | 0 | 7 | 32/32 |
| control | `source_alone` | 5/32 | 3 | 0 | 6 | 31/32 |
| control | `text_to_text` | 2/32 | 1 | 0 | 7 | 32/32 |
| candidate | `archived_c2c` | 16/32 | 10 | 9 | 2 | 32/32 |

## Candidate Control Overlap

| Candidate | Status | Teacher-only recovered | Max retained by control |
|---|---|---:|---:|
| `archived_c2c` | `teacher_recovery_not_explained_by_controls` | 9 | 0 |

## Teacher-Only Provenance

| Example ID | Teacher norm | Source-correct | Control-correct | Candidate-correct |
|---|---:|---|---|---|
| 13cb77b698eeadb5 | luke played 177 rounds of the trivia game, so he scored a total of 177 x 46 = 8142 points. to find out how many points he scored in the game, we need to add the points he scored in each round to the total points he scored in | none | none | archived_c2c |
| 1d50b408c8f5cd2c | on the previous day, there were 703 visitors. on the current day, there were 246 visitors. in total, there were 703 + 246 = 949 visitors. since there are 25 days in a month, there were 949 | none | none | archived_c2c |
| 2de1549556000830 | the frog jumped 33 inches farther than the grasshopper, so the frog jumped 9 + 33 = 42 inches. the mouse jumped 3 inches lesser than the frog, so the mouse jumped 42 - 3 = 39 inches. #### 39 the answer | none | none | archived_c2c |
| 4c84ebf42812703b | the recipe calls for 12 cups of flour and mary already put in some cups of flour. if she still needs 2 more cups of flour, then she has already put in 12 - 2 = 10 cups of flour. therefore, mary put in 10 cups of flour. #### | none | none | archived_c2c |
| 4d780f825bb8541c | julia played tag with 12 kids on monday and 14 kids on tuesday, so she played with a total of 12+14 = 26 kids. if she spent a total of 34 hours to play tag on both days, then she spent 34/26 | none | none | archived_c2c |
| 6e9745b37ab6fc45 | if there were 600 visitors the previous day, and 661 visitors came to the buckingham palace on that day, then there were 661 - 600 = 61 more visitors than the previous day. #### 61 the answer is: 61 | none | none | archived_c2c |
| aee922049c757331 | dan has $ 4. he bought a candy bar for $ 8, so he has $ 4 - $ 8 = -$ 4 left. his friend gave him $ 5, so he now has $ -$ 4 + $ 5 = $ 1 left. #### 1 the | none | none | archived_c2c |
| b1200c32546a34a5 | the total number of bananas and oranges is 142 + 356 = 498 the total number of groups is 47 + 178 = 225 the size of each group of oranges is 498 / 225 = 2 | none | none | none |
| de1bf4d142544e5b | if there were 97 alligators in total and 40 of them were hiding, then the number of alligators not hiding is 97 - 40 = 57 #### 57 the answer is: 57 | none | none | archived_c2c |
| e3ab8666238a289e | there were 3 birds and 4 storks sitting on the fence. after 2 more birds came to join them, there were a total of 3 + 2 = 5 birds sitting on the fence. so, there are 5 - 4 = 1 more bird than storks sitting on the | none | none | archived_c2c |

## Gate Notes

- At least one candidate recovers teacher-only IDs not recovered by provided controls.

## Artifacts

- target: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` method `target_alone`
- teacher: `results/svamp32_c2c_mps_compat_replay_20260505/c2c_generate.jsonl` method `c2c_generate`
- source.source_alone: `results/svamp_exactid_baselines32_20260423/source_alone.jsonl` method `source_alone`
- source.text_to_text: `results/svamp_exactid_baselines32_20260423/text_to_text.jsonl` method `text_to_text`
- control.source_alone: `results/svamp_exactid_baselines32_20260423/source_alone.jsonl` method `source_alone`
- control.text_to_text: `results/svamp_exactid_baselines32_20260423/text_to_text.jsonl` method `text_to_text`
- candidate.archived_c2c: `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl` method `c2c_generate`
