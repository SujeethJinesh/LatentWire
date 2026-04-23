# C2C Teacher Innovation Probe

- date: `2026-04-23`
- gate: `candidate_teacher_recovery_explained_by_controls`
- target: `8 / 32`
- teacher: `16 / 32`
- teacher-only IDs: `10`

## Row Summary

| Role | Label | Correct | Wins vs target | Teacher-only recovered | Losses vs target | Numeric coverage |
|---|---|---:|---:|---:|---:|---:|
| source | `source` | 5/32 | 3 | 1 | 6 | 31/32 |
| source | `t2t` | 2/32 | 1 | 0 | 7 | 32/32 |
| control | `zero_source` | 8/32 | 2 | 1 | 2 | 32/32 |
| control | `shuffled_source` | 9/32 | 2 | 1 | 1 | 31/32 |
| control | `target_self_repair` | 14/32 | 6 | 3 | 0 | 32/32 |
| control | `selected_route_no_repair` | 10/32 | 2 | 1 | 0 | 32/32 |
| candidate | `query_pool_matched` | 9/32 | 2 | 1 | 1 | 32/32 |

## Candidate Control Overlap

| Candidate | Status | Teacher-only recovered | Max retained by control |
|---|---|---:|---:|
| `query_pool_matched` | `teacher_recovery_explained_by_controls` | 1 | 1 |

## Teacher-Only Provenance

| Example ID | Teacher norm | Source-correct | Control-correct | Candidate-correct |
|---|---:|---|---|---|
| 13cb77b698eeadb5 | luke played 177 rounds of the trivia game, so he scored a total of 177 x 46 = 8142 points. to find out how many points he scored in the game, we need to add the points he gained in each round to the total points he scored in | none | none | none |
| 1d50b408c8f5cd2c | on the previous day, there were 703 visitors. on the current day, there were 246 visitors. in total, there were 703 + 246 = 949 visitors. since there are 25 days in a year, there were 949 | none | none | none |
| 2de1549556000830 | the frog jumped 33 inches farther than the grasshopper, so the frog jumped 9 + 33 = 42 inches. the mouse jumped 3 inches lesser than the frog, so the mouse jumped 42 - 3 = 39 inches. #### 39 the answer | none | none | none |
| 4c84ebf42812703b | the recipe calls for 12 cups of flour and 5 cups of sugar. mary already put in some cups of flour, and she still needs 2 more cups of flour. so the total number of cups of flour she has put in is 12 - 2 = 10 cups. therefore | none | target_self_repair | none |
| 4d780f825bb8541c | julia played tag with 12 kids on monday and 14 kids on tuesday, so she played with a total of 12+14 = 26 kids. if she spent a total of 34 hours to play tag on both days, then she spent 34/26 | none | selected_route_no_repair, target_self_repair | none |
| 575d7e83d84c1e67 | emily has 6 beads and each necklace takes 3 beads, so she can make 6/3 = 2 necklaces. therefore, emily can make 2 necklaces for her 72 friends. #### 2 the answer is: 2 | source | shuffled_source, zero_source | query_pool_matched |
| 6e9745b37ab6fc45 | if there were 600 visitors the previous day, and 661 visitors came to the buckingham palace on that day, then there were 661 - 600 = 61 more visitors than the previous day. #### 61 the answer is: 61 | none | none | none |
| aee922049c757331 | dan has $ 4. he bought a candy bar for $ 8, so he has $ 4 - $ 8 = -$ 4 left. his friend gave him $ 5, so he now has $ -$ 4 + $ 5 = $ 1 left. #### 1 the | none | none | none |
| de1bf4d142544e5b | if there were 97 alligators and 40 of them were hiding, then the number of alligators not hiding is 97 - 40 = 57 #### 57 the answer is: 57 | none | target_self_repair | none |
| e3ab8666238a289e | there were 3 birds and 4 storks sitting on the fence. after 2 more birds joined, there were a total of 3 + 2 = 5 birds sitting on the fence. there were 4 storks sitting on the fence. so, there are 5 - 4 = 1 | none | none | none |

## Gate Notes

- Teacher-only recovery is fully reproduced by at least one provided control.

## Artifacts

- target: `results/svamp_exactid_baselines32_20260423/target_alone.jsonl` method `target_alone`
- teacher: `results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl` method `c2c_generate`
- source.source: `results/svamp_exactid_baselines32_20260423/source_alone.jsonl` method `source_alone`
- source.t2t: `results/svamp_exactid_baselines32_20260423/text_to_text.jsonl` method `text_to_text`
- control.zero_source: `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_zero_source.jsonl` method `rotalign_kv`
- control.shuffled_source: `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_shuffled_source_salt1.jsonl` method `rotalign_kv`
- control.target_self_repair: `results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl` method `target_self_repair`
- control.selected_route_no_repair: `results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl` method `selected_route_no_repair`
- candidate.query_pool_matched: `results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl` method `rotalign_kv`
