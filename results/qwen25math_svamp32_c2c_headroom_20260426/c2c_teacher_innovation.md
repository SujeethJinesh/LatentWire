# C2C Teacher Innovation Probe

- date: `2026-04-26`
- gate: `no_candidates`
- target: `8 / 32`
- teacher: `15 / 32`
- teacher-only IDs: `9`

## Row Summary

| Role | Label | Correct | Wins vs target | Teacher-only recovered | Losses vs target | Numeric coverage |
|---|---|---:|---:|---:|---:|---:|
| source | `source` | 6/32 | 5 | 3 | 7 | 26/32 |
| source | `t2t` | 8/32 | 3 | 0 | 3 | 32/32 |

## Candidate Control Overlap

- No candidates provided.

## Teacher-Only Provenance

| Example ID | Teacher norm | Source-correct | Control-correct | Candidate-correct |
|---|---:|---|---|---|
| 14bfbfc94f2c2e7b | you have read 19 movies and watched 16 books. to find out how many more movies you have read than books, you need to subtract the number of books you have read from the number of movies you have read. so, you have read 19 - 16 = 3 more movies | source | none | none |
| 1d50b408c8f5cd2c | on the previous day, there were 703 visitors. on the current day, there were 246 visitors. in total, there were 703 + 246 = 949 visitors in total. if there were 25 days, then there were 949 | none | none | none |
| 2de1549556000830 | the frog jumped 33 inches farther than the grasshopper, so the frog jumped 9 + 33 = 42 inches. the mouse jumped 3 inches lesser than the frog, so the mouse jumped 42 - 3 = 39 inches. #### 39 the answer | source | none | none |
| 3e8a5691f5443495 | if each basket has 4 red peaches and 3 green peaches, then each basket has a total of 4+3=7 peaches. if there are a total of 7 peaches in all baskets, then there must be 1 basket of peaches. #### 1 the answer is | none | none | none |
| 47464cc0b064f172 | the waiter had 14 customers initially. after some left, he still had 14 customers. then he got 10 new customers, so he now has 14 + 10 = 24 customers. #### 24 the answer is: 24 | none | none | none |
| 4d780f825bb8541c | julia played tag with 12 kids on monday and 14 kids on tuesday, so she played with a total of 12+14=26 kids. if she spent a total of 34 hours to play tag on both days, then she spent 34/26= | source | none | none |
| 575d7e83d84c1e67 | emily has 6 beads and each necklace takes 3 beads, so she can make 6/3 = 2 necklaces. #### 2 the answer is: 2 | none | none | none |
| 6e9745b37ab6fc45 | if there were 600 visitors the previous day, and 661 visitors came to the buckingham palace on that day, then the difference in the number of visitors is 661 - 600 = 61 visitors. therefore, 61 more visitors visited the buckingham palace on that | none | none | none |
| de1bf4d142544e5b | if 40 alligators were hiding, then there are 97 - 40 = 57 alligators that were not hiding. #### 57 the answer is: 57 | none | none | none |

## Gate Notes

- No candidate rows were provided.
- No controls were provided, so source-specificity cannot be claimed.

## Artifacts

- target: `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl` method `target_alone`
- teacher: `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl` method `c2c_generate`
- source.source: `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl` method `source_alone`
- source.t2t: `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl` method `text_to_text`
