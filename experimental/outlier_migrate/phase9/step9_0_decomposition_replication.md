# Phase 9 Step 9.0: Decomposition Replication

This no-GPU check recomputes the strict set-leaving / within-set rank-shuffling decomposition from existing activation packets before any Phase 9 method run.

Definitions:

- Strict set-leaving: a channel ranked inside the top-1% boundary at decode position 100 has rank outside that boundary at decode position 20000.
- Within-set rank shuffling: a channel remains inside the top-1% set at decode position 20000 but moves by more than two rank positions.
- Original drift metric: the earlier rank-change metric, which conflates set-leaving and within-set shuffling.

| Model | Run | Strict set-leaving | Within-set shuffling | Original drift | Gate status |
| --- | --- | ---: | ---: | ---: | --- |
| Granite-4-H-Small | `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z` | `0.566234756098` | `0.270934959350` | `0.836763211382` | PASS: set-leaving > 0.50 |
| Nemotron-3-Nano | `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z` | `0.533713200380` | `0.269082383666` | `0.801667853751` | PASS: set-leaving > 0.50 |
| DeepSeek-R1-Distill-Qwen-1.5B | `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z` | `0.670572916667` | `0.165736607143` | `0.834914434524` | PASS: set-leaving > 0.50 |

## Decision Gate

- Set-leaving above `0.50` on all three required models: `true`.
- If this value is `false`, Phase 9 must stop and surface to the human because the paper premise is at risk.
