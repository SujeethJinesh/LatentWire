# Fresh MMLU-Pro Mac Continue Summary

- headline: `BOUNDED_NEGATIVE`
- status: `MAC_DONE`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- split counts: `{'dev': 73, 'confirm': 26, 'gate': 23}`
- scored dev/gate rows: `96`
- confirm rows scored: `0`
- I_beyond bits: `2.231310`
- source_top1 MI bits: `0.558794`
- source_scores MI bits: `2.790104`

## WZ / Score Packet

- selected alpha: `-0.4`
- gate accuracy: `{'wz': 0.2608695652173913, 'target_only': 0.17391304347826086, 'source_index': 0.2608695652173913, 'wrong_row': 0.21739130434782608, 'derangement': 0.21739130434782608, 'random_same_byte': 0.30434782608695654, 'label_shuffle': 0.21739130434782608}`
- best baseline: `random_same_byte`
- delta_beyond_score: `{'n': 23, 'delta': 0.0, 'ci95_low': -0.2608695652173913, 'ci95_high': 0.21739130434782608}`
- delta_beyond_label: `{'n': 23, 'delta': 0.043478260869565216, 'ci95_low': -0.13043478260869565, 'ci95_high': 0.2608695652173913}`
- delta_vs_best_baseline: `{'n': 23, 'delta': -0.043478260869565216, 'ci95_low': -0.30434782608695654, 'ci95_high': 0.2608695652173913}`
- controls: `{'wrong_row': 0.21739130434782608, 'derangement': 0.21739130434782608, 'random_same_byte': 0.30434782608695654}`

## L-B1 Damage Avoidance

- gate accuracy: `0.173913`
- source-index accuracy: `0.260870`
- target accuracy: `0.173913`
- AURC delta vs source-index+confidence: `-0.222541`
- risk@coverage50: `0.750000`
- accuracy@coverage50: `0.250000`
- damage reduction: `2` of `2`
- repair rate: `0.000000`
- candidate leakage accuracy: `0.260870`
- packet<->source-top1 MI bits: `0.570322`

## Rerank Slice

- status: `MAC_DONE`
- generated rows: `3` / `3`
