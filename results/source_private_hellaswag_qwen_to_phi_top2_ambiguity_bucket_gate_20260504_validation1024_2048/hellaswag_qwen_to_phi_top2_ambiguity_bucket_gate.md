# HellaSwag Qwen-To-Phi Top1/Top2 Ambiguity Bucket Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- ambiguity-bucket accuracy: `0.467448`
- ambiguity-bucket delta: `0.000000`
- ambiguity-bucket CI95 low: `0.000000`
- overrides / helps / harms: `0 / 0 / 0`
- source top1/top2 oracle accuracy: `0.675781`
- no-syndrome top-pair accuracy: `0.467448`
- best destructive: `source_score_row_shuffle_before_encoding_bucket_control` (`0.467448`)

## Interpretation

This gate tests whether the remaining Qwen top1/top2 oracle headroom can be recovered by a source-private ambiguity packet selected on official train and frozen on validation. It is not promotable unless it separates from no-syndrome source-index and destructive packet controls.

## Lay Explanation

Qwen sends Phi its two best guesses and a few tiny confidence clues. Phi only switches away from the fixed safe answer in clue-patterns that helped on training questions. If the same behavior appears when the clues are shuffled, zeroed, or made from Phi itself, the packet is not useful cross-model communication.
