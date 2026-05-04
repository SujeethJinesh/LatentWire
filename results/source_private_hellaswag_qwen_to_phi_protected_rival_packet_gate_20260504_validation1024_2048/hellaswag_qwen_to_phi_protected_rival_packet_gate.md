# HellaSwag Qwen-To-Phi Protected Rival Packet Gate

- pass gate: `False`
- eval rows: `768`
- selected source mode: `code16_hybrid_rival_policy`
- selected l2: `3000.0`
- fixed hybrid accuracy: `0.467448`
- protected pair decoder accuracy: `0.462240`
- protected pair decoder delta: `-0.005208`
- protected pair decoder CI95 low: `-0.013021`
- hybrid-rival oracle accuracy: `0.678385`
- source top-2 oracle accuracy: `0.675781`
- best destructive control: `source_score_row_shuffle_before_encoding_pair_decoder_control` at `0.467448`

## Interpretation

This gate directly tests the source-score top-2 headroom exposed by the previous oracle switch decomposition. The pair oracles are high, but the receiver must learn a reliable pairwise choice from only fit/select labels and Phi side information. A failure means the shared candidate-ID basis contains headroom but the current tiny calibration surface and low-rate pair features do not expose a safe decision rule.

## Lay Explanation

Qwen often has the right answer somewhere in its top two guesses. This test sends Phi the two Qwen-side choices that matter most and asks Phi to pick between them. If Phi still picks the wrong one too often, the problem is not that the rival was hidden; it is that we do not yet know how to teach Phi when that rival is better.
