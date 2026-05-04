# HellaSwag Qwen-To-Phi Official-Train Source Dictionary Gate

- pass gate: `False`
- official calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- source dictionary accuracy: `0.429688`
- source dictionary delta: `-0.037760`
- source dictionary CI95 low: `-0.063802`
- official dev delta: `0.008065`
- hybrid-rival oracle accuracy: `0.678385`

## Interpretation

This gate tests whether the official-train source side alone can learn the protected-rival decision frontier. It uses out-of-bag train rows to avoid scoring a row with a packet model that trained on that same row. A failure means the larger-data source dictionary does not solve the Qwen-to-Phi blocker without receiver-side Phi calibration or a richer interface.

## Lay Explanation

We trained Qwen on many official training questions to learn when its backup answer should replace its safe answer. That looked useful on Qwen's training-dev split, but when frozen and tested on the held-out Qwen-to-Phi slice it made too many bad swaps.
