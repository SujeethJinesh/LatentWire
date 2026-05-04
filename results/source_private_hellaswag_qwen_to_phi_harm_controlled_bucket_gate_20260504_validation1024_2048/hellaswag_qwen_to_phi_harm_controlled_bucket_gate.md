# HellaSwag Qwen-To-Phi Harm-Controlled Bucket Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- harm-controlled accuracy: `0.467448`
- harm-controlled delta: `0.000000`
- harm-controlled CI95 low: `0.000000`
- overrides / helps / harms: `0 / 0 / 0`
- hybrid/rival/Phi oracle accuracy: `0.766927`

## Interpretation

This gate tests the highest-priority receiver branch after shallow linear switchers failed. It uses official-train labels to select only low-harm complementarity buckets, then freezes that rule on Qwen-to-Phi validation. The receiver sees quantized source packet fields and Phi-local side information, not raw Qwen score vectors.

## Lay Explanation

Qwen sends Phi a tiny message naming its safe answer, its backup answer, and a few coarse confidence levels. Phi then uses a rule learned on training questions: only switch away from the safe answer in buckets where training showed switches usually helped and rarely hurt.
