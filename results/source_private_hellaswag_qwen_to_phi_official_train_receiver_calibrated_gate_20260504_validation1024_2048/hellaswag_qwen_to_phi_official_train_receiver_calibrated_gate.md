# HellaSwag Qwen-To-Phi Official-Train Receiver-Calibrated Gate

- pass gate: `False`
- calibration rows: `1487`
- Phi train score cache hit: `True`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- receiver-calibrated accuracy: `0.466146`
- receiver-calibrated delta: `-0.001302`
- receiver-calibrated CI95 low: `-0.007812`
- hybrid/rival/Phi oracle accuracy: `0.766927`

## Interpretation

This gate tests the next live branch after the source-only dictionary failed: learn the receiver decision on official-train rows where both Qwen packet features and Phi local score features are available. A pass would show that receiver-side calibration, not more source-only fitting, unlocks the Qwen-to-Phi candidate frontier.

## Lay Explanation

We let Phi practice on training questions too. For each question, the rule sees Qwen's safe and backup answers plus Phi's own four answer scores, then learns whether to keep Qwen's safe answer, take Qwen's backup, or trust Phi. The frozen rule is then tested on held-out questions.
