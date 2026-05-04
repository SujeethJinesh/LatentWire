# HellaSwag Qwen-To-Phi Oracle Switch Decomposition Gate

- pass gate: `False`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- target-or-hybrid oracle accuracy: `0.604167`
- Qwen score top-2 oracle accuracy: `0.675781`
- target+hybrid+Qwen-top2 oracle accuracy: `0.776042`
- selected switcher accuracy: `0.460938`
- selected switcher delta: `-0.006510`
- selected switcher CI95 low: `-0.015625`
- forced switcher accuracy: `0.460938`
- eval-label diagnostic best accuracy: `0.467448`

## Interpretation

This gate decomposes the target-or-hybrid oracle gap. It tests whether the byte-scale Qwen packet plus receiver-local Phi scores can learn when to switch from Qwen hybrid to Phi. If the eval-label diagnostic is also near fixed hybrid, the current feature/packet surface does not expose the oracle headroom and a richer source-code dictionary or interface is needed.

## Lay Explanation

Qwen's tiny hint is usually better, but Phi is sometimes right when Qwen is wrong. This test tries to learn those moments without looking at the held-out answers. It also reports a cheating diagnostic to ask whether this feature set could recover the oracle gap even if we were allowed to tune on the test answers.
