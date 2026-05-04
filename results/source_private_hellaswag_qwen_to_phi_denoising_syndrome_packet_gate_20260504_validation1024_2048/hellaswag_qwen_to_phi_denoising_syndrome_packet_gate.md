# HellaSwag Qwen-To-Phi Denoising Syndrome Packet Gate

- pass gate: `False`
- eval rows: `768`
- selected source mode: `code8_hybrid_selected_margin`
- selected source l2: `300.0`
- fixed hybrid accuracy: `0.467448`
- denoising syndrome accuracy: `0.463542`
- delta vs fixed hybrid: `-0.003906`
- CI95 low vs fixed hybrid: `-0.010417`
- target-or-hybrid oracle accuracy: `0.604167`

## Interpretation

This gate tests the highest-priority Mac-local rate-distortion branch: a tiny source syndrome is decoded with Phi's local score simplex as side information. A pass would be a real receiver-side positive method over fixed hybrid. A failure weakens this ridge-denoising version of the syndrome idea while preserving oracle headroom for stronger methods.

## Lay Explanation

The source sends a tiny correction clue. Phi then uses its own four answer scores to decide whether that clue should repair its answer. The controls check whether the clue only helps when it comes from the right source example.
