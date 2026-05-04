# HellaSwag Qwen-To-Phi Held-Out Uncertainty Router Gate

- pass gate: `False`
- calibration rows: `1487`
- eval rows: `768`
- fixed hybrid accuracy: `0.467448`
- uncertainty-router accuracy: `0.466146`
- uncertainty-router delta: `-0.001302`
- uncertainty-router CI95 low: `-0.003906`
- overrides / helps / harms: `3 / 0 / 1`
- source top1/top2 oracle accuracy: `0.675781`
- best destructive: `target_derived_source_packet_router_control` (`0.467448`)

## Interpretation

This gate is the strongest bounded test of the shallow uncertainty-router branch: it uses official-train calibration and a smooth ridge scorer, but the receiver-visible source side remains quantized candidate IDs and uncertainty bins rather than raw Qwen scores.

## Lay Explanation

Qwen sends a tiny coarse message saying which answers it thinks are most plausible and how confident it is. Phi uses its own scores plus a rule learned on training examples to decide whether to keep Qwen's safe answer, use Qwen's backup, or trust Phi's own favorite.
