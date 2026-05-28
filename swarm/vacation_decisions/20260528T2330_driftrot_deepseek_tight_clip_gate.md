# DriftRot DeepSeek Tight-Clip Gate

Date: 2026-05-28T23:30Z

## Decision

Run DeepSeek-R1-Distill-Qwen-1.5B with tight clip `[0.5, 2.0]`, reusing the
existing V1 ParoQuant DeepSeek BF16/static caches.

## Rationale

Granite tight clip now passes all recoverable positive-gap traces, improving
median recovery `0.754 -> 0.922` and fixing the severe negative tail. This is
useful but high novelty risk if left as a single-model config retune.

DeepSeek is the cheapest next cross-model tail-control surface because the V1
ParoQuant DeepSeek packet already has a strong median (`0.756`) but a negative
CI lower bound (`-0.246`). Tight clip should only stay live as a DriftRot
tail-control candidate if it improves or preserves the DeepSeek tail without
materially degrading median recovery.

## Gate

Promote cross-model tight-clip evidence if:

- DeepSeek median recovery remains within `0.05` of baseline ParoQuant, and
- CI lower bound improves, or the worst included trace improves.

Demote to Granite-only tail-control evidence if tight clip reduces DeepSeek
median materially or worsens the negative tail.

This remains a Tier-2 DriftRot candidate, not the headline method.
