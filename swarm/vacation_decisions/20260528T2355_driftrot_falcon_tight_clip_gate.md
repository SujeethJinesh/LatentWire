# DriftRot Falcon Tight-Clip Gate

Date: 2026-05-28T23:55Z

## Decision

Run Falcon-H1-0.5B-Instruct with tight clip `[0.5, 2.0]`, reusing the existing
V1 Falcon BF16/static caches.

## Rationale

Granite tight clip is positive, while DeepSeek tight clip is ambiguous: lower
CI improves but median and worst trace regress. Falcon has a clear tail-control
surface under baseline ParoQuant: median recovery `0.381`, worst trace
`-1.323`, and no no-gap traces.

## Gate

Promote tight clip as a broader tail-control candidate if Falcon median remains
within `0.05` of baseline and worst trace or lower CI improves. Demote to
Granite-specific if Falcon median or worst trace regresses materially.
