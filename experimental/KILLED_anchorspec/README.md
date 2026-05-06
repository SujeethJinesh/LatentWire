# KILLED: AnchorSpec

## What Was Tried

AnchorSpec would use sink mass or anchor retention as a trigger for
self-speculative early exit in reasoning models.

## Why It Died

The early-exit/probe-guided reasoning space is crowded, and recent work narrows
the novelty of another trigger-only method. The branch would require a strong
head-to-head against dedicated early-exit systems before becoming publishable.

## Salvage Value

Sink mass and anchor telemetry remain useful diagnostics for SinkKV and
ThoughtFlow. They should not be promoted as an early-exit method without a new
novelty audit and preregistration.
