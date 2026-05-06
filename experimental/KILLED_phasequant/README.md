# KILLED: PhaseQuant

## What Was Tried

PhaseQuant would condition KV or activation precision on reasoning-phase
signals.

## Why It Died

ThoughtFlow showed that the available phase/saliency signals are fragile across
surfaces. A quantization method downstream of an unstable phase classifier is
too risky for the current sprint.

## Salvage Value

Phase markers remain useful for diagnostic stratification. They should be used
to analyze failures, not to drive a headline quantization policy.
