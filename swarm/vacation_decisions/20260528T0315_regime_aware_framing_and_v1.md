# Regime-Aware Framing and V1 Baseline-Vetting Decision

Date: 2026-05-28T03:15Z

## Adopted Framing

The paper now frames long-reasoning W4A16 protection as regime-dependent:
static and hard-switch channel policies fail under drift, budgeted EMA succeeds
in the Nemotron MoE-hybrid regime, rotation succeeds in the Granite
dense-hybrid regime, and a cheap calibration protocol chooses a remedy or
rejects channel-set protection.

## Decision Rule Language

The decision rule is a prospective calibration protocol derived from the
four-model study and evaluated descriptively on those four models. It must not
be called pre-specified or validated unless a genuinely held-out frozen-threshold
run is executed after thresholds are frozen.

## V1 Action

Run V1 ParoQuant-on-Nemotron immediately as baseline-vetting. Compare against
the validated Nemotron M11b top-10 median recovery of 0.8147397989034302.

## M-FISH Constraint

M-FISH remains conditional and must be Fisher-weighted dynamic, not static
Fisher. It is only paper-central if it improves at least one non-Nemotron
architecture. Current feasibility supports a dense-Transformer DeepSeek gate;
hybrid cache autograd remains unresolved.

## Title

Channel-Set Drift in Long-Reasoning W4A16: Why Static Protection Fails and How
to Choose a Remedy
