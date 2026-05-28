# Preregistration: DriftRot Granite Clip/CVaR Retune

Date: 2026-05-28

## Purpose

Test whether a drift-aware ParoQuant configuration change can improve the
Granite-Small ParoQuant tail/CI without claiming ParoQuant itself as our
method.

## Baseline

The baseline is the existing ParoQuant-style rotation packet:

- run: `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z`
- group size: 128
- rotations: 8
- scale clip: `[0.25, 4.0]`

## Candidate Configurations

All candidates keep the same group size, rotation count, pairing rule, prompt
set, scoring window, and BF16/static references as the baseline.

1. `clip_loose`: scale clip `[0.125, 8.0]`
2. `clip_tight`: scale clip `[0.5, 2.0]`

The baseline `[0.25, 4.0]` remains in the comparison table.

## Calibration / Confirmation Split

The positive-gap Granite traces are split before running candidate scoring:

- calibration traces: prompt indices `[1, 2, 5, 8]`
- confirmation traces: prompt indices `[4, 7, 9, 10]`
- no-gap traces: prompt indices `[0, 3, 6, 11]`, reported but excluded from
  recovery medians as in prior packets

Trace 4, the known severe ParoQuant negative tail in the baseline packet, is
held out for confirmation. This makes the screen stricter: a selected config
must improve the tail without being selected on that tail trace.

## Selection Rule

On calibration traces, choose the candidate with the best lexicographic
objective:

1. no median recovery loss greater than 0.05 versus baseline;
2. highest worst-trace recovery;
3. highest median recovery.

If both candidates lose more than 0.05 median recovery on calibration, no
candidate is promoted.

## Confirmation Pass Criteria

On confirmation traces, a candidate passes this screen if any holds:

- median recovery beats baseline by at least 0.05;
- worst-trace recovery improves by at least 0.10 without median recovery loss
  greater than 0.05;
- bootstrap CI lower bound over positive-gap confirmation traces improves by at
  least 0.10.

This is a DriftRot screen, not a full method claim. A paper-level method claim
requires either a repeat split, another model, or a final 12-trace + BCa
confirmation packet for at most two finalists.

## Stop Conditions

Stop this branch if both candidates reproduce the baseline tail or introduce a
new negative tail worse than the baseline confirmation worst trace.
