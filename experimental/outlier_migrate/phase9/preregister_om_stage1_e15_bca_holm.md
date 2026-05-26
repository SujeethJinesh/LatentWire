# Stage 1 E15: BCa Bootstrap and Holm-Bonferroni Correction

## Purpose

E15 is a post-processing-only audit of the method table intervals and
multiple-comparison status. It does not run model inference and does not inspect
new outcomes before the procedure is fixed.

## Inputs

The runner consumes existing result packets that already contain per-trace
recovery values in either:

- `per_trace_metrics.json`, under `traces[*].recoveries`; or
- `bootstrap_ci.json`, under `results_by_regime[*].per_trace_recovery_included`
  when the packet explicitly preserved the included per-trace values.

Default input packets are the validated method-table packets available before
E15:

- M2 position-conditional switching
- M10 position-binned scales
- M11 EMA
- M11b Granite-Small budget sweep
- M11b Nemotron salvage packet
- M18 activation+K coupling
- DecDEC proxy
- M26 stable core
- ParoQuant Granite-Small

Stage 1 E3 may be added explicitly with `--packet LABEL=PATH` after it lands.

## Statistic

For each `(packet, regime)` with exact per-trace recovery values:

1. Recompute the median recovery.
2. Compute a two-sided 95% BCa bootstrap interval for the median.
3. Compute a one-sided bootstrap p-value for `median recovery > 0`.
4. Apply Holm-Bonferroni correction across all `(packet, regime)` tests in the
   method table family.

The bootstrap seed is fixed at `20260615` by default and is configurable from
the command line. The default bootstrap sample count is 10,000.

## Missing-Input Rule

E15 must fail loudly if an explicitly requested packet or regime lacks exact
per-trace recovery values. It must not fabricate values from summary medians or
from previously reported confidence intervals.

## Outputs

The runner writes one JSON artifact, `e15_bca_holm.json`, containing:

- input packet provenance;
- per-regime per-trace source used;
- recomputed medians;
- BCa intervals;
- one-sided bootstrap p-values;
- Holm-Bonferroni adjusted p-values and rejection flags;
- any missing-input diagnostics.

The checker validates the JSON artifact and fails if any missing input remains.
