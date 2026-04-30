# Source-Private Systems Comparison Table

- date: `2026-04-29`
- status: reviewer-facing systems and baseline comparison artifact
- artifact root: `results/source_private_systems_comparison_table_20260429/`
- scale rung: artifact consolidation / systems-frontier comparison on Mac CPU

## Purpose

This table prevents the systems story from becoming an overclaim. It separates:

- headline source-private packet rows;
- same-surface, same-byte text and random controls;
- scalar/QJL source-coding comparators on their own control-clean surface;
- endpoint text-rate rows;
- KV/TurboQuant/KIVI/QJL byte-floor accounting rows.

The claim is not that source-private packets are better KV-cache compression.
The claim is that they occupy a far-left-rate task-communication point with
strict source-destroying controls, while KV/cache methods are systems neighbors
with different access assumptions.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_systems_comparison_table.py \
  --output-dir results/source_private_systems_comparison_table_20260429
```

## Headline

- headline learned pass rows: `3`
- headline learned minimum delta vs target: `+0.625`
- same-surface structured-text matched-byte max accuracy: `0.250`
- same-surface structured-text matched-byte max delta vs target: `+0.000`
- scalar source-code comparator accuracy: `1.000`
- QJL-style residual comparator accuracy: `1.000`
- minimum endpoint non-packet QJL 1-bit byte ratio vs packet: `10752.0x`

## Interpretation

The learned synonym-dictionary packet remains a scoped positive method:

- `4` bytes beats target and same-byte text on synonym-stress rows;
- same-byte structured text, random bytes, and answer-only text stay at target;
- held-out family-B rows are included as a negative boundary;
- scalar and QJL comparators are strong source-coding baselines, but they live
  on a different feature/slot surface and should not be used as direct evidence
  that the synonym receiver generalizes;
- endpoint KV/TurboQuant/KIVI rows are byte-floor accounting only, not kernel
  implementations or production GPU latency claims.

## Reviewer-Facing Claim Boundary

Fair:

> Source-private packets give a far-left-rate task communication point under
> strict source controls, and the artifact compares this point against
> matched-byte text, structured text at higher rates, scalar/QJL source-coding
> comparators, and KV byte-floor accounting.

Not fair:

> The packet method beats TurboQuant, KIVI, or QJL as general KV-cache
> compression.

## Next Exact Gate

Implement an anchor-relative sparse innovation receiver for the held-out
family-B split. The decisive pass condition remains: calibration sees only
family-A/synonym-stress surfaces, evaluation uses held-out family-B, exact
transformed phrase overlap is `0`, source-destroying controls stay within
target + `0.03`, and the packet beats target by at least `+0.10` to `+0.15`
with paired uncertainty.
