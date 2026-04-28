# Source-Private Tool-Trace Figure Assets

- date: `2026-04-30`
- status: setup and rate-curve figure assets generated
- live branch: explicit source-private tool-trace packet handoff
- scale rung: paper asset generation

## Question

Can we turn the most important draft placeholders into concrete paper assets?

The skeptical draft review marked the rate curve as essential because
structured JSON/free-text relays become oracles at `32` bytes. This gate creates
the setup diagram and rate-curve data/figure before LaTeX conversion.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_tool_trace_figures.py \
  --output-dir results/source_private_tool_trace_latex_or_figures_20260430
```

## Artifacts

- `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.svg`
- `results/source_private_tool_trace_latex_or_figures_20260430/source_private_setup.pdf`
- `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.svg`
- `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.pdf`
- `results/source_private_tool_trace_latex_or_figures_20260430/rate_curve.csv`
- `results/source_private_tool_trace_latex_or_figures_20260430/manifest.json`

## Figure 1

`source_private_setup.svg` shows the source-private communication setup:

- public task and target state on the target side
- private tool trace on the source side
- rate-capped `REPAIR_DIAG` packet
- target-side decoder using `(X,T,M)`

## Figure 2

`rate_curve.svg` plots accuracy versus communicated bytes averaged over the
representative core seed `29` and held-out seed `30` deterministic reviewer-risk
surfaces.

Key behavior:

- compact packet is oracle at `2` bytes
- JSON and free-text relays stay at target-only through `16` bytes
- JSON and free-text relays become oracle at `32` bytes
- full diagnostic text is oracle at roughly `14` bytes
- full hidden-log relay is oracle at roughly `366-374` bytes

## Decision

Use these assets in the next LaTeX conversion gate. The rate curve should be
shown in the main paper, not an appendix, because it explains the structured
text relay baseline and prevents overclaiming the compact-packet result.

## Next Gate

`source_private_tool_trace_latex_20260430`:

- copy the markdown draft into an ICLR LaTeX source file
- include figure references for the two SVG assets
- add concrete citation placeholders / BibTeX entries
- add counts to the main table
