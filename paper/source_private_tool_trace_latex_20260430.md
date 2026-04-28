# Source-Private Tool-Trace LaTeX Draft

- date: `2026-04-30`
- status: ICLR LaTeX source created and compiled
- live branch: explicit source-private tool-trace packet handoff
- scale rung: paper-source conversion

## Outputs

- `paper/iclr2026/source_private_tool_trace.tex`
- `paper/iclr2026/source_private_tool_trace.bib`
- `paper/iclr2026/source_private_tool_trace.pdf`
- `results/source_private_tool_trace_latex_compile_20260430/manifest.md`

## Contents

The LaTeX draft includes:

- abstract
- introduction
- problem setup
- benchmark
- method
- baselines and controls
- main result table with counts and rates
- threat-model control table
- setup and rate-curve figure references
- target-decoder smoke ablation
- interpretability
- related work with concrete BibTeX entries
- limitations
- conclusion
- appendix claim boundary and artifacts

## Compile Command

```bash
cd paper/iclr2026
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

## Compile Result

- output: `paper/iclr2026/source_private_tool_trace.pdf`
- pages: `7`
- size: `225146` bytes
- figure format: PDF assets referenced from
  `results/source_private_tool_trace_latex_or_figures_20260430/`
- log audit: no overfull boxes, undefined references, or citation warnings found
  by `rg -n "Overfull|undefined|Citation|LaTeX Warning|Package natbib Warning|Warning--"`
- remaining warnings: underfull page-fill boxes only

## Next Gate

`source_private_tool_trace_final_review_20260430`:

- run the final skeptical reviewer pass on the compiled PDF/source
- decide whether to scale the target-decoder smoke or keep it as an ablation
- polish tables, figure captions, and related-work framing for submission
