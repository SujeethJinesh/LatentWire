# Structural Audit Completion

Date: 2026-05-26

## Triggered Path

Phase 4B: targeted structural changes.

No full rewrite was triggered. The audit found no `STRUCTURAL_GAP` sections.
The current measurement-first structure was judged fundamentally sound for a
COLM workshop mechanism paper.

## What Changed

- Title changed from a stronger "Budget-Tuned Remedy" framing to
  `Channel-Set Drift in Long-Reasoning W4A16 LLMs: Mechanisms, Failed Remedies, and Budget-Dependent Signals`.
- Abstract now calls M11b the strongest budget-dependent channel-set signal
  rather than the strongest remedy.
- Contribution 4 was renamed to `Budget-dependent signal and scope`.
- Method section now groups the eight method classes by mechanism tested:
  static coverage, hard switching, smoothing, cross-tensor signal, reactive
  selection, larger-budget tracking, and stable cores.
- M11b/ParoQuant subsection heading now foregrounds baseline pressure:
  `Budget helps, but rotation remains the stronger Granite baseline`.
- Discussion now explicitly marks saturated branches and the live branch:
  rotation plus budget-aware protection.

## What Was Preserved

- All numerical claims and confidence intervals.
- All verified references and bibliography entries.
- The LLM Use Disclosure section.
- Ethics and reproducibility statements.
- Appendices and provenance mapping.
- Main measurement-first result order: drift, components, intervention ladder,
  budget/rotation pressure, KL mechanism.

## Verification

- Canonical paper rebuilt successfully with only TeX underfull warnings.
- Release paper mirror rebuilt successfully.
- Page count: 12 total pages for both PDFs.
- References start on page 7 for both PDFs, so main body remains within the
  8-page target.
- Citation check: no missing bibliography keys; 37 bibliography entries and 29
  cited keys in both canonical and release TeX.
- LLM Use Disclosure remains present in both canonical and release TeX.
- Main-body internal/sprint vocabulary check passed:
  `experimental/`, `vacation`, `vac12`, `swarm/`, `PASS_`, `KILL_`,
  `FAIL_INFRA`, `Step 9`, and `Phase N` do not appear before the bibliography.
- Prose regression keywords checked: `load-bearing`, `therefore`, `Thus`,
  `thus`, `Moreover`, `Furthermore`, `Additionally`, `Notably`, `Importantly`,
  `delve`, `leverage`, `harness`, `navigate`, and em-dash do not appear in the
  main body.

## Gate Outcome

The paper remains structurally aligned with the evidence. Stage 1 GPU work can
begin next under the raised 360-hour cap unless a newer instruction changes the
queue.
