# Desk-Rejection Gate

Date: 2026-05-29

Target: Efficient Reasoning @ COLM 2026, COLM format, 4--10 pages main text excluding references and appendices.

## Gate Results

| Gate | Status | Evidence / action |
|---|---|---|
| 0.1 Page limit | PASS | Rebuilt PDF has 17 total pages. Main text ends on page 10; references begin on page 11. A repetitive standalone conclusion was folded into the limitations/framing paragraph to keep main text within the 10-page cap without dropping claims. |
| 0.2 Anonymity | PASS_AFTER_FIX | Paper author block is anonymous. Active paper, `CLAIM_AUDIT.md`, `artifacts/systems_cost/cost_table.csv`, and the final supplementary pack were scanned for author names, GitHub handles, local repo names, and absolute identity-revealing paths. The final pack previously contained `/workspace/LatentWire` and cache paths; these were anonymized to placeholders such as `<REPO_ROOT>` and `<HF_CACHE>`, and the tarball was regenerated. |
| 0.3 COLM format | PASS | TeX uses `\usepackage[submission]{../../../colm/2026/colm2026_conference}` with anonymous author block, abstract, main sections, ethics, reproducibility, and LLM-use disclosure. |
| 0.4 Self-containment | PASS | Main text includes the load-bearing drift result, intervention table, rotation baseline result, decision protocol, negative DriftRot screens, M-SURFACE numbers, and systems-cost envelope. Appendices contain provenance and expanded tables only. |
| 0.5 LLM-use disclosure | PASS | `LLM Use Disclosure` remains present and explicit. It was not weakened in this pass. |

## Scans Performed

- PDF text extraction: no unresolved `??` markers found.
- Active bibliography: 32 cited entries, zero cited-missing, zero active orphaned entries.
- Supplementary tarball: original absolute local paths found, sanitized, refreshed with the final paper/hardening artifacts, and tarball regenerated.
- Final pack size after anonymization: 44 MiB.

No desk-rejection blocker remains in this gate.
