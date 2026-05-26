# SA-DESK Pre-Submission Desk-Rejection Audit

Date: 2026-05-26

Scope: polished paper source/PDF, release-facing package state, `swarm/final_report.md`,
and current audit docs. This audit intentionally does not edit paper, release,
swarm, or experiment files.

Current paper readiness status: not submission-ready until the CRITICAL items
below are fixed or explicitly ruled out. Estimated distance to workshop
submission readiness: one focused compliance pass, mostly anonymization and
disclosure cleanup. Current story: channel-set drift is robust in long-reasoning
W4A16 traces; hard switches fail; budget-tuned EMA is a partial remedy; ParoQuant
is a stronger Granite rotation baseline. Exact blocker: release-facing
anonymization, LLM-use disclosure, and a paper/release reproducibility mismatch.

## External Policy Sources Checked

- COLM 2026 CFP: double-blind submissions, no acknowledgments or identifying
  links such as GitHub, 9-page main-text limit with unlimited citations, optional
  ethics statement outside the page limit, and COLM 2026's modified ICLR-style
  LLM-use policy: <https://colmweb.org/cfp.html>.
- COLM 2026 submission instructions: official 2026 template link, double
  submission reminders, and LLM-use disclosure examples requiring disclosure for
  originating research ideas, writing original paper content including
  references, generating data or plots, or evaluation:
  <https://colmweb.org/submission-instructions.html>.
- COLM author guide: supplementary code/data may be supplied but must be
  anonymized, self-contained, or reachable through an anonymous URL:
  <https://colmweb.org/AuthorGuide.html>.
- Efficient Reasoning @ COLM 2026: requires COLM format, 4-10 pages of main
  text excluding references, self-contained main text, double-blind anonymized
  submissions, and anonymized supplementary/linked code:
  <https://wdlctc.github.io/efficient-reasoning-2026/>.
- OpenReview COLM venue list confirms a COLM 2026 Workshop CBW venue, but I did
  not find a public CBW CFP/page-limit/dual-submission policy page:
  <https://openreview.net/venue?id=colmweb.org%2FCOLM>.

## Findings

| Severity | Finding | Evidence | Concrete fix |
|---|---|---|---|
| CRITICAL | Release-facing docs leak an identifying GitHub repository URL. If `release/` or release docs are submitted or linked as supplementary material, this violates double-blind anonymization. | `release/VERIFICATION.md:18` and `release/VERIFICATION.md:147` clone `https://github.com/SujeethJinesh/LatentWire.git`. COLM and Efficient Reasoning both require linked/supplementary code to be anonymized. | Replace public owner URLs with an anonymous artifact URL or omit clone-origin lines from the anonymous package. Re-run verification from the anonymous artifact and scrub all release docs for owner/repo identity before submission. |
| CRITICAL | No LLM-use disclosure is present, but the project process appears to have used LLM/agent assistance beyond minor grammar or formatting. COLM 2026 requires disclosure for substantive LLM use. | The paper has Ethics and Reproducibility statements at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:399` and `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:402`, but no LLM Usage Statement. Current repo instructions and audit workflow indicate agent-written audit/code/paper-facing work. | Add an anonymized LLM Usage Statement in the paper or required OpenReview field. It should distinguish writing/editing assistance, code generation, plot/data generation, evaluation/audit use, and human responsibility for all claims. |
| SUBSTANTIAL | The paper overclaims release reproducibility and points to a non-existent script path. This is a final unsupported-claim risk, not just a packaging nit. | Paper says the release includes `release/scripts/reproduce_all.sh` and that full-fidelity reproduction reruns every paper-referenced method on a GPU at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:403`. Actual README uses `src/scripts/reproduce_all.sh` at `release/README.md:38`; release docs state full model execution is not implemented at `release/README.md:41`, `release/docs/reproducing_results.md:3`, and `release/docs/architecture_overview.md:21`. Final report repeats this limitation at `swarm/final_report.md:81`. | Rewrite the paper reproducibility statement to match the release: FAST_VERIFY replays frozen claim values; full GPU reruns live in archived experimental packet runners and are not implemented in `release/`. Fix the script path to `release/src/scripts/reproduce_all.sh` or the command run from `release/`. |
| SUBSTANTIAL | Dual-workshop submission intent is ambiguous. Efficient Reasoning explicitly allows ongoing/under-review work if no other policy is breached, but CBW policy could not be verified from public sources. | Final report lists both workshop acceptance estimates at `swarm/final_report.md:130` and `swarm/final_report.md:131`, then says "Submit the current draft to the workshop" at `swarm/final_report.md:138`. Efficient Reasoning policy is permissive, but no public CBW policy was found beyond the OpenReview venue listing. | Either submit to one workshop only, or obtain written policy confirmation from both organizers that the same anonymous PDF can be simultaneously submitted to both non-archival COLM workshops. Record the answer before upload. |
| SUBSTANTIAL | CBW page-count compliance depends on whether CBW counts main text or total PDF. Under the user-provided CBW limit of <=8 pages for a long paper, the main body appears to fit, but the total PDF does not. | pypdf inspection: `outlier_migrate_colm2026.pdf` has 10 total pages; references start on PDF page 7; Ethics/Reproducibility and appendix/provenance continue through page 10. Source appendix starts at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:424`. | If CBW counts total PDF pages, create a CBW-specific anonymous PDF with appendix removed or moved to supplementary material and total pages <=8. If CBW counts main text excluding references/appendix, document that policy source before submission. |
| SUBSTANTIAL | CBW scope fit is weaker than Efficient Reasoning unless the workshop explicitly welcomes long-decode/activation-drift work as context-beyond-window methodology. | The paper's core claim is W4A16 channel-set drift and quantization remedies (`experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:24`, `:62`, `:131`). It uses long decode horizons up to 20K tokens (`:66`, `:74`) but does not evaluate retrieval, memory extension, context-window extrapolation, or beyond-window task success. | For CBW, add a cover/abstract framing that the contribution is a measurement of state-summary drift during long generation, not a context-extension method. Prefer Efficient Reasoning as the primary target if only one workshop is chosen. |
| MINOR | COLM 2026 template compliance looks acceptable in the current paper source. | Source uses `\documentclass{article}` and `\usepackage[submission]{../../../colm/2026/colm2026_conference}` at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:1` and `:3`. Local COLM template warns authors to use current style files and has a 9-page main-text limit; no `geometry` override was found in the paper source. | No template edit required, but build the final submission from a clean tree and inspect the log for style/package warnings before upload. |
| MINOR | Paper author block and PDF metadata do not show an immediate author identity leak. | Source author is anonymous at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:13`. pypdf metadata inspection found 10 pages and metadata only for Creator (`LaTeX with hyperref`), Producer (`xdvipdfmx`), and CreationDate; no Author or Title metadata fields were present. | Keep checking the uploaded PDF after any rebuild. Some PDF upload/rebuild paths can add author metadata. |
| MINOR | The release README itself is mostly anonymous, but the broader release package is not yet anonymous because of verification docs. | `release/README.md:56` through `release/README.md:60` uses anonymous BibTeX author text. The identifying leak is in `release/VERIFICATION.md`, not README. | Audit the entire supplementary zip, not just README, before upload: `rg -n "github.com|LatentWire|Sujeeth|/workspace|OpenAI|Claude|Codex" release docs experimental/outlier_migrate/paper`. |
| MINOR | Cost-forecast citations are non-primary and easy reviewer targets, though not a desk-rejection issue. | Intro cites inference-cost forecasts at `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:28`; bibliography includes TechCrunch/Gartner/MarketsandMarkets at `:171`, `:178`, `:185`, and `:191`. | If space permits, replace or downweight commercial/press motivation with peer-reviewed systems evidence, or remove market-forecast claims because they are not load-bearing. |

## Page And Format Readout

- PDF page count: 10 total pages by pypdf.
- Main-body fit: references begin on PDF page 7, so the main paper appears to fit
  COLM's 9-page main-text cap and Efficient Reasoning's 4-10 page main-text cap.
- CBW fit: likely OK only if CBW excludes references and appendix from the
  <=8-page limit. Not verified from public policy.
- Main-text self-containment: mostly OK for Efficient Reasoning. The method,
  results, limitations, and conclusion are in the main body; provenance details
  are appendix-only.

## Anonymization Checklist Result

- Author names/affiliations in paper: pass (`Anonymous authors`).
- Acknowledgments: pass; no acknowledgment section found.
- PDF metadata: pass on current PDF; no Author/Title metadata found.
- Identifying paper URLs/repos: pass in paper main text; no GitHub URL found.
- Identifying supplementary/release docs: fail because of `release/VERIFICATION.md`.
- Self-citations: no obvious self-citation pattern found from the audited source,
  but this cannot be fully certified without the real author list.

## Highest-Priority Fix Order

1. Scrub or replace the identifying GitHub clone URLs in release-facing
   verification material and regenerate anonymous verification evidence.
2. Add the required LLM-use disclosure if any substantive LLM/agent assistance was
   used for research ideas, original text, references, code, plots, data, or
   evaluation.
3. Fix the reproducibility statement so it no longer claims full GPU reruns from
   `release/`, and correct the script path.
4. Decide single-workshop versus dual-workshop submission after confirming CBW
   policy and page-count semantics.

## Bottom Line

The current paper PDF is close on template and main-body length, and the paper's
scientific hedging is substantially better than earlier audit notes. The
desk-rejection risk is now mostly procedural: anonymized release packaging,
mandatory LLM-use disclosure, and venue-specific dual/page policy. The strongest
unsupported-claim risk is the paper's full-fidelity release reproduction sentence,
which directly conflicts with the release docs.
