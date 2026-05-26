# Audit Summary

Date: 2026-05-26

This summary aggregates the flight-window audit outputs after the paper polish
pass. Severity labels follow the sprint protocol: CRITICAL must be fixed before
sprint completion, SUBSTANTIAL should be fixed or explicitly deferred, and
MINOR can remain for post-submission polish.

| Audit | Findings | Severity | Status |
|---|---|---|---|
| Desk rejection avoidance | Release verification leaked an identifying GitHub URL. | CRITICAL | Fixed by replacing clone commands with anonymous artifact placeholders. |
| Desk rejection avoidance | Submission lacked an explicit LLM-use disclosure despite substantive LLM-assisted drafting, auditing, and summarization. | CRITICAL | Fixed with an LLM Usage Statement in the paper. |
| Desk rejection avoidance | Dual-workshop policy and exact CBW compliance remain organizer/human decisions. | SUBSTANTIAL | Deferred to human submission decision; paper main body now fits the 8-page target. |
| Citation and novelty | No fabricated or missing load-bearing citation found. Quamba2, DecDEC, ParoQuant, and SLQ wording was too strong in places. | SUBSTANTIAL | Fixed by softening to scoped comparison/extension language. |
| Release code correctness | Paper/release overclaimed full GPU reproduction inside `release/`; current package provides FAST_VERIFY claim replay plus minimal public interfaces. | CRITICAL | Fixed in paper, README, and verification docs. |
| Release code reproducibility | FAST_VERIFY uses frozen paper-claim values; full model-inference reruns remain in archived experimental packet runners. | SUBSTANTIAL | Documented as an explicit limitation rather than hidden. |
| Release code hardcoding/determinism | Method helpers and configs are intentionally minimal and do not implement the full archived pipeline. | SUBSTANTIAL | Deferred; acceptable for first-complete draft because the release no longer overclaims full rerun coverage. |
| Results/statistics/figures | Main text omitted the Nemotron static top-10 control row and appendix omitted Nemotron top-5/static rows. | SUBSTANTIAL | Fixed in main and appendix tables. |
| Results/statistics/figures | Paper counted seven channel-set families while listing eight including static union. | SUBSTANTIAL | Fixed to eight throughout main text. |
| Results/statistics/figures | KL wording implied formal model selection stronger than the tested residual comparison. | SUBSTANTIAL | Fixed to "lowest residuals among tested functional forms." |
| Results/statistics/figures | No-gap trace handling differed between early static-union gates and later method packets. | SUBSTANTIAL | Clarified in Discussion. |
| Results/statistics/figures | Figure generator hardcodes displayed recovery/component values. | SUBSTANTIAL | Added a figure source manifest that maps each plot to the result artifacts used for the displayed values. |
| Baseline release checks | `pytest release/tests`, `verify_environment.py`, and `reproduce_all.sh --fast-verify` passed locally before the final audit edits. | MINOR | Re-run required after final commit. |

## Critical Findings

All CRITICAL findings identified by the current audit set have been addressed in
the source tree. No audit found a headline numerical mismatch or evidence that
the Nemotron M11b PASS was an artifact.

## Substantial Deferred Items

- The release package is a minimal public interface with FAST_VERIFY replay, not
  a fully rewritten GPU reproduction of the archived experimental runners.
- Exact workshop targeting and any organizer clarification about dual submission
  remain human decisions.
- Some figure values are mirrored from result packets through a manifest rather
  than loaded directly from every raw cache. This is documented and should be
  acceptable for the submission draft, but a future release polish pass should
  make all plotting fully artifact-driven.
