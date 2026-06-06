# Response Plan

## Top objections and edits made

| Objection | Edit made |
| --- | --- |
| LatentWire reads like a failed positive search rather than a contribution | Reframed the draft around a bounded-negative falsification ladder and added a claim-boundary box. |
| Exact-discrete result could be overclaimed as a formal theorem | Added the 160-unique-example caveat, called it operational support, and kept the future continuous/privacy lanes open. |
| L-IB1 could be mistaken for a privacy-positive | Kept L-IB1 as `KILL_UTILITY_IS_IDENTITY`; added utility/leakage figure and table. |
| Gold-aware oracle ceilings could contaminate the LatentWire story | Added an explicit rejected-oracle row in the ladder and provenance. |
| Channel-Set lacks a confirmed positive method | Reframed the paper as measurement/regime only and kept C-A1 parked. |
| Channel-Set figures were missing | Added strict set-leaving, threshold, within-set shuffle, cached screen, sentinel-status, and defense-blocker figures. |
| Channel-Set needed a drift-to-loss link | Added a simple protected/unprotected per-channel error model and scoped the remaining error attribution as the parked C-A1 question. |
| Channel-Set needed stronger significance framing | Added the OSC token-persistence tension and clarified that trace-level drift is the relevant long-reasoning granularity. |
| LatentWire weak-signal regime could weaken the headline null | Added the explicit rebuttal using source+target upper bound and receiver-conditioning information. |
| LatentWire exact-discrete result could look too empirical | Added the a-priori discrete-content-as-visible-code argument while retaining the operational/sample caveat. |
| Numeric claims were not traceable enough | Added provenance tables for both papers and validators for broken figure refs/package guardrails. |

## Unresolved before camera-ready

- Venue LaTeX sources and PDFs are now generated in `paper/latentwire/main.tex` and `paper/channel_set/main.tex`.
- C-A1 remains separate optional GPU work and does not gate these papers.
