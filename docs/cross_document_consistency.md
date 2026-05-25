# Cross-Document Consistency Audit

Date: 2026-05-25

Scope: paper, release docs, verification log, collaboration state, and sprint notes.

## Findings

| Severity | Finding | Status |
|---|---|---|
| CRITICAL | Release docs previously implied full GPU reproduction through the same scripts, but scripts only replayed frozen claims. | Fixed: docs now state FAST_VERIFY only and scripts fail outside supported modes. |
| SUBSTANTIAL | `swarm/external_collaboration_state.md` predates the final Nemotron Path C result and paper integration. | Pending final documentation update. |
| SUBSTANTIAL | Final report and sprint marker are not yet written. | Pending final documentation stage. |
| MINOR | Older progress notes contain superseded provisional Nemotron values. | Acceptable because they are timestamped progress notes; final docs should point to the Path C packet. |

## Consistency Checks

- Paper and release agree on headline title and scoped W4A16/small-model framing.
- Paper and release agree on M11b Granite, M11b Nemotron, M26, ParoQuant, KL,
  FFT, and set-leaving headline values.
- Release verification now agrees with README: fast verification is verified;
  full model-inference reproduction is not implemented in `release/`.
- Committee scores are documented through round 6.

## Remaining Documentation Work

Before sprint-complete:

1. update `swarm/external_collaboration_state.md` to final Path C state;
2. write `swarm/final_report.md`;
3. write `swarm/sprint_complete_marker.md`;
4. ensure final docs do not claim full GPU release reproduction.
