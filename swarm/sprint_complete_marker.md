# Sprint Complete Marker

Generated: 2026-05-25 UTC

## Final State

First-complete workshop draft, release FAST_VERIFY package, committee review,
audit reports, external collaboration export, final report, and the
submission-polish pass are present.

## Paper

Title: **Channel-Set Drift in Long-Reasoning W4A16 LLMs: Mechanisms and a
Budget-Tuned Remedy**

Primary contributions:

1. Four-model long-decode set-leaving characterization for W4A16 reasoning
   models, with the main body now compressed to the workshop page target.
2. Quamba2/DecDEC-aware scoped novelty: reasoning-scale horizons, hybrid and
   pure-Transformer model set, and reasoning workloads.
3. Mechanism findings: boundary discontinuities are harmful; smoothing alone is
   insufficient; budget tuning gives partial cross-model recovery; dense-grid KL
   weakens compound-error explanations in the measured Granite packet.
4. Baseline context: ParoQuant rotation recovers more than channel-set methods
   on Granite-Small.

## Committee Scores

| Round | Efficient Reasoning | Context Beyond Window | Average |
|---|---:|---:|---:|
| 3 | 7.0 | 7.0 | 7.0 |
| 4 | 7.5 | 8.0 | 7.75 |
| 5 | 7.5 | 8.0 | 7.75 |
| 6 | 7.5 | 8.0 | 7.75 |

Six-round cap reached; no new load-bearing critiques surfaced after round 3.

## Reproducibility

- Release package: `release/`
- Clean FAST_VERIFY verification: passed twice from GitHub clones.
- Local tests: `7 passed`.
- Full model-inference GPU reproduction: not implemented in `release/`.
  Scripts now fail explicitly outside dry-run/fast-verify to avoid overclaiming.

## Audits

All requested audit reports are in `docs/`. All CRITICAL findings are fixed.
Remaining substantial limitation is the lack of full GPU runners in `release/`.

## GPU Hours

Approximate cumulative GPU hours tracked in `swarm/state.json`: `278.643`.

## Autonomous Decisions For Human Review

- Accepted Path C static-1% Nemotron salvage and used the valid packet.
- Chose a concise mechanism-plus-remedy title and foregrounded the Nemotron
  top-10 M11b result while preserving Granite CI caution.
- Moved internal provenance details out of the main paper body and into
  appendix/release documentation.
- Stopped new GPU experiments after Nemotron and focused on integration,
  release, verification, and audits.
- Reclassified release reproducibility honestly as FAST_VERIFY-only.

## Open Questions

1. Is FAST_VERIFY plus archived packets sufficient for workshop artifact
   release, or should full GPU runners be implemented before submission?
2. Which workshop should be primary: Context Beyond Window or Efficient
   Reasoning?
3. For ICLR, should the next branch be rotation+budget composition, M18b
   cross-tensor top-5, or M11c budget sweep?

## Latest Commit

This marker is included in the final sprint documentation commit. Use
`git log -1` on `origin/main` for the exact pushed hash.
