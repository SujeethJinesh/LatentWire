# Final Positive-Method Decision: Three Untested Branches

Created: 2026-05-29T03:40:22Z
Git commit context: `f65e36879a95ea48609371ae8471e21536305bb0`

## Branch Outcomes

| Branch | Decision | Evidence | Next action |
|---|---|---|---|
| Complete M-SURFACE SSM/B/C internals | `KILL_MSURFACE_NO_LOWER_INTERNAL_SURFACE` | Complete Granite internals run: post-block mean leaving 0.546; SSM input 0.484; SSM C 0.486; SSM B 0.618; B/C concat 0.602; Mamba out-proj input 0.887; attention o-proj input 0.799. No internal surface is <0.30-0.40 or >=0.15 below post-block. | Stop M-SURFACE as positive method; add as stronger negative result. |
| Restricted residual KLLOOK | `NO_RUN_EXCEEDS_ONE_HOUR_RUNNER_GATE` | Correct runner is not a <=1 hour build. Existing M-KLLOOK is original-basis channel selection, not rotated residual correction. K-RES audit already supports killing the current proxy. | NO_RUN; do not reopen selector search without a new preregistered runner. |
| Falcon BranchRot / M-BRANCH | `DEFERRED_TO_FUTURE_WORK_NOT_RUN_THIS_PASS` | BranchRot was explicitly deferred unless M-SURFACE promoted. M-SURFACE did not promote, and no branch-local cache exists. | Future work only. |

## Decision

All final bounded positive-method gates failed, were no-run by the hard runner gate, or were deferred by the revised priority order. Stop positive-method search.

Final paper path: `MECHANISM_REGIME_PAPER`.

## Required Paper Framing

- ParoQuant remains the strongest baseline, not our contribution.
- K-RES is elevated to a mechanism finding: residual energy is not aligned enough with loss benefit under the tested proxy.
- M-SURFACE is a stronger negative result: drift reaches internal SSM/B/C surfaces at meaningful magnitude; the slight reduction at SSM input/C is not enough for a positive method.
- BranchRot is future work, not a sprint claim.
