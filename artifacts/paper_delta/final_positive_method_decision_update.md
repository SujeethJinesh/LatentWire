# Paper Delta: Final Positive-Method Decision

Final path: `MECHANISM_REGIME_PAPER`.

Updates to integrate:

- K-RES top-8x32 is a mechanism finding, not merely a failed attempt: valid residual-energy columns worsened Granite tail recovery, so residual norm is not aligned enough with loss benefit under this proxy.
- Restricted residual KLLOOK is `NO_RUN_EXCEEDS_ONE_HOUR_RUNNER_GATE`; the repo's existing M-KLLOOK runner is original-basis channel selection and cannot be reused as a rotated-basis residual oracle.
- Complete Granite M-SURFACE internals kill surface-local protection for this sprint: post-block mean leaving 0.546, SSM input 0.484, SSM C 0.486, SSM B 0.618, B/C concat 0.602, Mamba out-proj input 0.887, attention o-proj input 0.799.
- Falcon BranchRot is future work, not a sprint claim, because it was deferred unless M-SURFACE promoted and no branch-local cache exists.
- Stop positive-method search and finalize the mechanism/regime paper.
