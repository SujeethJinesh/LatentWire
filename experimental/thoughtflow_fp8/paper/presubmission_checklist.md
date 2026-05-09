# ThoughtFlow-FP8 Presubmission Checklist

Status: buildable falsification-methodology candidate, not camera-ready final.
The paper body is copyedit-only unless the human author explicitly reopens it;
use this checklist for review readiness.

## Anonymization

- [ ] No author names in `thoughtflow_fp8_colm2026.tex`.
- [ ] No acknowledgments in the review draft.
- [ ] No identity-revealing GitHub URLs.
- [ ] Artifact paths are local repo paths only.

## Format

- [ ] Confirm final venue page limit before submission.
- [ ] Rebuild `thoughtflow_fp8_colm2026.pdf`.
- [ ] Tables use `booktabs` style and contain no raster screenshots of numbers.
- [ ] Figures, if any are added, are PDF/vector where possible.

## Claims And Evidence

- [ ] The paper stays framed as falsification methodology, not a positive FP8
  systems result.
- [ ] No throughput, latency, CUDA, or production-serving claim is made without
  a matching artifact.
- [ ] Every numerical claim resolves through `reviewer_pack.md` to an existing
  artifact path.
- [ ] Committee-review caveats about proxy baselines and keep-rate comparisons
  remain visible.

## Citations

- [ ] Verify every bibliography entry against a primary source.
- [ ] Recheck ThinKV, LongFlow, R-KV, and any 2025/2026 citation before final
  submission.
- [ ] Remove or soften any citation whose abstract/body does not support the
  paper sentence that cites it.

## Reproducibility

- [ ] Reproducibility Statement names commit SHA, hardware/software context,
  commands, seeds, and artifact paths.
- [ ] `reviewer_pack.md` resolves every referenced artifact path.
- [ ] Fresh-shell reproduction audit can follow the reviewer pack without
  hidden state.

## Limitations And Ethics

- [ ] Limitations emphasize that this is not a camera-ready positive-method
  paper and not an MLSys systems paper.
- [ ] Limitations acknowledge repo-local timestamping, proxy baseline scope,
  and diagnostic rather than definitive promotion/failure language.
- [ ] Ethics Statement says no human-subject data and notes model-evaluation
  limitations.
