# OutlierMigrate Presubmission Checklist

Status: not camera-ready candidate yet. Phase 2 Nemotron-3 partial validation
and a fresh paper/reviewer-pack update are still required before this checklist
can be marked complete.

## Anonymization

- [ ] No author names in `outlier_migrate_colm2026.tex`.
- [ ] No acknowledgments in the review draft.
- [ ] No GitHub URLs or identity-revealing repository links in the paper.
- [ ] Artifact paths are local repo paths only.

## Format

- [ ] Confirm final venue page limit before submission.
- [ ] Rebuild `outlier_migrate_colm2026.pdf` with `build.sh`.
- [ ] Tables use `booktabs` style and contain no raster screenshots of numbers.
- [ ] Figures, if added, are PDF/vector where possible.

## Claims And Evidence

- [ ] Every numerical claim cites an exact artifact path with a passing
  `artifact_check.json`.
- [ ] Phase 0 and Phase 1 report the strict set-leaving, within-set shuffling,
  and original migration-fraction readouts.
- [ ] Phase 2 language matches the checker-backed outcome exactly.
- [ ] Qwen3.6 and Kimi Linear are not described as measured unless a passing
  result packet exists.
- [ ] No claim says this is the first dynamic-outlier finding in Mamba.

## Citations

- [ ] Verify every bibliography entry against a primary source.
- [ ] Confirm QMamba, OuroMamba, MambaQuant, Mamba-PTQ, Quamba, Kimi Linear,
  Qwen3.6, LLM.int8, SmoothQuant, AWQ, QuaRot, KVQuant, and BlockDialect claims.
- [ ] Remove or soften any citation whose abstract/body does not support the
  paper sentence that cites it.

## Reproducibility

- [ ] Reproducibility Statement names commit SHA, GPU model, runtime, prompt
  set SHA, model snapshot SHA, commands, seeds, and artifact paths.
- [ ] `reviewer_pack.md` resolves every artifact path.
- [ ] Fresh-shell reproduction audit can recompute the headline metric from the
  result packet.

## Limitations And Ethics

- [ ] Limitations distinguish Granite-family evidence from cross-architecture
  validation.
- [ ] Limitations acknowledge deterministic AIME slice, bootstrap-over-trace
  uncertainty, and no end-to-end quantization speedup yet.
- [ ] Ethics Statement says no human-subject data and notes model-evaluation
  limitations.
