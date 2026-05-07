# SSQ-LR Reviewer Pack

- status: S1b state-heterogeneity signal alive; S2 recipe currently fails
- current decision: weakened after held-out S2 scouts
- camera-readiness: not submittable as a method paper until S2/S3 have real
  passing evidence
- gate ladder: S1--S3 remain the only promotion path; resource-limited S2
  scouts are not current evidence for promotion.

## Paper Link

- Draft PDF: `experimental/ssq_lr/paper/ssq_lr_colm2026.pdf`
- Draft TeX: `experimental/ssq_lr/paper/ssq_lr_colm2026.tex`
- Preregistration:
  `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md`

## Current Claim

SSQ-LR asks whether recurrent SSM state in hybrid reasoners can be quantized
below FP16 without quality loss. The current artifacts show non-post-hoc S1b
state heterogeneity, but they do not claim that a usable state-quantization
recipe exists. The held-out S2 scouts currently weaken the method branch.

## Promotion Ladder Boundary

S1 has a real trace packet builder/checker, and S2/S3 have follow-up contract
checks in `experimental/shared/followup_gate_contracts.py`. Resource-limited S2
scouts are current evidence for weakening the branch, not for promotion. S1b
does not authorize a quantization recipe, byte-savings claim, or transfer claim.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | S1b uses a held-out 12-prompt local surface. S2 uses short continuation replay; that is useful as a Mac filter but still not task accuracy or native GPU evidence. | S2 weakened. |
| Ablations | BF16 no-op, INT8, FP8-style, MXFP4-style, INT3, random same-L2 noise, scale shuffle, and byte accounting are present in the S2 scouts. | Adequate for Mac demotion; not enough for promotion. |
| Correctness | The packet checker now requires prefill_end, 2k_or_end, 8k_or_end, and final_minus_128 buckets for every prompt/layer pair, at least 12 prompts unless resource-limited, matching model IDs, BF16 controls, SSM/Mamba layer-kind and recurrent-state tensor-kind labels, finite row fields, prompt hash provenance, architecture-map hash provenance tied to the claimed model, a `trace_plan_path` whose file hash matches the registered `trace_plan_hash`, saved tensor artifact provenance for every referenced tensor, and recomputed S1 `summary.json` gate aggregates with prompt-level lower bounds plus Holm-corrected distribution tests. Non-rehearsal packets must also include the copied `.pt` state tensors; the checker reloads them and recomputes max-abs, RMS, std, kurtosis, and outlier-mass row metrics. Distribution-only significance also needs a 1.25x per-layer effect-size floor for counted layer/metric tests. The shared builder now automatically marks resource-limited packets non-promotable. S2/S3 follow-up contracts additionally reject missing recipe IDs, byte accounting, BF16 no-op drift, same-byte controls, paired CIs, frozen recipe/source hashes, and retuned transfer rows. | Artifact path is hardened. |
| Reproducibility | Synthetic S1 emits a real-schema rehearsal packet; S1b and S2 scouts are reproducible from cached Granite Tiny weights with repo-local commands. The S2 follow-up contract now records both accuracy and continuation-NLL quality bounds. | Strong enough to weaken S2; not positive evidence. |
| Novelty | The wedge is sub-FP16 recurrent state for hybrid reasoners, not weight-only or KV-cache quantization. | Plausible only if real S2/S3 pass. |
| Camera-readiness | The draft is a preregistration shell. It needs real S1/S2/S3 tables before submission as a method paper; current S2 tables are failing held-out scouts. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic S1 schema rehearsal | 288 rows, 12 prompts, 6 recurrent layers, 4 buckets; passes `--mode real --project ssq_lr` as `SCHEMA_REHEARSAL_NOT_PROMOTABLE...` | validates checker path only |
| held-out S1b | 192 saved-tensor rows, 12 held-out prompts, layers `0`, `12`, `18`, `30`; primary layers `0`, `12`, `30` pass with selected ratio `2.459`, CI low `1.861`, Holm p-min `2.78e-05` | non-post-hoc heterogeneity evidence |
| S2 MXFP4 scouts | MXFP4 preserves BF16 argmax on short continuation replay but reaches only `3.765x`--`3.938x` after scale bytes | fails byte gate |
| S2 INT3 scouts | 4-prompt block-256 scout passes (`5.224x`, zero argmax delta), but 12-prompt block-64 and block-256 scouts fail because INT3 loses argmax fidelity on held-out prompts | fails held-out quality gate |
| architecture provenance | shared config-derived hashes exist for live hybrid targets | packet provenance ready |
| trace collection plan | `experimental/shared/results/hybrid_trace_plan_20260507/ssq_lr_trace_plan.jsonl` enumerates 5,184 required S1 capture rows | execution checklist only |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing buckets, incomplete prompt/layer matrices, stale summary fields, too few prompts, promotable resource-limited decisions, mismatched model IDs, missing controls, missing trace-plan hash pinning, missing tensor artifacts, tensor hash mismatches, and row metrics that do not recompute from saved state tensors | ready for real S1 |
| follow-up contract checker | `experimental.shared.followup_gate_contracts --gate ssq_lr_s2/ssq_lr_s3` enforces recipe/byte/CI/no-retuning fields before later evidence can be cited | contract ready, no model rows |

## Reviewer Risks

- S1b heterogeneity is real on a resource-limited held-out surface, but S2 has
  no passing held-out recipe.
- Current S2 continuation replay is a short Mac filter, not task accuracy.
- A sub-FP16 state recipe may need per-model retuning, which would kill the
  cross-model claim.
- Byte savings must include scale and metadata overhead, not just nominal dtype.

## Next Exact Gate

Only continue SSQ-LR if there is a new frozen sub-4-bit recipe or native
packed-state rationale that can plausibly clear S2 with paired quality bounds.
Do not send current MXFP4 or INT3 scouts to GPU as a positive method claim.
