# SSQ-LR Reviewer Pack

- status: S1b state-heterogeneity signal alive; layer-selective S2b candidate alive
- current decision: weakened; only layers `0,30` survive longer-window S2b replay
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
state heterogeneity, and the longer-window S2b localization replay identifies a
layer-selective candidate. They do not yet claim that a usable
state-quantization recipe exists because S3 transfer and verbosity/length drift
are untested.

## Promotion Ladder Boundary

S1 has a real trace packet builder/checker, and S2/S3 have follow-up contract
checks in `experimental/shared/followup_gate_contracts.py`. Resource-limited S2
scouts are current evidence for narrowing the recipe to layers `0,30`, not for
promotion. S1b/S2b do not authorize a quantization recipe, byte-savings claim,
or transfer claim.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | S1b uses a held-out 12-prompt local surface. S2b uses continuation replay; the three-layer recipe fails the longer-window replay, but layers `0,30` pass when localized. Future promotion must add no-retuning transfer plus verbosity/length drift, not only BF16-argmax fidelity. | Layer-selective S2b candidate alive, not promotable. |
| Ablations | BF16 no-op, INT8, FP8-style, MXFP4-style, INT3, random same-L2 noise, scale shuffle, and byte accounting are present in the S2 scouts. A promotable S2 must also compare against FP16 stochastic-rounding SSM cache and INT16/block-scaled SSM cache baselines from Nemotron-style deployment. | Adequate for Mac demotion; not enough for promotion. |
| Correctness | The packet checker now requires prefill_end, 2k_or_end, 8k_or_end, and final_minus_128 buckets for every prompt/layer pair, at least 12 prompts unless resource-limited, matching model IDs, BF16 controls, SSM/Mamba layer-kind and recurrent-state tensor-kind labels, finite row fields, prompt hash provenance, architecture-map hash provenance tied to the claimed model, a `trace_plan_path` whose file hash matches the registered `trace_plan_hash`, saved tensor artifact provenance for every referenced tensor, and recomputed S1 `summary.json` gate aggregates with prompt-level lower bounds plus Holm-corrected distribution tests. Non-rehearsal packets must also include the copied `.pt` state tensors; the checker reloads them and recomputes max-abs, RMS, std, kurtosis, and outlier-mass row metrics. Distribution-only significance also needs a 1.25x per-layer effect-size floor for counted layer/metric tests. The shared builder now automatically marks resource-limited packets non-promotable. S2/S3 follow-up contracts additionally reject missing recipe IDs, byte accounting, BF16 no-op drift, same-byte controls, paired CIs, frozen recipe/source hashes, and retuned transfer rows. | Artifact path is hardened. |
| Reproducibility | Synthetic S1 emits a real-schema rehearsal packet; S1b and S2 scouts are reproducible from cached Granite Tiny weights with repo-local commands. The S2 follow-up contract records both accuracy and continuation-NLL quality bounds. | Strong enough to stop the mixed recipe; not positive evidence. |
| Novelty | The wedge is sub-FP16 recurrent state for hybrid reasoners, not weight-only or KV-cache quantization. Nemotron-style deployment already treats Mamba SSM cache precision as a distinct issue and uses FP16 stochastic rounding, so SSQ-LR must beat that baseline rather than merely identify the cache as important. | Plausible only if real S2/S3 pass. |
| Camera-readiness | The draft is a preregistration shell. It needs real S1/S2/S3 tables before submission as a method paper; current S2 tables are failing held-out scouts. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic S1 schema rehearsal | 288 rows, 12 prompts, 6 recurrent layers, 4 buckets; passes `--mode real --project ssq_lr` as `SCHEMA_REHEARSAL_NOT_PROMOTABLE...` | validates checker path only |
| held-out S1b | 192 saved-tensor rows, 12 held-out prompts, layers `0`, `12`, `18`, `30`; primary layers `0`, `12`, `30` pass with selected ratio `2.459`, CI low `1.861`, Holm p-min `2.78e-05` | non-post-hoc heterogeneity evidence |
| S2 MXFP4 scouts | MXFP4 preserves BF16 argmax on short continuation replay but reaches only `3.765x`--`3.938x` after scale bytes | fails byte gate |
| S2 INT3 scouts | 4-prompt block-256 scout passes (`5.224x`, zero argmax delta), but 12-prompt block-64 and block-256 scouts fail because INT3 loses argmax fidelity on held-out prompts | fails held-out quality gate |
| S2b mixed-block short-window scout | `mixed_int3_mxfp4_low_error_25pct` reaches `4.192x` counted state-memory reduction with zero BF16-argmax delta and `0.03956` selected NLL-delta CI high on 12 prompts | passed only as short-window filter |
| S2b mixed-block longer-window scout | Same recipe and prompt count with `--max-input-tokens 24 --prefix-tokens 8` keeps `4.192x`, but selected accuracy CI high is `0.0667` and selected NLL-delta CI high is `0.0764` | fails S2; stops mixed recipe before GPU |
| S2b layer-localization longer-window scout | Layers `0` and `30` pass individually with INT3 at `5.224x`, layer `12` fails, and combined layers `0,30` pass with zero selected accuracy delta and `0.04294` NLL CI high | narrows the live layer set but does not survive the stricter prefilter as pure INT3 |
| S3 prefilter replay | On layers `0,30`, pure INT3 weakens (`accuracy CI high 0.105`), while `mixed_int3_mxfp4_low_error_25pct` reaches `4.192x` with zero selected accuracy drift and `0.05044` NLL CI high | freezes the current live recipe candidate |
| cache-only S3 transfer prefilter | `experimental/shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507/` is checker-clean, cites one frozen recipe hash, and emits no retuned rows, but has only one complete local transfer model | superseded by Granite 350M transfer replay |
| Granite 350M 12-prompt transfer | `experimental/shared/results/ssq_lr_s3_transfer_granite_350m_12p_layers0_30_20260507/` fails S2 for the frozen `0,30` transfer surface; layer `0` passes alone on 350M, layer `30` fails, and local two-model S3 packets for both mixed25 and INT3 layer-0 recipes fail | current S3 branch is quality-failed, not GPU-ready |
| architecture provenance | shared config-derived hashes exist for live hybrid targets | packet provenance ready |
| trace collection plan | `experimental/shared/results/hybrid_trace_plan_20260507/ssq_lr_trace_plan.jsonl` enumerates 5,184 required S1 capture rows | execution checklist only |
| model eligibility | Granite Tiny is cached locally and used for real resource-limited S1b/S2 scouts; larger live targets remain uncached or GPU-sized. | Mac smoke path exercised; frontier validation still GPU-sized |
| real-packet checker | rejects missing buckets, incomplete prompt/layer matrices, stale summary fields, too few prompts, promotable resource-limited decisions, mismatched model IDs, missing controls, missing trace-plan hash pinning, missing tensor artifacts, tensor hash mismatches, and row metrics that do not recompute from saved state tensors | ready for real S1 |
| follow-up contract checker | `experimental.shared.followup_gate_contracts --gate ssq_lr_s2/ssq_lr_s3` enforces recipe/byte/CI/no-retuning fields before later evidence can be cited | contract ready; current S3 model rows fail |

## Reviewer Risks

- S1b heterogeneity is real on a resource-limited held-out surface, but S2b
  only survives after dropping layer `12`.
- Current S2 continuation replay is a short Mac filter, not task accuracy.
- Argmax fidelity alone is insufficient; future S2 must include continuation
  NLL and verbosity/length drift.
- A sub-FP16 state recipe may need per-model retuning, which would kill the
  cross-model claim.
- Byte savings must include scale and metadata overhead, not just nominal dtype.
- Nemotron-style FP16 stochastic-rounding SSM cache is the main deployment
  baseline; see `references/754_ssq_lr_nemotron_state_cache_refs_20260507.md`.

## Next Exact Gate

The frozen `mixed_int3_mxfp4_low_error_25pct` recipe on layers `0,30` has now
failed no-retuning transfer to Granite 350M on the 12-prompt Mac surface. The
next admissible SSQ-LR action is not GPU validation; it is either a new
preregistered recipe/layer-selection rule or branch demotion. Do not send
current resource-limited scouts to GPU as a positive method claim.
