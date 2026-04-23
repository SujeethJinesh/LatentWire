# SVAMP32 Dynalign Source Controls

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The project is still blocked on a source-specific positive
method. This turn resolves the cheapest remaining legacy-dynalign ambiguity on
the frozen SVAMP32 C2C-teacher gate.

## Current Story

SVAMP32 C2C remains the strongest same-pair teacher surface:

- target-alone: `8/32`
- C2C: `16/32`
- C2C-only target-complementary wins: `10`

The old prefdist dynalign salts were the only surviving internal rows with any
teacher-only overlap not already explained by target self-repair. The question
for this turn was whether those overlaps survive exact-ID zero-source and
deterministic shuffled-source controls.

## Exact Blocking Gap

Before training a new connector, we needed to know whether any existing legacy
dynalign win on the SVAMP32 C2C-only IDs is truly source-conditioned. If not,
legacy dynalign should be demoted to a mechanism probe / weak lower-bound
comparator rather than treated as a live paper method.

## What Ran

Using the same prefdist dynalign checkpoint and exact 32-example materialized
SVAMP slice:

- salt 1 matched row versus zero-source and shuffled-source controls
- salt 2 matched row versus zero-source and shuffled-source controls
- C2C-teacher innovation probe on each salt/control bundle

Materialized slice:

- `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`

Legacy dynalign checkpoint:

- `checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt`

## Evidence

### Salt 1

- matched row: `9/32`
- teacher-only recovered: `1/10`
- recovered teacher-only ID: `575d7e83d84c1e67`
- gate result: `candidate_teacher_recovery_explained_by_controls`

Control result:

- zero-source also recovers `575d7e83d84c1e67`
- shuffled-source also recovers `575d7e83d84c1e67`

Interpretation: the only salt 1 teacher-only hit is not source-specific.

### Salt 2

- matched row: `8/32`
- teacher-only recovered: `2/10`
- recovered teacher-only IDs:
  - `4d780f825bb8541c`
  - `e3ab8666238a289e`
- losses vs target: `4`
- gate result: `candidate_teacher_recovery_partially_control_explained`

Control result:

- shuffled-source recovers `4d780f825bb8541c`
- zero-source does not recover either matched teacher-only ID, but it does
  recover a different teacher-only ID, `aee922049c757331`
- `e3ab8666238a289e` remains matched-only in this probe

Interpretation: salt 2 preserves one weak unmatched teacher-only clue, but the
row is still not a paper-grade positive result. It has only `2/10` teacher-only
recoveries total, one is explained by shuffled-source, and the row incurs `4`
target losses. This is useful as a lower-bound comparator, not as a live method.

## Decision

Alive:

- A new source-control-contrastive innovation connector remains alive.
- Legacy dynalign salt 2 remains a weak lower-bound clue because
  `e3ab8666238a289e` is matched-only in this exact probe.

Killed / demoted:

- Salt 1 as a source-specific signal is killed.
- Legacy dynalign as a publishable positive method is demoted. The best old row
  still has partial control explanation and too many target losses.

Promoted:

- The next mainline branch should be a compact learned connector
  (query bottleneck / innovation resampler / Q-former-style sidecar) trained
  directly against the C2C-teacher innovation surface with matched, zero-source,
  and shuffled-source controls built into the objective and evaluation.

## Next Exact Gate

Train or run a source-control-contrastive connector on this same frozen SVAMP32
gate and promote only if it satisfies all of:

- recovers at least `4/10` C2C-only teacher wins
- reaches at least `11/32` overall
- loses at most `1` target-correct ID
- zero-source and shuffled-source controls each recover at most `1` of the same
  matched teacher-only wins

If that fails, keep C2C strictly as a teacher/headroom surface and stop treating
same-family latent transfer as a live positive-method claim.
