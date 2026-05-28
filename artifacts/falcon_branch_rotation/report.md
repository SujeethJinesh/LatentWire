# Falcon Branch Rotation Diagnostic

Created: `2026-05-28T15:37:22Z`

## Status

- Paper readiness: not ICLR-ready.
- Current story: ParoQuant-style rotation is positive on Granite and Nemotron; Falcon remains unresolved.
- Blocking gap: Falcon needs either a rotation smoke result, a branch/surface diagnostic showing a lower-drift target, or a channel fallback that improves recovery under matched cost.

No GPU jobs were run for this artifact. This packet only maps Falcon-H1 branch surfaces and prepares a gated diagnostic/protection configuration.

## Decision

**Decision: `NEEDS_GPU_DIAGNOSTIC`.**

Falcon branch hooks are feasible, but cached branch-local activations do not exist. Existing cached evidence is post-block only:

- Phase 7 Falcon post-block strict set-leaving at 20K: `0.674`.
- Phase 9 Falcon post-block strict set-leaving at 10K: `0.639`.
- Latest funnel layer endpoint readout remains post-block and is not branch-local.

There is no observed branch-local drift or covariance range at least `0.15` below post-mixer/post-block. Do not promote BranchRot to endpoint scoring before a tiny diagnostic captures projected attention, projected Mamba, post-mixer, and post-block surfaces on the same traces.

## Hypothesis

Falcon-H1 may look broadband at the block output because attention and Mamba branches are summed before the residual/MLP. If one branch has lower drift or lower rotated covariance range, a branch-local rotation/protection policy could rescue Falcon without changing the whole block basis.

## Branch Surfaces

Primary surfaces to capture:

| Surface | Hook target | Width | Use |
|---|---|---:|---|
| `attention_projected_presum` | output of self-attention after output projection and branch multiplier, before branch sum | 1024 | attention branch drift/range |
| `mamba_projected_presum` | output of Mamba after output projection and branch multiplier, before branch sum | 1024 | Mamba branch drift/range |
| `post_mixer_pre_residual` | attention + Mamba branch sum before residual add | 1024 | branch-sum control |
| `post_block_output` | decoder block output after residual and MLP | 1024 | cached/post-block reference |

Native pre-projection surfaces are hookable but lower priority because their widths differ from the residual stream and tensor-parallel layouts can make interpretation harder.

## Promotion Rule

Promote to branch-local rotation/protection only if the diagnostic shows at least one projected branch surface with:

- strict set-leaving at least `0.15` absolute below matched post-block, or
- covariance range/CVaR materially below post-mixer in the same traces, and
- the direction is stable across the stratified smoke traces `[7, 1, 11]`.

Otherwise, record Falcon as likely branch-broadband and continue with ParoQuant Falcon smoke or Falcon HYST fallback.

## Caveats

- This artifact does not prove BranchRot works.
- No cached branch-local activation packet was found.
- A runner for branch-local rotation scoring does not yet exist; `command.sh` is a guarded diagnostic template only.

