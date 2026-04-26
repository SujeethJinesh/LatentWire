# SVAMP32 Source-Only Sidecar Router Gate - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: a new source-derived positive method plus larger-slice,
  seed-repeat, uncertainty, and cross-family gates
- current story: removing target memory from source-signal formation avoids
  leakage, but raw source numeric answers are too weak to recover the clean
  C2C-only IDs
- blocker: no clean source-necessary wins and no target-self preservation

## Gate

After the anti-memory Perceiver objective failed, this gate tested the cleanest
source-only sidecar: source-generated numeric predictions are compressed into a
residue code, then decoded against the target-side candidate pool.

The source signal has no target-only or slot memory access. Controls were:

- `zero_source`
- `shuffled_source`
- `label_shuffle`
- `same_norm_noise`
- `target_only`
- `slots_only`

## Evidence

| Moduli | Bytes | Matched | Target-Self | Clean Matched | Clean Necessary | Control Clean Union | Failing Criteria |
|---|---:|---:|---:|---:|---:|---:|---|
| `2,3,5,7` | 1 | 4/32 | 0/3 | 0/6 | 0/6 | 0/6 | min_correct, min_target_self, min_clean_source_necessary |
| `97` | 1 | 4/32 | 0/3 | 0/6 | 0/6 | 0/6 | min_correct, min_target_self, min_clean_source_necessary |

Source numeric coverage was `32/32`, so the failure is signal quality, not
parser coverage. The source-alone row is wrong on all `6` clean residual IDs
and all `3` target-self preserve IDs.

## Decision

Kill the simple source-generated numeric sidecar/router branch. It is useful as
a clean negative control because target/control leakage is not the issue here;
the source signal itself is too weak.

## Next Gate

Move to a stronger source-derived signal:

- learned source latent or token-level predictor of C2C residues with
  cross-fitting and label-shuffle controls, or
- token/layer-level C2C residual distillation with matched-vs-control
  separation, or
- a minimal source-only latent sidecar fused with target candidate pools under
  the same control matrix.

Do not scale source-generated numeric residue sidecars.
