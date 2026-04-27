# SVAMP70 Syndrome Bounds After Sketch Kill

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; the project still lacks
   a deployable source-derived positive method that survives live/holdout
   source controls.
2. Current paper story: source-conditioned sidecars and syndrome decoders remain
   useful diagnostics, but the current SVAMP70 surface does not support a
   promotable source-syndrome method.
3. Exact blocker to submission: `source_likelihood_sketch` is killed, and the
   post-kill syndrome bounds either fail target-self preservation or fail
   holdout source controls.
4. Current live branch: none. The next branch should be source-surface
   discovery, not another shallow likelihood or source-answer syndrome variant.
5. Highest-priority gate this cycle: replay C2C-teacher and source-teacher
   syndrome bounds on the frozen SVAMP70 live/holdout artifacts.
6. Scale-up rung: strict small/medium kill check on existing exact-ID
   artifacts.

## Commands

All commands were run at commit
`894d258a475fb41200444e2d08f6a3911eb4d198` with
`./venv_arm64/bin/python`.

Live C2C-teacher syndrome bound:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_syndrome_sidecar_probe.py \
  --target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --shuffle-offset 1 \
  --moduli-set 2,3 --moduli-set 2,3,5 --moduli-set 2,3,5,7 --moduli-set 97 \
  --min-correct 25 --min-clean-source-necessary 3 --min-numeric-coverage 70 \
  --output-json results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_live_c2c_syndrome_bound_after_sketch_kill.json \
  --output-md results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_live_c2c_syndrome_bound_after_sketch_kill.md
```

The holdout C2C command used the matching holdout paths. The source-teacher
commands replaced `--teacher c2c=...` with `--teacher source=...`, included
`c2c_generate` as a candidate, and used thresholds
`--min-correct 21 --min-clean-source-necessary 2 --min-numeric-coverage 60`.

## Results

### C2C Teacher

Live:

- best matched: `24/70`
- target-only: `21/70`
- clean source-necessary: `4/6`
- control clean union: `0/6`
- target-self matched: `14`
- status: `syndrome_sidecar_bound_fails_gate`
- blocker: below the predefined `25/70` minimum and one provenance issue

Holdout:

- best matched: `17/70`
- target-only: `8/70`
- clean source-necessary: `0/2`
- control clean union: `2/2`
- target-self matched: `11`
- status: `syndrome_sidecar_bound_fails_gate`
- blocker: the holdout clean pool is explained by source-destroying controls

### Source Teacher

Live:

- matched: `15/70`
- target-only: `21/70`
- clean source-necessary: `6/6`
- control clean union: `0/6`
- target-self matched: `4`
- status: `syndrome_sidecar_bound_fails_gate`
- blocker: it recovers source-only IDs by sacrificing target-self preservation

Holdout:

- matched: `9/70`
- target-only: `8/70`
- clean source-necessary: `1/2`
- control clean union: `1/2`
- target-self matched: `5`
- status: `syndrome_sidecar_bound_fails_gate`
- blocker: weak total gain and failed source-control separation

## Artifact Hashes

- `svamp70_live_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `662534e42454526872e760d3ca622daa25b04c84a82bf8600111533770b0d857`
- `svamp70_live_c2c_syndrome_bound_after_sketch_kill.md`
  - sha256:
    `1278aa33c3f89601b027eb130dcb6ffbc8cd3bc3545c6510f3e7cd8182c92614`
- `svamp70_holdout_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `a1125332f34e4585cb3efbc5d6e5b4ad7a4059695d03347881f1fde0000a9d29`
- `svamp70_holdout_c2c_syndrome_bound_after_sketch_kill.md`
  - sha256:
    `1044c8ec73575892d77f83f0705ade14328b8ec7525311e074f5184447c64a9a`
- `svamp70_live_source_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `3369b5590c7ea36c732ca8b03904bffe4af18d8067e2c4624e4798a6ce9fcb0a`
- `svamp70_live_source_syndrome_bound_after_sketch_kill.md`
  - sha256:
    `d9b5ad34397d5d9fffa0f27829b7039573213e84af3b15e6d55e3141a054de35`
- `svamp70_holdout_source_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `e6fd0ea71815835dc9a9026b8e3d75751c22830fe40b905f2ae1c533b672001f`
- `svamp70_holdout_source_syndrome_bound_after_sketch_kill.md`
  - sha256:
    `e9aa9e145fa86f3bc37a87500b0b6bdbccbbcd653899a4f4b222413185f7682f`

## Decision

Do not implement a richer source-syndrome predictor on this exact SVAMP70
live/holdout surface yet. The C2C-teacher bound has live headroom but fails
holdout controls, while the source-teacher bound recovers clean source-only IDs
by destroying target-correct preservation.

This kills the current post-sketch syndrome continuation as a live branch. The
next highest-value branch is source-surface discovery for a stronger frozen
surface where:

- C2C/source candidate bounds have clean source-necessary wins on both live and
  holdout slices
- target-self preservation remains close to target-alone
- controls have clean union `0`

## Hard Blocker

New source-surface discovery and C2C/model-feature collection require model
generation or forward passes. MPS remains blocked by orphaned PID `31103`:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Superseding update: the `chal311_380` artifacts were later found already
materialized and weak (`source-only=3/70`). The older adjacent-slice command is
therefore obsolete and should not be run as the next MPS action. It also
referenced a non-existent `scripts/run_baselines.py` helper; the correct
surface materializer in this repo is `scripts/materialize_generation_baselines.py`.

After PID `31103` is cleared, follow the next gate recorded in
`paper/postkill_historical_cpu_audit_20260427.md`.
