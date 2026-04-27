# Post-Kill Historical And CPU Artifact Audit

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing a
   deployable source-derived method with live/holdout controls, seed stability,
   paired uncertainty, systems accounting, and cross-family falsification.
2. Current paper story: target-side candidate decoding and C2C/source
   side-information bounds show real headroom, but every deployable source
   signal tried so far is either control-explained, target-self destructive,
   seed-unstable, or weaker than strong text/C2C baselines.
3. Exact blocker to submission: no live branch remains after
   `source_likelihood_sketch` and post-sketch syndrome bounds failed; MPS is
   blocked by orphaned PID `31103`.
4. Current live branch: none.
5. Highest-priority gate: use CPU-only artifact replay to verify whether any
   existing surface or historical positive branch is still worth promoting
   before spending new MPS time.
6. Scale-up rung: post-kill strict-small/medium branch selection.

## Historical Branch Audit

Three targeted audits re-read the MD files and result folders for `rotalign`,
`latent_bridge`, and the broader results tree.

RotAlign/DynAlign:

- `dynalign_module_replace_residrank16` remains the strongest GSM mechanism
  clue: GSM8K32 reached `4/32` vs target `2/32`, and GSM70 seed 0 reached
  `8/70` vs target `4/70`.
- It is not live: seed 3 fell to `2/70`, seed 4 was finite but only `4/70`,
  and seeds 1/2 had nonfinite checkpoint failures.
- If revived later, it must be a target-safe V-side conditional innovation
  variant with zero/shuffle/predictor-only controls, not another raw DynAlign
  seed repeat.

Latent bridge/query memory:

- ID-weighted query innovation remains the best latent-bridge source-specific
  clue: one clean ID (`aee922049c757331`) was not retained by translated-KV-zero.
- The row still fails promotion: best controlled variants recover only `1/6`
  clean residual IDs and do not preserve the target-self repair ceiling.
- Perceiver/query-memory checkpoints stay killed; the four-ID positive-looking
  answer-likelihood smoke failed on the required six-clean-ID expansion.

Side-information family:

- SVAMP32 source-contrastive sidecar remains the best historical
  source-derived positive clue, but medium/disjoint validation showed fixed
  decoded guards do not generalize.
- The post-kill chal241-310 CV router below confirms that shallow source-text
  feature gates do not rescue adjacent same-pair surfaces.

## CPU Artifact Gates

### Chal241-310 Post-Kill CV Router

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_sidecar_cv_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --accept-penalty 0.10 \
  --min-correct 12 \
  --min-target-self 0 \
  --min-clean-source-necessary 2 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 63 \
  --output-json results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_sidecar.json \
  --output-md results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_sidecar.md \
  --output-predictions-jsonl results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_predictions.jsonl \
  --prediction-method source_cv_router_penalty010_postkill_sidecar
```

Result: fail. Best row (`2,3`) matched `10/70`, recovered only `1` clean
source-necessary ID, had control clean union `0`, and accepted harm `1`.

Artifact hashes:

- JSON:
  `99a742cd10efaf43136be8d3d666b1bfc3fcb73507c66289d58cec5c1654e51b`
- MD:
  `672fc1e882b01908d227ab814c8359ecca30e107e2e376437f943abd086f74f1`
- predictions JSONL:
  `24dbd297d4c18cabe79f141e34df858df6437419d54e6913b0fff9e0770e7a88`

Decision: kill this CV source-text router on `chal241-310`; do not spend C2C
or learned-interface compute on that weak adjacent surface.

### Consolidated Existing-Surface Scan

Command:

```bash
mkdir -p .debug/cpu_only_next_gate_20260427 && ./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface svamp70_live=target_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=already_killed_live_surface \
  --surface svamp70_holdout=target_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=already_failed_holdout_controls \
  --surface svamp70_chal171_240=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=weak_surface \
  --surface svamp70_chal241_310=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=cv_router_already_failed \
  --surface svamp70_chal311_380=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=post_blocker_next_surface \
  --min-source-only 6 \
  --output-json .debug/cpu_only_next_gate_20260427/source_headroom_surfaces_with_chal311.json \
  --output-md .debug/cpu_only_next_gate_20260427/source_headroom_surfaces_with_chal311.md
```

Result summary:

| Surface | Status | Target | Source | Source-only | Oracle | Decision |
|---|---|---:|---:|---:|---:|---|
| `svamp70_live` | strong | 21/70 | 13/70 | 9 | 30/70 | consumed and killed by controls |
| `svamp70_holdout` | strong | 8/70 | 8/70 | 6 | 14/70 | consumed and failed controls |
| `svamp70_chal171_240` | weak | 22/70 | 8/70 | 2 | 24/70 | no spend |
| `svamp70_chal241_310` | weak | 10/70 | 5/70 | 4 | 14/70 | CV router failed |
| `svamp70_chal311_380` | weak | 21/70 | 8/70 | 3 | 24/70 | no spend |

Artifact hashes:

- JSON:
  `181df1b5b0f71c6bde86cccc7d72cddea77c61bbe54c2f762f8fb07952e885eb`
- MD:
  `0bbf961578eaf80db54ed99bcb3c82b1bafbd31b884f0544d58dd42068fc3981`

Decision: existing-artifact CPU surface mining is exhausted. In particular,
do not run the previously recorded `chal311_380` MPS scout; the artifacts
already exist and fail the source-mass threshold.

## Hard Blocker And Resume Gate

MPS remains blocked by orphaned PID `31103`:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Observed state: `STAT=UE`, `PPID=1`, command
`scripts/calibrate.py ... --device mps --dtype float32 ...`.

If the process remains present, stop MPS work and clear it at the OS/session
level. If it is absent, the next exact gate is a new stronger-source surface
scout, not another adjacent Qwen2.5-Math-1.5B -> Qwen3 SVAMP slice:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Promotion rule for that scout: do not implement or run C2C/connector methods
unless ordered ID parity holds, numeric coverage is high, source-only over
target is at least `6/70`, and target-or-source oracle beats target by at
least `6/70`.

## Decision

End state reached for the current loop segment: hard blocker. The current live
branch is none; the previous live branch is decisively killed, historical
branches are pruned to mechanism clues, and no CPU-only next command remains
that can produce a promotable result.
