# SVAMP32 Full32 Target Sampling Reachability Gate

Date: `2026-04-27`

## Cycle State

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives text relay, source-destroying controls, and seed/rung confirmation.
2. Current paper story: target/no-source sampling can create receiver-side
   candidate headroom, but numeric and process sidecars have not turned that
   headroom into source communication.
3. Exact blocker: determine whether the target/no-source candidate pool has
   enough clean C2C residual reachability to justify another source-derived
   selector or connector gate.
4. Current live branches: broader target/no-source candidate-pool discovery;
   JEPA-style anti-collapse connector only if a non-leaky reachable surface
   exists.
5. Highest-priority gate: SVAMP32 full 32 target-only sampling with 8 samples
   per ID.
6. Scale-up rung: strict small gate.

## Commands

Sample 8 target/no-source candidates per frozen SVAMP32 ID:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 43 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl \
  --output-json results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.json \
  --output-md results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.md
```

Materialize and audit:

```bash
./venv_arm64/bin/python scripts/materialize_no_source_candidate_surface.py \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --candidate target_sample_s0=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s0 \
  --candidate target_sample_s1=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s1 \
  --candidate target_sample_s2=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s2 \
  --candidate target_sample_s3=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s3 \
  --candidate target_sample_s4=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s4 \
  --candidate target_sample_s5=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s5 \
  --candidate target_sample_s6=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s6 \
  --candidate target_sample_s7=path=results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl,method=target_sample_s7 \
  --min-source-only 0 \
  --date 2026-04-27 \
  --output-dir results/svamp32_target_sampling_full32_s8_20260427/no_source_surface

./venv_arm64/bin/python scripts/analyze_target_sampling_reachability.py \
  --samples-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --date 2026-04-27 \
  --output-json results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --output-md results/svamp32_target_sampling_full32_s8_20260427/reachability.md

./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set svamp32_full32_s8=path=results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json,role=target_no_source_full32,note=8_target_samples_per_id \
  --date 2026-04-27 \
  --output-json results/svamp32_target_sampling_full32_s8_20260427/headroom.json \
  --output-md results/svamp32_target_sampling_full32_s8_20260427/headroom.md
```

## Evidence

Raw target/no-source sampling:

- rows: `256`
- exact ID coverage: `32/32`
- numeric coverage: `256/256`
- candidate oracle: `14/32`
- target baseline: `8/32`
- raw sample oracle gain over target: `7`
- C2C clean residual IDs in pool: `2/6`
- C2C teacher-only IDs in pool: `4/9`
- source-contrastive clean IDs in pool: `2/4`
- mean unique sampled answers per ID: `3.344`
- duplicate nonempty row fraction: `0.582`

The two C2C-clean residual IDs reached by the full32 pool are the same two
already reached by the clean6 16-sample gate:

- `3e8a5691f5443495`
- `575d7e83d84c1e67`

Merged no-source surface:

- target correct: `8/32`
- source correct: `6/32`
- target-side oracle with text relay plus samples: `18/32`
- oracle gain: `10`
- remaining clean source-only after no-source baselines: `2`
- remaining clean IDs with gold in target-side pool: `0/2`

## Decision

This is a reachability pass, not a communication-method pass. The target-side
pool is much larger, but it does not expand the clean C2C residual selector
surface beyond the already-tested two IDs. That means another deterministic
numeric/process selector on the same reachable clean IDs would violate the
anti-loop rule.

Do not top up to 16 samples yet. The clean6 `s16` gate and this full32 `s8`
gate both saturate at the same C2C-clean IDs. Top-up is only justified if the
next method needs a full no-source baseline frontier, not to rescue the current
selector family.

## JEPA / Anti-Collapse Implication

JEPA, LeJEPA, and V-JEPA remain useful as connector design constraints:
answer-masked dual source views, frozen target/candidate latent targets,
matched-source margins over zero/shuffled/target-only/slots-only controls, and
variance/effective-rank/covariance/Barlow/SIGReg telemetry. This result changes
the next JEPA gate: it should not train to reconstruct all target states or rank
all no-source samples. It should target source innovation over a frozen
target-side candidate pool and pass only if matched source selects a
source-necessary clean candidate that the source-destroying controls miss.

## Next Exact Gate

Switch from hand-built selectors to a bounded learned source-conditioned
candidate generator or frozen-latent/rate-capped connector smoke. Use the
full32 no-source pool as the target-prior baseline. Pass requires:

- at least `1` C2C-clean source-necessary recovery beyond target/text/no-source
  controls
- zero-source, shuffled-source, target-only/slots-only, and random same-byte
  controls recover `0` clean IDs
- no target-correct harm
- exact ID parity and numeric coverage
- bytes/latency/generated-token accounting
- collapse telemetry showing non-collapsed side information
