# No-Source Candidate Surface And Target Sampling

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke / source-surface discovery
- live branch after this cycle: target-only sampled candidate surface plus source-derived selector

## Question

Can existing no-source candidate artifacts provide a target-side pool where a
source-derived sidecar can recover clean source-necessary SVAMP70 live IDs?

## Commands

```bash
./venv_arm64/bin/python scripts/materialize_no_source_candidate_surface.py \
  --base-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --candidate target_self_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair \
  --candidate process_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=process_repair_selected_route \
  --expand-candidate-scores zero_source_pool=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair \
  --min-source-only 0 \
  --date 2026-04-27 \
  --output-dir results/no_source_candidate_surface_20260427
```

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set zero_source_candidate_surface=path=results/no_source_candidate_surface_20260427/source_contrastive_target_set.json,role=no_source_target_pool,note=target+t2t+target_self+process_repair+zero_source_seed_pool \
  --target-set canonical_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=canonical,note=target+t2t_only \
  --target-set target_self_surface=path=results/target_self_repair_candidate_surface_20260427/live_target_self_repair_target_set.json,role=target_self_only,note=target+t2t+target_self_repair \
  --date 2026-04-27 \
  --output-json results/no_source_candidate_surface_20260427/target_side_candidate_headroom.json \
  --output-md results/no_source_candidate_surface_20260427/target_side_candidate_headroom.md
```

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file data/svamp_eval_70.jsonl \
  --target-set-json results/no_source_candidate_surface_20260427/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --output-jsonl results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --output-meta-json results/target_only_sampling_clean3_20260427/clean_source_only_eval.meta.json
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 11 \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 96 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/target_only_sampling_clean3_20260427/target_only_samples.jsonl \
  --output-json results/target_only_sampling_clean3_20260427/target_only_samples.json \
  --output-md results/target_only_sampling_clean3_20260427/target_only_samples.md
```

## Evidence

Existing no-source artifacts are strong target-side baselines but not sufficient
for the source-side selector branch:

| Surface | Target | Source | Target-Side Oracle | Clean In Pool |
|---|---:|---:|---:|---:|
| zero-source candidate surface | `21/70` | `13/70` | `48/70` | `0/3` |
| target-self surface | `21/70` | `13/70` | `47/70` | `0/3` |
| canonical live | `21/70` | `13/70` | `33/70` | `0/6` |

The zero-source surface leaves three clean source-only IDs after no-source
baselines, but none of those three gold answers appear in the no-source pool:

- `14bfbfc94f2c2e7b`
- `2de1549556000830`
- `41cce6c6e6bb0058`

Target-only stochastic sampling on just those three IDs recovers one ID:

- candidate oracle: `1/3`
- recovered ID: `14bfbfc94f2c2e7b`
- correct sampled methods for that ID: `target_sample_s1`, `target_sample_s2`,
  `target_sample_s4`

## Decision

Kill the immediate branch: source-derived selector over the existing
zero-source candidate surface. It has no clean source-necessary target in the
pool.

Revive a narrower branch: sampled target-only candidate surface plus a
source-derived selector. The smoke is not positive evidence yet, but it proves
one remaining source-clean ID can be made reachable without source leakage.

## Artifacts

- `results/no_source_candidate_surface_20260427/manifest.md`
- `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`
- `results/no_source_candidate_surface_20260427/target_side_candidate_headroom.md`
- `results/target_only_sampling_clean3_20260427/clean_source_only_eval.meta.json`
- `results/target_only_sampling_clean3_20260427/target_only_samples.md`

Hashes:

- `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`:
  `fb615786f89643c6208909534f59896c2f7d8987b29043842941a769e52e26aa`
- `results/no_source_candidate_surface_20260427/target_side_candidate_headroom.json`:
  `4ebf19932e36de5deaf1f463667b4c1dfb096743140f0779b3afe334722969b9`
- `results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl`:
  `a84668c43d47dd72be58daa1a608295bdc42c261c83777ab2eaa33f78c48946b`
- `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl`:
  `f40a89d736afc5da7b28a4b4e01bfdb12650a97dbb67b84b94104674ac427908`
- `results/target_only_sampling_clean3_20260427/target_only_samples.json`:
  `fccf83e7e0b61c023b85b80d524f6bc6ecf9becf852b69702fa115076113de0f`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_materialize_no_source_candidate_surface.py \
  tests/test_sample_target_candidate_surface.py \
  tests/test_analyze_target_side_candidate_headroom.py \
  tests/test_build_source_contrastive_target_set.py -q
```

Result: `5 passed`.

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/materialize_no_source_candidate_surface.py \
  scripts/sample_target_candidate_surface.py \
  scripts/analyze_target_side_candidate_headroom.py \
  scripts/build_source_contrastive_target_set.py
```

Result: passed.

## Next Gate

Materialize a clean3 target set containing the target-only sampled candidates,
then run the source-derived candidate sidecar and controls on that sampled pool.
Promote only if matched source selects `14bfbfc94f2c2e7b` while shuffled-source,
zero-source, random same-byte, target-only, and slots-only controls do not.
