# Generation Baseline Materialization

- date: `2026-04-26`
- eval file: `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310.jsonl`
- materialized eval file: `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310_70.jsonl`
- limit: `70`
- dry run: `False`

## Run Rows

| Method | Status | Return | Output | Log |
|---|---|---:|---|---|
| `source` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/logs/source.log` |
| `target` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/logs/target.log` |
| `t2t` | `ran` | 0 | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/text_to_text.jsonl` | `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/logs/t2t.log` |

## Method Summaries

| Method | Correct | Accuracy | Exact ID parity | Numeric coverage | Empty predictions |
|---|---:|---:|---:|---:|---:|
| `source` | 5/70 | 0.071 | `True` | 63/70 | 0 |
| `target` | 10/70 | 0.143 | `True` | 70/70 | 0 |
| `t2t` | 14/70 | 0.200 | `True` | 70/70 | 0 |

## Pairwise Versus Target

| Method | Method-only | Target-only | Both correct | Oracle |
|---|---:|---:|---:|---:|
| `source` | 4 | 9 | 1 | 14/70 |
| `t2t` | 10 | 6 | 4 | 20/70 |

## Post-Kill Source Sidecar CV Router

- date: `2026-04-27`
- status: `source_sidecar_cv_router_fails_gate`
- command:

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

Best row:

- moduli: `2,3`
- matched: `10/70`
- clean source-necessary: `1`
- control clean union: `0`
- accepted harm: `1`
- decision: fail; do not spend C2C or connector compute on this weak adjacent
  surface

Artifacts:

- `source_cv_router_penalty010_postkill_sidecar.json`
  - sha256:
    `99a742cd10efaf43136be8d3d666b1bfc3fcb73507c66289d58cec5c1654e51b`
- `source_cv_router_penalty010_postkill_sidecar.md`
  - sha256:
    `672fc1e882b01908d227ab814c8359ecca30e107e2e376437f943abd086f74f1`
- `source_cv_router_penalty010_postkill_predictions.jsonl`
  - sha256:
    `24dbd297d4c18cabe79f141e34df858df6437419d54e6913b0fff9e0770e7a88`

## Commands

### source

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods source --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/.source_alone.jsonl.tmp.source.41197 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### target

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods target --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/.target_alone.jsonl.tmp.target.41197 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```

### t2t

```bash
/Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/python /Users/sujeethjinesh/Desktop/LatentWire/latent_bridge/evaluate.py --translator /Users/sujeethjinesh/Desktop/LatentWire/checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/_artifacts/svamp_chal241_310_70.jsonl --task-type generation --device mps --max-new-tokens 64 --source-reasoning-mode brief_analysis --methods t2t --prediction-output /Users/sujeethjinesh/Desktop/LatentWire/results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/.text_to_text.jsonl.tmp.t2t.41197 --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false
```
