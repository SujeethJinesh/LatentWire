# SVAMP32 Stronger-Source Margin Audit - 2026-04-24

## Status

- status: `no_source_margin_signal`
- readiness impact: negative for source-final/source-margin methods; positive
  for narrowing the next gate
- gate: a stronger source must expose at least `2/6` clean residual IDs through
  source-final correctness, text relay, or positive gold-vs-target-wrong source
  margin advantage
- outcome: strongest source tested, `Qwen/Qwen2.5-7B-Instruct`, exposes only
  `1/6` clean text-relay final answers and `1/6` clean positive source-margin
  advantage

## Motivation

The previous audit killed source-final and source-margin evidence for the
same-pair `Qwen/Qwen2.5-0.5B-Instruct` to `Qwen/Qwen3-0.6B` surface. The exact
blocking question was whether the failure was merely weak source quality. This
turn tests two stronger cached Qwen sources on the same frozen SVAMP32 clean
residual IDs before any new connector training.

Local reference anchors:

- `paper/svamp32_source_margin_audit_20260424.md`
- `paper/svamp32_source_oracle_bound_20260424.md`
- `paper/svamp32_answer_teacher_microfit_20260424.md`
- `references/451_answer_teacher_microfit_refs.md`

## Decision Surface

Top moves considered:

- Stronger-source source-informativeness audit. It matters because it isolates
  source weakness from connector weakness on the same exact IDs. It might fail
  if C2C's headroom is cache-fusion-specific rather than extractable source
  answer evidence. It costs one to two model-scored audits plus small
  source/text generation subsets and helps same-family, reproducibility, and
  interpretability.
- C2C-residual distillation sidecar. It matters because C2C remains `16/32`
  while source-answer channels are weak. It might fail by learning
  target-cache shortcuts unless the same zero/shuffle/target-only controls are
  mandatory. It costs new implementation and helps same-pair and efficiency if
  it clears.
- Latent arithmetic-syndrome sidecar. It matters because it treats the target
  candidate pool as decoder side information and transmits compact source
  checks. It might fail through dataset-prior leakage. It helps efficiency and
  interpretability, but is a new branch after source-informativeness is known.

I executed the stronger-source audit because it is the cheapest decisive gate.

## Implementation

Extended `scripts/analyze_svamp32_source_margin_audit.py`:

- added optional `--source-jsonl` and `--text-jsonl`
- records unknown final-answer provenance explicitly when those files are
  absent
- added `--dtype` for larger cached sources such as 7B in float16

Added `scripts/materialize_generation_id_subset.py`:

- materializes generation examples by stable IDs from target-set fields
- used to create a 9-example clean-plus-target-self subset for cheap 7B
  source/text generation

Unit tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_source_margin_audit.py \
  tests/test_materialize_generation_id_subset.py -q
```

Result: `6 passed in 0.03s`.

Full verification:

```bash
./venv_arm64/bin/python -m pytest -q
```

Result: `660 passed in 25.35s`.

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/analyze_svamp32_source_margin_audit.py \
  scripts/materialize_generation_id_subset.py
```

Result: pass.

Additional checks:

- JSON artifact validation: pass
- `git diff --check`: pass

## Commands

1.5B margin audit:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_margin_audit.py \
  --source-model Qwen/Qwen2.5-1.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_stronger_source_margin_audit_20260424/qwen25_15b_to_qwen3_06b_source_margin_clean_self.json \
  --output-md results/svamp32_stronger_source_margin_audit_20260424/qwen25_15b_to_qwen3_06b_source_margin_clean_self.md \
  --device mps \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --score-target-self
```

1.5B full-32 source-final materialization:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/svamp32_stronger_source_baselines_20260424 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-1.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking
```

7B clean-plus-target-self subset:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --id-fields clean_residual_targets target_self_repair \
  --output-jsonl results/svamp32_stronger_source_margin_audit_20260424/svamp32_clean_self_eval.jsonl
```

7B source/text generation on the 9-ID subset:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp32_stronger_source_margin_audit_20260424/svamp32_clean_self_eval.jsonl \
  --task-type generation \
  --device mps \
  --dtype float16 \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --methods source \
  --prediction-output results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_source_alone_clean_self.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp32_stronger_source_margin_audit_20260424/svamp32_clean_self_eval.jsonl \
  --task-type generation \
  --device mps \
  --dtype float16 \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --methods t2t \
  --prediction-output results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_text_to_text_clean_self.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false
```

7B final combined audit:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_margin_audit.py \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --source-jsonl results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_source_alone_clean_self.jsonl \
  --text-jsonl results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_text_to_text_clean_self.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.json \
  --output-md results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.md \
  --device mps \
  --dtype float16 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --score-target-self
```

## Evidence

1.5B source:

- source-final full SVAMP32: `3/32`
- source-final clean residual: `0/6`
- source-margin positive clean IDs: `3/6`
- source-margin positive+advantage clean IDs: `1/6`
- positive-advantage ID: `6e9745b37ab6fc45`

7B source:

- source-final clean-plus-target-self subset: `2/9`
- source-final clean residual: `0/6`
- text-relay clean-plus-target-self subset: `1/9`
- text-relay clean residual: `1/6`
- text-relay clean ID: `e3ab8666238a289e`
- source-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `1/6`
- positive-advantage ID: `6e9745b37ab6fc45`
- mean source margin: `1.343750`
- mean target margin: `-3.619792`
- mean source-minus-target margin: `4.963542`

Clean residual table for the final 7B audit:

| Example ID | Gold | Target Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `13cb77b698eeadb5` | 8142 | 46 | 8 | 17 | -5.312500 | -12.984375 | 7.671875 | no_source_margin_advantage |
| `1d50b408c8f5cd2c` | 949 | 1 | 246 | 1 | -8.500000 | -11.593750 | 3.093750 | no_source_margin_advantage |
| `2de1549556000830` | 39 | 33 | 3 | 1 | -7.500000 | -9.343750 | 1.843750 | no_source_margin_advantage |
| `6e9745b37ab6fc45` | 61 | 600 | 661 | 39 | 20.312500 | -8.296875 | 28.609375 | source_positive_advantage |
| `aee922049c757331` | 1 | 17 | 4 | -4 | 13.437500 | 14.531250 | -1.093750 | no_source_margin_advantage |
| `e3ab8666238a289e` | 1 | 4 | 4 | 1 | -4.375000 | 5.968750 | -10.343750 | no_source_margin_advantage |

The strongest source exposes two isolated clean signals, but not through one
source-necessary channel: `6e9745b37ab6fc45` has a positive source-margin
advantage while source/text generation are wrong; `e3ab8666238a289e` is fixed
by text relay while source margin is negative and the target-alone margin was
already positive. This is below the method-promotion threshold and still risky
as target-prior leakage.

## Artifacts

See `results/svamp32_stronger_source_margin_audit_20260424/manifest.md`.

## Subagent Synthesis

- source-pair agent recommended exactly this `Qwen/Qwen2.5-1.5B-Instruct`
  stronger-source gate first
- ablation agent recommended source/text final provenance plus margin matrices,
  and warned that stronger aggregate source accuracy may still miss the clean
  residual IDs
- repo-audit agent recommended optional source/text provenance in the audit so
  stronger-source margin runs do not depend on stale source JSONL artifacts
- creative/literature agent recommended a Wyner-Ziv latent syndrome sidecar
  grounded in decoder-side-information coding, but only if matched source
  checks beat zero/shuffle controls

## Hypothesis Update

- killed: increasing Qwen source scale to 1.5B or 7B is sufficient to create a
  source-answer/source-margin surface for the six clean SVAMP32 residual IDs
- killed: source-final answer copying is a plausible positive-method path for
  the current frozen slice
- weakened: text relay from a stronger source as the main comparator; it
  recovers only `1/6` clean and harms target-self rows on this subset
- revived weakly: one clean ID, `6e9745b37ab6fc45`, has real source-margin
  advantage under 1.5B and 7B sources
- still alive: C2C/cache-residual distillation, because C2C gets `16/32` and
  all source-answer channels remain below gate
- still alive: latent syndrome sidecar, but only if it uses strict
  matched-vs-zero/shuffle source controls
- promoted: C2C-residual distillation or source-control syndrome sidecar as
  the next method gate; not another source-answer connector

## Next Exact Gate

Implement a C2C-residual or syndrome-style sidecar that treats the target's own
candidate pool as decoder side information and must clear:

- `>=2/6` clean residual IDs
- matched source beats zero-source, shuffled-source, target-only, and slots-only
  controls
- `>=14/32` target-self/self-repair floor preserved
- exact ID parity and numeric coverage recorded

Do not run another same-family source-answer or margin-only connector gate
until a new frozen slice or new pair is selected.
