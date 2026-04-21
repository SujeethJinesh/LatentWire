# Interpretability Telemetry: Next Tests

Date: 2026-04-20

This note extends the current telemetry story with the next reusable
interpretability tests. It is grounded in:

- `paper/bytes_accuracy_table_20260420.md`
- `paper/layer_localization_20260420.md`
- `paper/failure_taxonomy_20260419.md`
- `results/layer_knockout_20260420/*`

Current readout worth preserving:

- exact Qwen GSM70: `fixed prior = 0.0857`, `grouped subspace + rank-4 residual = 0.0571`, `C2C = 0.1286`
- weak modular family shares the same layer signature:
  `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`
- dynalign smoke:
  `bridge_ridge_qk_dynalign_module_replace = 0.4000` on `gsm8k_5`,
  but only `0.1000` on the controlled `gsm8k_eval_10` slice
- current layer-knockout sanity check:
  top-5 and offset-5 both land at `0.2000`

Primary interpretability refs:

- **AtP\***: An efficient and scalable method for localizing LLM behaviour to
  components — 2024-03-01
  https://arxiv.org/abs/2403.00745
- **Causal Head Gating**: A Framework for Interpreting Roles of Attention
  Heads in Transformers — 2025-05-19
  https://arxiv.org/abs/2505.13737
- **SCBench**: A KV Cache-Centric Analysis of Long-Context Methods — 2024-12-13
  https://arxiv.org/abs/2412.10319
- **Rethinking Key-Value Cache Compression Techniques for Large Language Model
  Serving** — 2025-03-31
  https://arxiv.org/abs/2503.24000

## 1) Single-layer leave-one-out / add-back

Goal: determine whether the recurrent signature is causal or just a repeated
selector artifact.

What to test:

- leave out each shared layer one at a time: `27`, `5`, `23`, `22`, `8`
- add back one layer at a time from the full 5-layer knockout
- compare against matched random same-count layers

Why this is reusable:

- the current top-5 and offset-5 knockouts both tie at `0.2000`
- a stricter one-layer sweep is the cheapest way to separate a real causal
  circuit from budget saturation

Exact commands:

```bash
source .venv/bin/activate

# Create a fresh controlled 10-example slice from GSM100 if you want a rerunable GSM10 control.
python scripts/split_jsonl_dataset.py \
  --input data/gsm8k_100.jsonl \
  --search-output .debug/gsm8k_eval_10.jsonl \
  --eval-output .debug/gsm8k_eval_90.jsonl \
  --search-count 10 \
  --seed 0

# Leave-one-out on the shared layer signature.
for layer in 27 5 23 22 8; do
  python scripts/evaluate.py \
    --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
    --source-model Qwen/Qwen2.5-0.5B-Instruct \
    --target-model Qwen/Qwen3-0.6B \
    --eval-file data/gsm8k_eval_70.jsonl \
    --task-type generation \
    --device mps \
    --dtype float32 \
    --max-new-tokens 64 \
    --source-use-chat-template \
    --target-use-chat-template \
    --source-enable-thinking false \
    --target-enable-thinking false \
    --methods target t2t rotalign \
    --gate-mode checkpoint \
    --kv-transport k_only \
    --position-selection-ratio 0.5 \
    --position-selection-metric attention \
    --drop-target-layers "$layer" \
    --drop-target-layer-mode target
done

# Add-back from the 5-layer knockout: restore one shared layer at a time.
for layer in 27 5 23 22 8; do
  drop=$(printf '%s\n' 27 5 23 22 8 | grep -v "^${layer}$" | paste -sd, -)
  python scripts/evaluate.py \
    --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
    --source-model Qwen/Qwen2.5-0.5B-Instruct \
    --target-model Qwen/Qwen3-0.6B \
    --eval-file .debug/gsm8k_eval_10.jsonl \
    --task-type generation \
    --device mps \
    --dtype float32 \
    --max-new-tokens 64 \
    --source-use-chat-template \
    --target-use-chat-template \
    --source-enable-thinking false \
    --target-enable-thinking false \
    --methods target t2t rotalign \
    --gate-mode checkpoint \
    --kv-transport k_only \
    --position-selection-ratio 0.5 \
    --position-selection-metric attention \
    --drop-target-layers "$drop" \
    --drop-target-layer-mode target
done
```

What to read out:

- if removing a shared layer hurts more than a matched random layer, the layer
  signature is causal
- if the shared drop and random drop are similar, the signature is likely a
  budget artifact or a repeated selector preference

## 2) Per-layer causal drops on controlled GSM10 / GSM70

Goal: separate a true causal layer from a broad layer-budget effect.

What to test:

- `gsm8k_eval_70` and the fresh `.debug/gsm8k_eval_10.jsonl`
- `drop_target_layer_mode target` versus `drop_target_layer_mode zero`
- use the dynalign checkpoint and the ctxonly null side by side

Why this is reusable:

- the evaluator already supports `--drop-target-layers`
- the current telemetry already shows the same signature across weakly alive
  variants, so a causal drop curve is the right next diagnostic

Exact commands:

```bash
source .venv/bin/activate

for eval_file in .debug/gsm8k_eval_10.jsonl data/gsm8k_eval_70.jsonl; do
  for mode in target zero; do
    python scripts/evaluate.py \
      --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
      --source-model Qwen/Qwen2.5-0.5B-Instruct \
      --target-model Qwen/Qwen3-0.6B \
      --eval-file "$eval_file" \
      --task-type generation \
      --device mps \
      --dtype float32 \
      --max-new-tokens 64 \
      --source-use-chat-template \
      --target-use-chat-template \
      --source-enable-thinking false \
      --target-enable-thinking false \
      --methods target t2t rotalign \
      --gate-mode checkpoint \
      --kv-transport k_only \
      --position-selection-ratio 0.5 \
      --position-selection-metric attention \
      --drop-target-layers 27,5,23,22,8 \
      --drop-target-layer-mode "$mode"
  done
done
```

## 3) Teacher-signal attribution

Goal: determine whether dynalign’s smoke is real teacher signal.

What to compare:

- `bridge_ridge_qk_dynalign_ctxonly_module_replace` = matched null
- `bridge_ridge_qk_dynalign_module_replace` = teacher-aware branch
- `bridge_ridge_qk_dynalign_dwakd_module_replace` = stronger weighted teacher
- `bridge_ridge_qk_dynalign_interact_module_replace` = interaction teacher

Why this is reusable:

- all checkpoints already exist under `checkpoints/`
- the current telemetry already shows the key contrast:
  `dynalign = 0.4000` vs `dynalign_ctxonly = 0.2000`

Exact commands:

```bash
source .venv/bin/activate

for ckpt in \
  checkpoints/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_dwakd_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_dwakd_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.pt
do
  for eval_file in .debug/gsm8k_eval_10.jsonl data/gsm8k_5.jsonl data/gsm8k_eval_70.jsonl; do
    python scripts/evaluate.py \
      --translator "$ckpt" \
      --source-model Qwen/Qwen2.5-0.5B-Instruct \
      --target-model Qwen/Qwen3-0.6B \
      --eval-file "$eval_file" \
      --task-type generation \
      --device mps \
      --dtype float32 \
      --max-new-tokens 64 \
      --source-use-chat-template \
      --target-use-chat-template \
      --source-enable-thinking false \
      --target-enable-thinking false \
      --methods target t2t rotalign \
      --gate-mode checkpoint \
      --kv-transport k_only \
      --position-selection-ratio 0.5 \
      --position-selection-metric attention
  done
done
```

What to read out:

- if `dynalign` stays high while `ctxonly` collapses, the teacher signal is
  real
- if `dwakd` or `interact` lift the controlled GSM10 slice above `0.1000`,
  the teacher lane is still alive

## 4) Token span change audits

Goal: verify that byte-span / span-align differences are actually nontrivial
once the prompts stress tokenizer boundaries.

What to test:

- the default byte-stress prompts in `scripts/analyze_byte_alignment.py`
- the existing Qwen byte-stress audit prompts
- compare `bytespan` versus `spanalign`

Why this is reusable:

- the current telemetry says the original diagnostic slice changed `0` prompts
  versus char-span alignment
- the Qwen stress audit changed only `1 / 8`, so we need a stronger stress slice

Exact commands:

```bash
source .venv/bin/activate

mkdir -p .debug
cat > .debug/byte_stress_8.txt <<'EOF'
Compute 7½% of $1,234.56, then add 3 km in meters.
If a café sells 12 croissants at €2.50 each, what is the revenue?
Solve: α + β = 17, and β = 5. What is α?
Emoji check: 🧪 + 🧠 = one combined clue. What two objects are shown?
Chemistry tokenization: NaCl, H₂O, CO₂, and 10⁻³ mol/L.
Code stress: for i in range(3): total += nums[i]. What is loop count?
Multilingual stress: 東京 to Paris is written as Tokyo to Paris in English.
Units stress: 5 µs + 20 ms + 3 ns; which unit is largest?
EOF

python scripts/analyze_byte_alignment.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --prompt-file .debug/byte_stress_8.txt \
  --output-jsonl .debug/byte_stress_8.jsonl \
  --output-md .debug/byte_stress_8.md \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false
```

Optional if you want to refresh the byte-span / span-align fits on the same
stress slice:

```bash
python scripts/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file .debug/byte_stress_8.txt \
  --output checkpoints/byte_stress_spanalign.pt \
  --bits 4 \
  --alignment auto \
  --layer-pairing interp \
  --layer-selection-ratio 0.5 \
  --quantization-correction bridge_ridge_qk_spanalign_module_replace \
  --whitening \
  --target-whitening

python scripts/calibrate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file .debug/byte_stress_8.txt \
  --output checkpoints/byte_stress_bytespan.pt \
  --bits 4 \
  --alignment auto \
  --layer-pairing interp \
  --layer-selection-ratio 0.5 \
  --quantization-correction bridge_ridge_qk_bytespan_module_replace \
  --whitening \
  --target-whitening
```

## 5) Selector entropy / score-gap correlations

Goal: turn the repeated weak-layer signature into a reusable diagnostic, not
just a visual pattern.

What to measure:

- correlation of `mean_score_entropy` with `mean_score_gap`
- correlation of `mean_score_gap` with accuracy
- optionally, correlation of `selector_entropy_avg` with per-example
  correctness from the prediction JSONLs

Why this is reusable:

- the current `layer_localization_20260420.jsonl` already stores the relevant
  per-layer aggregates
- a small one-off analysis over the sidecar files is enough to make the
  relationship explicit

Exact command:

```bash
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("paper/layer_localization_20260420.jsonl").read_text().splitlines() if line.strip()]

def corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)

by_method = {}
for row in rows:
    by_method.setdefault(row["method"], []).append(row)

for method, items in sorted(by_method.items()):
    entropy = [float(r["mean_score_entropy"]) for r in items]
    gap = [float(r["mean_score_gap"]) for r in items]
    acc = [float(r["accuracy"]) for r in items]
    print(method)
    print("  corr(entropy, gap) =", corr(entropy, gap))
    print("  corr(gap, accuracy) =", corr(gap, acc))
    print("  corr(entropy, accuracy) =", corr(entropy, acc))
PY
```

## 6) Paired flip tables

Goal: keep the paper reviewer-facing even while the method remains negative.

What to rebuild:

- paired flip table
- bytes-vs-accuracy table
- full reviewer artifact bundle

Exact commands:

```bash
source .venv/bin/activate

python scripts/build_main_paired_table.py
python scripts/build_bytes_accuracy_table.py
python scripts/build_reviewer_artifacts.py
```

## Recommended order

1. teacher-signal attribution
2. single-layer leave-one-out / add-back
3. per-layer causal drops
4. token span change audits
5. selector entropy / score-gap correlations
6. paired flip refresh

The first three are the most likely to tell us whether the current smoke is a
real causal effect or just a repeated selector pattern.
