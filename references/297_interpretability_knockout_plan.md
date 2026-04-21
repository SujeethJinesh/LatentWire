# Interpretability / Knockout Plan

Date: 2026-04-20

This note records the next cheap, decisive ablations suggested by the current
telemetry:

- `paper/bytes_accuracy_table_20260420.md`
- `paper/layer_localization_20260420.md`
- `paper/failure_taxonomy_20260419.md`
- `results/layer_knockout_20260420/*`

Current readout:

- `fixed prior` on exact Qwen GSM70 is `0.0857`
- `C2C` is `0.1286`
- `grouped subspace + rank-4 residual` is `0.0571`
- the weak modular bridge family all shares the same layer signature:
  `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`
- `dynalign` is the only clear smoke branch (`0.4000` on `gsm8k_5`) but it
  only falls back to the controlled floor (`0.1000`) on `gsm8k_eval_10`
- current top-5 and offset-5 layer knockouts both land at `0.2000`, so the
  current knockout is a sanity check, not a falsifier

## References

- **AtP\***: An efficient and scalable method for localizing LLM behaviour to
  components, 2024-03-01
  https://arxiv.org/abs/2403.00745
- **Causal Head Gating**: A Framework for Interpreting Roles of Attention
  Heads in Transformers, 2025-05-19
  https://arxiv.org/abs/2505.13737
- **SCBench**: A KV Cache-Centric Analysis of Long-Context Methods,
  2024-12-13
  https://arxiv.org/abs/2412.10319
- **Rethinking Key-Value Cache Compression Techniques for Large Language
  Model Serving**, 2025-03-31
  https://arxiv.org/abs/2503.24000

## 1) Dynalign teacher-null / teacher-strength contrast

Goal: determine whether the `0.4000` dynalign smoke is actually teacher signal
or just a small-sample artifact.

Most useful currently runnable proxy:

- `bridge_ridge_qk_dynalign_ctxonly_module_replace` as the matched null
- `bridge_ridge_qk_dynalign_module_replace` as the current teacher-aware branch
- `bridge_ridge_qk_dynalign_dwakd_module_replace` as a stronger teacher-weighted
  branch
- `bridge_ridge_qk_dynalign_interact_module_replace` as an interaction-teacher
  variant

What to report:

- `gsm8k_5`
- `gsm8k_eval_10`
- `gsm8k_eval_70`
- optional byte-stress slice after the next byte audit

Why this is cheap and decisive:

- all four checkpoints already exist under `checkpoints/`
- the current telemetry already shows the relevant contrast:
  `dynalign = 0.4000` vs `dynalign_ctxonly = 0.2000`

Recommended commands:

```bash
source .venv/bin/activate

python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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

python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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

python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_dwakd_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_dwakd_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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
```

Interpretation:

- if `dynalign` drops to the `ctxonly` floor under a stricter held-out slice,
  the smoke is probably real but brittle
- if `dynalign` stays high while `ctxonly` collapses, the teacher signal is
  real
- if `dwakd` or `interact` lift the fair `gsm8k_eval_10` slice above the
  `0.1000` floor, the teacher lane is still alive

## 2) Shared-layer knockout falsifier

Goal: test whether the repeated layer signature is genuinely causal.

Current signature:

- `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`

Current result:

- `top-5 layer knockout = 0.2000`
- `offset-5 layer knockout = 0.2000`

Recommended next step:

- drop the shared 5-layer signature with `--drop-target-layer-mode target`
- compare against the matched offset-5 knockout and a random same-count set
- then do a single-layer leave-one-out sweep on `27`, `5`, `23`, `22`, `8`

Why this is cheap and decisive:

- the evaluator already supports `--drop-target-layers` and
  `--drop-target-layer-mode`
- no code changes are required
- the current top-5 vs offset-5 tie means the next useful test is a stricter
  shared-vs-random comparison, not another broad sweep

Recommended commands:

```bash
source .venv/bin/activate

python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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
  --drop-target-layer-mode target

python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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
  --drop-target-layers 26,4,21,20,7 \
  --drop-target-layer-mode target
```

Optional stronger falsifier:

```bash
python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
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
  --drop-target-layers 27 \
  --drop-target-layer-mode zero
```

Interpretation:

- if the shared 5-layer knockout hurts much more than the offset-5 or random
  knockout, the layer signature is causal
- if all of them tie, the current signature is likely a reporting artifact or a
  budget artifact

## 3) Byte-stress calibration / byte-span vs spanalign

Goal: decide whether byte-level span remapping actually changes pairings once
the calibration data stresses tokenizer boundaries.

Current evidence:

- on the 16-prompt diagnostic slice, `bytespan` changed `0` prompts versus
  char-span `spanalign`
- on the new Qwen byte-stress audit, only `1 / 8` prompts changed

Recommended next step:

- build a tiny 8-prompt byte-stress calibration file from the default stress
  prompts in `scripts/analyze_byte_alignment.py`
- re-fit `bridge_ridge_qk_bytespan_module_replace` and
  `bridge_ridge_qk_spanalign_module_replace`
- compare their held-out GSM behavior and the `changed_vs_spanalign_prompts`
  count

Why this is cheap and decisive:

- the byte-stress prompts already exist in the repo as defaults
- the current telemetry says the current calibration slices are too weak to
  distinguish byte-span from span-align

Recommended commands:

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

Interpretation:

- if byte-span still changes almost no prompts, it is a control, not a live
  remedy
- if the stress slice increases pair changes materially, byte-level remapping
  is a real upstream effect and can be stacked after the current transport

## Recommended order

1. dynalign teacher-null / teacher-strength contrast
2. shared-layer knockout vs matched random layers
3. byte-stress calibration

The first two are the decisive interpretable falsifiers. The third is the best
cheap check on whether the tokenizer-side remapping lane is actually doing
anything.
