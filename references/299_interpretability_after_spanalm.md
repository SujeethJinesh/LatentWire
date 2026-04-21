# Interpretability After Span-ALM

Date: 2026-04-20

This note records what is now saturated after the dynalign likelihood / span-ALM
branches, and the next interpretable tests we still need before any positive-method
claim.

Grounding artifacts:

- `paper/bytes_accuracy_table_20260420.md`
- `paper/layer_localization_20260420.md`
- `paper/failure_taxonomy_20260419.md`
- `latent_bridge/current_readout_20260418.md`
- `results/layer_knockout_20260420/*`

Current saturation readout:

- exact Qwen GSM70 still centers on a narrow internal hierarchy:
  - `fixed prior = 0.0857`
  - `grouped subspace + rank-4 residual = 0.0571`
  - `C2C = 0.1286`
- the weak modular bridge family shares the same layer-localization signature:
  `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`
- the dynalign teacher ladder now saturates as a boundary story:
  - `dynalign = 0.4000` on `gsm8k_5`
  - `dynalign_ctxonly = 0.2000`
  - `dynalign_dwakd = 0.4000`
  - `dynalign_interact = 0.2000`
  - `dynalign_likelihood = 0.2000`
  - `dynalign_spanalm = 0.2000`
  - all of the above fall back to `0.1000` on the controlled `gsm8k_eval_10`
- the current knockout readout is still only a sanity check:
  - `top-5 layer knockout = 0.2000`
  - `offset-5 layer knockout = 0.2000`
- byte-span is still not yet a strong tokenizer-level signal:
  - current diagnostic span alignment changes `0` prompts vs `spanalign`
  - the byte-stress audit changes only `1 / 8` prompts

Primary interpretability refs:

- **AtP\***: An efficient and scalable method for localizing LLM behaviour to
  components, 2024-03-01
  https://arxiv.org/abs/2403.00745
- **Causal Head Gating**: A Framework for Interpreting Roles of Attention
  Heads in Transformers, 2025-05-19
  https://arxiv.org/abs/2505.13737
- **SCBench**: A KV Cache-Centric Analysis of Long-Context Methods, 2024-12-13
  https://arxiv.org/abs/2412.10319
- **Rethinking Key-Value Cache Compression Techniques for Large Language
  Model Serving**, 2025-03-31
  https://arxiv.org/abs/2503.24000

## 1) Teacher ablations

Goal: decide whether the dynalign smoke is real teacher signal or just a small
sample artifact that saturates once span/likelihood supervision is added.

Most useful comparison set:

- `bridge_ridge_qk_dynalign_ctxonly_module_replace`
- `bridge_ridge_qk_dynalign_module_replace`
- `bridge_ridge_qk_dynalign_dwakd_module_replace`
- `bridge_ridge_qk_dynalign_interact_module_replace`
- `bridge_ridge_qk_dynalign_likelihood_module_replace`
- `bridge_ridge_qk_dynalign_spanalm_module_replace`

What to report:

- `gsm8k_5`
- `gsm8k_eval_10`
- `gsm8k_eval_70`
- optionally the byte-stress slice after the next tokenization audit

Why this is the key next test:

- the current story says `dynalign` is live only as smoke, not as a controlled
  held-out win
- likelihood and span-ALM both now sit at the same controlled floor as the
  context-only null, so the teacher lane needs a cleaner falsifier

Exact commands:

```bash
source .venv/bin/activate

for ckpt in \
  checkpoints/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_dwakd_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_dwakd_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_interact_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_interact_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_likelihood_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_likelihood_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_spanalm_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_spanalm_module_replace_cal16_chat.pt
do
  for eval_file in data/gsm8k_5.jsonl .debug/gsm8k_eval_10.jsonl data/gsm8k_eval_70.jsonl; do
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

## 2) Single-layer leave-one-out / add-back

Goal: test whether the repeated layer signature is truly causal or only a
repeated selector preference.

Shared signature:

- `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`

Why this is reusable:

- the current 5-layer knockout and matched offset knockout both sit at `0.2000`
- the next useful step is a strict shared-vs-random comparison plus a one-layer
  leave-one-out/add-back sweep

Exact commands:

```bash
source .venv/bin/activate

# Leave one shared layer out at a time on the fair control slice.
for layer in 27 5 23 22 8; do
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
    --drop-target-layers "$layer" \
    --drop-target-layer-mode target
done

# Add back one shared layer at a time from the 5-layer knockout.
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

Readout rule:

- if the shared-layer knockout hurts much more than a matched random knockout,
  the layer signature is causal
- if the shared and random knockouts look the same, the signature is probably a
  budget artifact or selector saturation

## 3) Selector entropy / score-gap correlations

Goal: decide whether the recurring layer signature reflects a sharp selection
distribution or just a broad budget allocation.

What to compute:

- per-layer `score_entropy`
- per-layer `score_gap`
- per-layer `keep_fraction`
- correlation of those quantities with correctness and with the repeated
  top-layer signature

Why this is useful:

- the current telemetry already stores these values in the raw `selector_trace`
  records
- a strong entropy/gap signal would support a real localized selector rather
  than a diffuse floor effect

Exact one-liner:

```bash
source .venv/bin/activate
python - <<'PY'
import json, glob, math
from collections import defaultdict

rows = []
for path in glob.glob("results/layer_knockout_20260420/*.jsonl"):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            for tr in ex.get("selector_trace", []):
                rows.append({
                    "method": ex.get("method"),
                    "correct": bool(ex.get("correct")),
                    "layer": tr.get("target_layer"),
                    "entropy": tr.get("score_entropy"),
                    "gap": tr.get("score_gap"),
                    "keep": tr.get("keep_fraction"),
                })

by_method = defaultdict(list)
for r in rows:
    by_method[r["method"]].append(r)

for method, arr in sorted(by_method.items()):
    ent = [r["entropy"] for r in arr if r["entropy"] is not None]
    gap = [r["gap"] for r in arr if r["gap"] is not None]
    keep = [r["keep"] for r in arr if r["keep"] is not None]
    print(method, "n=", len(arr), "entropy_mean=", sum(ent)/len(ent), "gap_mean=", sum(gap)/len(gap), "keep_mean=", sum(keep)/len(keep))
PY
```

## 4) Per-example flips

Goal: separate aggregate accuracy from real example-level wins.

What to refresh:

- paired-flip table on the exact same examples
- method-only wins versus null-only wins
- McNemar and bootstrap intervals on the paired examples

Why this is reusable:

- it tells us whether any branch is truly better or just winning a different
  subset of examples

Recommended refresh commands:

```bash
source .venv/bin/activate
python scripts/build_main_paired_table.py
python scripts/build_bytes_accuracy_table.py
python scripts/build_reviewer_artifacts.py
```

Readout rule:

- if a branch wins only a few examples and the paired interval crosses zero, it
  is smoke, not a positive method
- if it wins on the same paired examples across teacher ablations and survives
  the nulls, then it is a real signal

## 5) Target-side refinement signals

Goal: rule out the possibility that the observed movement is only a target-side
refinement or a target perturbation artifact.

What to compare:

- `target-alone`
- `text-to-text`
- `bridge_ridge`
- `grouped subspace + rank-4 residual`
- `KVPress no-press`
- `KVPress ExpectedAttentionPress`
- `drop-target-layer-mode target` versus `drop-target-layer-mode zero`

Why this matters:

- the current frontier shows the best transport-plus-correction lane is still
  below the fixed prior and well below C2C
- if a branch only helps when the target side is perturbed, it is not a clean
  cross-model communication result

Exact commands:

```bash
source .venv/bin/activate

# Target-side perturbation control on the shared signature.
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
  --drop-target-layers 27,5,23,22,8 \
  --drop-target-layer-mode zero
```

Optional byte-stress audit:

```bash
source .venv/bin/activate
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

## Bottom line

After span-ALM / dynalign likelihood, the story is now saturated in the sense
that:

- teacher-side tweaks mostly tie the same controlled floor
- the same modular branches repeat the same layer signature
- the current layer knockout is still only a sanity check
- byte-span still needs tokenizer stress before it is a meaningful signal

The next positive-method claim needs all of the following to move:

- dynalign teacher ablations that separate real teacher signal from a null
- a stricter shared-layer leave-one-out/add-back sweep
- selector entropy / score-gap correlations
- paired-example flip tables
- target-side refinement controls that prove the gain is not just a target
  perturbation artifact
