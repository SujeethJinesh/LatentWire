# Telemetry Blocker Synthesis

Date: 2026-04-20

This note summarizes what is saturated after `dynalign_dwainteract`, which
evidence is already clean enough to keep in paper tables, and which next
ablation would most directly falsify a different hypothesis.

Grounding artifacts:

- `paper/failure_taxonomy_20260419.md`
- `paper/bytes_accuracy_table_20260420.md`
- `paper/layer_localization_20260420.md`
- `paper/paired_flip_table_20260420.md`
- `latent_bridge/current_readout_20260418.md`
- `results/layer_knockout_20260420/*`

## 1) What is saturated now

The strongest current saturation signal is the **dynalign teacher stack**.
After `dynalign_dwainteract`, the live branch closes back to the same weak
floor as the other local teachers:

- `dynalign = 0.4000` on `gsm8k_5`, but only `0.1000` on controlled
  `gsm8k_eval_10`
- `dynalign_dwakd = 0.4000` on `gsm8k_5`, but only `0.1000` on controlled
  `gsm8k_eval_10`
- `dynalign_dwainteract = 0.2000` on `gsm8k_5`, `0.1000` on controlled
  `gsm8k_eval_10`
- `dynalign_interact = 0.2000`
- `dynalign_likelihood = 0.2000`
- `dynalign_spanalm = 0.2000`
- `dynalign_ctxonly = 0.2000`
- `bytespan_module_replace = 0.2000`

That means the current local-teacher branch is saturated: interaction,
likelihood, span-ALM, byte-span, and context-only all land on the same
`gsm8k_5 = 0.2000` smoke / `gsm8k_eval_10 = 0.1000` controlled floor pattern.

The second saturation signal is the **shared-layer signature**. The weakly
alive modular family all concentrates on the same top target layers:

- `L27<-S23, L5<-S4, L23<-S20, L22<-S19, L8<-S7`

That signature repeats across:

- `bytespan_module_replace`
- `dynalign_ctxonly_module_replace`
- `dynalign_dwainteract_module_replace`
- `dynalign_dwakd_module_replace`
- `dynalign_interact_module_replace`
- `dynalign_likelihood_module_replace`
- `dynalign_module_replace`
- `dynalign_spanalm_module_replace`
- `module_adapter`
- `module_replace`
- `shared_plus_private_asym_adapter`
- `shared_plus_private_dynmap_adapter`
- `spanalign_module_replace`
- `tokenbasis_replace`
- `xattn_adapter`
- `xattn_dynmap_adapter`

The third saturation signal is that **the current knockout is only a sanity
check**:

- `dynalign_top5_layerdrop = 0.2000`
- `dynalign_offset5_layerdrop = 0.2000`

So the current five-layer knockout does not yet falsify a causal layer
signature; it only shows broad budget sensitivity.

The fourth saturation signal is that **token/byte remapping still needs
stress**:

- on the current GSM diagnostic slice, `bytespan` changes `0` prompts versus
  `spanalign`
- the byte-stress audit changes only `1 / 8` prompts

Finally, the external query-aware baseline is also a boundary, not a win:

- exact `KVPress ExpectedAttentionPress` ties exact `no-press` on this pair
- in-repo `attention_expected` also ties its shuffled null

Current paper-facing interpretation:

- the teacher lane is saturated
- the shared modular bridge family is saturated
- query-aware selector overlays are not enough
- the only remaining internal positive clue is still the pair-conditioned
  fixed sparse head prior, but it is not stable enough to support a positive
  method claim yet

## 2) What is already interpretable enough for paper tables

These are the artifacts I would keep in the paper now because they are
pairwise, budget-matched, and already separated by strong nulls:

1. **Bytes / accuracy frontier**
   - `fixed prior = 0.0857`
   - `grouped subspace + rank-4 residual = 0.0571`
   - `C2C = 0.1286`
   - `KVPress ExpectedAttentionPress = 0.1000`
   - `KVPress no-press = 0.1000`

2. **Paired flip table**
   - the same-example wins/losses are already recorded
   - most dynalign variants tie `target-alone` on the controlled slice
   - the paired deltas make clear that the teacher stack is not converting
     smoke into a stable held-out gain

3. **Layer localization table / heatmap**
   - the repeated `L27, L5, L23, L22, L8` signature is stable across weak
     modular branches
   - the current top layer has mean keep fraction `0.4992`, score top
     `0.9140`, and score gap `0.8785`

4. **Teacher ablation table**
   - `dynalign`, `dynalign_dwakd`, `dynalign_dwainteract`, `dynalign_interact`,
     `dynalign_likelihood`, `dynalign_spanalm`, `dynalign_ctxonly`
   - this is the cleanest table for showing that local teacher variants are
     saturated at the same floor

5. **Target-side refinement table**
   - `drop-target-layer-mode target` versus `zero`
   - matched shared-layer knockout versus offset knockout
   - this is enough to show that current gains are not just target perturbation

## 3) The next ablation that falsifies a different hypothesis

If the goal is to falsify a **different** hypothesis from the teacher story,
the next runnable one is:

> **Shared-layer leave-one-out / add-back against a matched random knockout**

Why this one:

- the current layer-localization telemetry says the weakly alive modular family
  repeatedly selects the same five layers
- but the existing top-5 versus offset-5 knockout tie means the current
  evidence is not yet causal
- a one-layer leave-one-out/add-back curve on `L27, L5, L23, L22, L8`,
  compared against a matched random same-count knockout, would tell us whether
  the signature is a real causal circuit or just repeated selector saturation

What it would falsify:

- if shared-layer knockouts hurt more than matched random knockouts, the fixed
  causal-subspace hypothesis gets stronger
- if shared and random knockouts look the same, the layer signature is mostly
  a budget artifact or selector preference, not a causal circuit

If you instead want to falsify the **teacher-signal** hypothesis itself, the
current no-code path is already the right direction, but a true teacher-null
shuffle would require a new switch in the evaluator. Under the no-core-code
constraint, the layer falsifier is the most actionable next ablation.

## 4) Exact commands to run next

### 4.1 Teacher ladder refresh

```bash
source .venv/bin/activate

for ckpt in \
  checkpoints/bridge_ridge_qk_dynalign_ctxonly_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_ctxonly_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_module_replace_20260420/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_module_replace_cal64_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_dwakd_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_dwakd_module_replace_cal16_chat.pt \
  checkpoints/bridge_ridge_qk_dynalign_dwainteract_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_dwainteract_module_replace_cal16_chat.pt \
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

### 4.2 Shared-layer leave-one-out / add-back

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

To do the one-layer sweep, repeat the first command with
`--drop-target-layers 27`, `5`, `23`, `22`, `8` one at a time, and compare
to the same-count random knockout.

### 4.3 Target-side refinement control

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
  --drop-target-layers 27,5,23,22,8 \
  --drop-target-layer-mode zero
```

### 4.4 Rebuild paper tables

```bash
source .venv/bin/activate
python scripts/build_main_paired_table.py
python scripts/build_bytes_accuracy_table.py
python scripts/build_reviewer_artifacts.py
```

### 4.5 Byte-stress audit

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

After `dynalign_dwainteract`, the local teacher route is saturated, the
modular bridge family repeats the same layer signature, and byte-span still
needs tokenizer stress before it is a real signal. The next table-ready story
is therefore a blocker story, not a positive-method story:

- teacher ladder: saturated
- paired flips: interpretable
- bytes frontier: interpretable
- layer localization: interpretable but not causal yet
- shared-layer leave-one-out/add-back: the next decisive falsifier
