# Competitor Execution Plan: Paper-Safe Baselines

Date: 2026-04-21

Scope: C2C, KVComm, kvpress, Quest, and KVzip. This plan inspects the local clones under `references/repos/` and gives commands that do not require editing competitor repos. It deliberately separates direct cross-model communication competitors from same-model cache-compression controls.

## Bottom Line

The paper-safe comparison table should have three blocks, not one merged leaderboard.

| Block | Repos | Claim it tests | Paper-safe status |
| --- | --- | --- | --- |
| Direct cross-model communication | C2C, KVComm | Whether latent/KV communication beats text or no-communication baselines across model instances | C2C is the closest GSM-capable competitor; KVComm is direct communication but not GSM-native |
| Same-model cache compression | kvpress, KVzip, Quest | Whether query-aware/query-agnostic cache reduction explains our gains | Useful controls, not direct cross-model communication competitors |
| Our matched LatentWire run | `latent_bridge/evaluate.py` | Same source/target pair, same GSM split, same decoding/scoring, byte accounting | Required anchor for any claim about our method |

Current repository state:

| Repo | Path | Dirty? | Primary source | Notes |
| --- | --- | --- | --- | --- |
| C2C | `references/repos/C2C` | clean | https://github.com/thu-nics/C2C, https://arxiv.org/abs/2510.03215 | Best direct peer; README has Qwen2.5 -> Qwen3 pretrained fusers on Hugging Face. |
| KVComm | `references/repos/KVComm` | dirty: `model_attn.py`, `models.py` | https://github.com/Zephyroam/KVComm, https://arxiv.org/abs/2510.12872 | Direct communication peer, but native tasks are HotpotQA/Qasper/MuSiQue/etc., not GSM. Do not run until dirty diff is isolated. |
| kvpress | `references/repos/kvpress` | dirty: `kvpress/pipeline.py`, `kvpress/presses/base_press.py`, `kvpress/utils.py` | https://github.com/NVIDIA/kvpress, https://arxiv.org/abs/2510.00636 | Our wrapper `scripts/run_kvpress_eval.py` patches compatibility without editing repo files. Use wrapper for GSM5/GSM10. |
| Quest | `references/repos/Quest` | clean | https://github.com/mit-han-lab/Quest, https://arxiv.org/abs/2406.10774 | Query-aware long-context sparsity control. Not GSM-native. CUDA/flash-attn oriented. |
| KVzip | `references/repos/KVzip` | clean | https://github.com/snu-mllab/KVzip, https://arxiv.org/abs/2505.23416 | Query-agnostic cache compression with native `gsm` data loader, but examples assume CUDA/flash-attn and larger models. |

## Fairness Rules

- Use the same `data/gsm8k_5.jsonl` and `/tmp/gsm8k_eval_10.jsonl` slices for LatentWire and kvpress wrapper runs.
- Use greedy decoding, `max_new_tokens=64`, Qwen3 target thinking disabled, and the same answer extractor for LatentWire and kvpress wrapper GSM runs.
- For C2C, use native `gsm8k` only as a direct-competitor smoke if the published fuser and native evaluator are available. Do not compare C2C byte counts unless we explicitly normalize payload size, fuser checkpoint size, and selector metadata.
- For KVComm and Quest, do not call the result “GSM” unless we implement and validate a native GSM adapter. Their paper-safe smoke is `limit=5/10` on their supported tasks.
- For KVzip, GSM is native through `-d gsm`, but it is same-model cache compression. It should be a compression-control row, not a cross-model communication row.
- Report every run with model pair, task split, prompt template, decoding budget, compression/selection ratio, payload bytes if available, wall-clock latency, examples/sec, tokens/sec, and exact command.
- Do not mutate dirty competitor clones. If we need a clean KVComm or kvpress repo-native run, clone a fresh copy under a new ignored scratch directory or use our wrapper.

## Shared GSM Slices

Create the GSM10 slice once:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
head -n 10 data/gsm8k_eval_70.jsonl > /tmp/gsm8k_eval_10.jsonl
```

GSM5 path:

```bash
/Users/sujeethjinesh/Desktop/LatentWire/data/gsm8k_5.jsonl
```

GSM10 path:

```bash
/tmp/gsm8k_eval_10.jsonl
```

## Required LatentWire Anchor Runs

Target-alone GSM5:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
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
  --source-reasoning-mode brief_analysis \
  --methods target \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --prediction-output results/competitor_bootstrap_20260421/latentwire_target_gsm5.jsonl
```

Target-alone GSM10:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file /tmp/gsm8k_eval_10.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --source-reasoning-mode brief_analysis \
  --methods target \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --prediction-output results/competitor_bootstrap_20260421/latentwire_target_gsm10.jsonl
```

Current best LatentWire-style K-only route-atom GSM5:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
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
  --source-reasoning-mode brief_analysis \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --runtime-head-selection-ratio 0.25 \
  --runtime-head-selection-metric headwise_route_atom \
  --prediction-output results/competitor_bootstrap_20260421/latentwire_route_atom_gsm5.jsonl
```

Current best LatentWire-style K-only route-atom GSM10:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file /tmp/gsm8k_eval_10.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --source-reasoning-mode brief_analysis \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.10 \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --runtime-head-selection-ratio 0.25 \
  --runtime-head-selection-metric headwise_route_atom \
  --prediction-output results/competitor_bootstrap_20260421/latentwire_route_atom_gsm10.jsonl
```

Interpretability fields to preserve from these sidecars: `avg_bits`, `avg_bytes`, `payload_bits_avg`, `metadata_bits_avg`, `selector_keep_fraction_avg`, `selector_entropy_avg`, `head_keep_fraction_avg`, `route_atom_keep_fraction_avg`, `route_atom_score_entropy_avg`, `route_atom_score_gap_avg`, and `route_atom_js_divergence_mean_avg`.

## C2C

C2C is the most relevant direct competitor because it explicitly maps/fuses a source model KV cache into a target model. Local wrapper state:

- `latent_bridge.baselines.C2CAdapter` knows published fuser subdirs.
- `scripts/bootstrap_c2c.py` creates a manifest without downloading by default.
- Existing manifest: `results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json`.

No-download manifest refresh:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --output-json results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json
```

Optional fuser download, only on a networked evaluation box:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --download \
  --output-json results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b_downloaded.json
```

Native C2C GSM5 smoke, CUDA box only after the fuser is present:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
cat > /tmp/c2c_gsm5.yaml <<'YAML'
model:
  model_name: Rosetta
  rosetta_config:
    base_model: Qwen/Qwen3-0.6B
    teacher_model: Qwen/Qwen2.5-0.5B-Instruct
    is_do_alignment: false
    alignment_strategy: "longest"
    checkpoints_dir: /PATH/TO/qwen3_0.6b+qwen2.5_0.5b_Fuser/final
  generation_config:
    do_sample: false
    max_new_tokens: 64
output:
  output_dir: /Users/sujeethjinesh/Desktop/LatentWire/results/competitor_bootstrap_20260421/c2c_gsm5_native
eval:
  dataset: gsm8k
  gpu_ids: [0]
  answer_method: generate
  use_cot: false
  use_template: true
  sample_interval: 1
  limit: 5
  math_grading_method: "comprehensive"
YAML
python script/evaluation/unified_evaluator.py --config /tmp/c2c_gsm5.yaml
```

Native C2C GSM10 smoke, CUDA box only:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/C2C
python - <<'PY'
from pathlib import Path
p = Path('/tmp/c2c_gsm5.yaml')
text = p.read_text()
text = text.replace('c2c_gsm5_native', 'c2c_gsm10_native').replace('limit: 5', 'limit: 10')
Path('/tmp/c2c_gsm10.yaml').write_text(text)
PY
python script/evaluation/unified_evaluator.py --config /tmp/c2c_gsm10.yaml
```

Blockers:

- Native C2C reads Hugging Face `gsm8k`, not our local JSONL. That is acceptable for smoke only, not for the final paired table unless we port or wrap the evaluator.
- Published fuser checkpoint download is a model artifact download and should not be done in this lightweight planning step.
- Byte parity with LatentWire is not automatic because C2C has a trained fuser and different communication semantics.

## kvpress

kvpress is a same-model cache-compression control. Use our wrapper because the vendored clone is dirty and the wrapper applies compatibility patches without modifying repo files.

GSM5 no-press:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_no_press_gsm5.jsonl
```

GSM5 expected-attention:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_5.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm5.jsonl
```

GSM10 no-press:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file /tmp/gsm8k_eval_10.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_no_press_gsm10.jsonl
```

GSM10 expected-attention:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file /tmp/gsm8k_eval_10.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm10.jsonl
```

Blockers:

- This is same-model compression; it cannot validate cross-model semantic transport by itself.
- Existing older smokes under `results/kvpress_expected_20260420/` are usable sanity checks, but the plan above writes into the requested bootstrap folder.
- If `ExpectedAttentionPress` fails on `mps`, repeat on CUDA before making negative claims.

## KVzip

KVzip is a query-agnostic compression/reconstruction control. It has a native `gsm` data loader, so GSM5/GSM10 are possible in repo-native terms, but the run is likely CUDA-heavy.

GSM5 native smoke, CUDA box:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B test.py \
  -m qwen2.5-7b \
  -d gsm \
  --kv_type retain \
  --ratio 0.6 \
  --num 5 \
  --tag latentwire_gsm5_retain06
```

GSM10 native smoke, CUDA box:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B test.py \
  -m qwen2.5-7b \
  -d gsm \
  --kv_type retain \
  --ratio 0.6 \
  --num 10 \
  --tag latentwire_gsm10_retain06
```

Optional compression sweep after the smoke passes:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B eval.py \
  -m qwen2.5-7b \
  -d gsm \
  --kv_type retain \
  --num 100 \
  --tag latentwire_gsm100_sweep
```

Blockers:

- README requires CUDA 12.1-style setup and `flash-attn==2.7.4.post1`.
- Default model aliases point to 7B-scale checkpoints; do not run on the local Mac unless the model is already cached and the attention path works.
- Compare as compression-control only. KVzip does not solve source-to-target model alignment.

## KVComm

KVComm is a direct communication peer, but it is not GSM-native. The paper-safe smoke is a supported QA task with `limit=5/10`; a GSM result would require a new adapter and validation.

HotpotQA baseline limit-5, CUDA box or clean isolated env:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 5 \
  --use_wandb false
```

HotpotQA KVComm limit-5:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 5 \
  --use_wandb false
```

HotpotQA baseline/KVComm limit-10:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 10 \
  --use_wandb false

python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 10 \
  --use_wandb false
```

Blockers:

- Local repo is dirty in model internals. Before running, either inspect and bless the local patch or clone a clean copy elsewhere.
- README pins `transformers==4.53.3`; do not mix with the LatentWire env without a separate venv.
- `com.py` defaults to `torch.bfloat16`, `attn_implementation="sdpa"`, and CUDA-style device mapping.
- No native GSM adapter exists in this clone. A GSM port would be new code, not a baseline reproduction.

## Quest

Quest is a query-aware sparsity/control baseline, not a direct communication competitor. It should be used for long-context sparsity interpretation, not GSM claims.

Passkey limit-style smoke by token budget, CUDA box:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
bash scripts/passkey.sh
```

LongBench query-aware smoke, CUDA box:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/evaluation/LongBench
python -u pred.py \
  --model longchat-v1.5-7b-32k \
  --task qasper
python -u pred.py \
  --model longchat-v1.5-7b-32k \
  --task qasper \
  --quest \
  --token_budget 2048 \
  --chunk_size 16
python -u eval.py \
  --model longchat-v1.5-7b-32k
```

Blockers:

- No GSM-native path. Do not report a GSM Quest score without a separate validated port.
- README requires Python 3.10, flash-attn, CMake/operator build pieces, and CUDA-oriented models.
- Use Quest only to answer whether query-aware sparsity is a sufficient control for our route selection story.

## Reporting Schema

Every paper-safe run should emit or be converted into this manifest schema:

```json
{
  "run_id": "short_stable_name",
  "repo": "c2c|kvcomm|kvpress|quest|kvzip|latentwire",
  "repo_path": "references/repos/...",
  "git_status": "clean|dirty|external_wrapper",
  "task": "gsm8k|hotpotqa|qasper|passkey",
  "split_or_limit": "gsm5|gsm10|limit5|limit10",
  "model_a_or_source": "...",
  "model_b_or_target": "...",
  "method": "...",
  "decoding": {"do_sample": false, "max_new_tokens": 64},
  "compression_or_selection": {"ratio": 0.5, "metric": "attention"},
  "metrics": {"accuracy": null, "latency_sec": null, "tokens_per_sec": null},
  "communication": {"payload_bytes": null, "metadata_bytes": null, "total_bytes": null},
  "command": "exact shell command",
  "paper_safe_comparison_group": "direct_cross_model|same_model_compression|long_context_sparsity|latentwire_anchor",
  "blockers": []
}
```

## Execution Order

1. Run LatentWire target and current best method on GSM5/GSM10.
2. Run kvpress wrapper `none` and `expected_attention` on GSM5/GSM10. This is the lowest-risk local competitor/control.
3. Refresh the C2C manifest without downloading. If a CUDA box is available, download the fuser and run native C2C GSM5/GSM10.
4. Run KVzip GSM5/GSM10 only on CUDA; keep it in the compression-control table.
5. Defer KVComm until the dirty local modifications are isolated. Run HotpotQA limit-5/10, not GSM.
6. Defer Quest to long-context interpretation. Do not include it in the GSM table.

## Current Blockers To Resolve Before Paper Claims

- C2C needs a final apples-to-apples local JSONL harness or an explicit note that native GSM8K is a smoke only.
- KVComm and kvpress vendored clones contain local edits; do not present repo-native runs from those trees unless the diffs are documented.
- KVzip and Quest are likely CUDA-only for faithful reproduction.
- The final table must separate “beats target-alone on GSM” from “beats cache-compression control” and “beats direct communication competitor.” These are distinct claims.
