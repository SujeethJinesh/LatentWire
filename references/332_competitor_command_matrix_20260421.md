# Competitor Command Matrix: Exact Qwen Pair

Date: 2026-04-21

Exact pair: `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B`

Scope: concrete command snippets and blocker checks for the five external families we keep reusing in paper-safe tables.

## Row Classification

| Row | Role in paper | Direct cross-model competitor? | Table control? | Runs on this machine without CUDA? | Needs a clean clone for repo-native validation? | Exact-pair status |
| --- | --- | --- | --- | --- | --- | --- |
| `C2C` | direct semantic communication baseline | yes | no | yes for manifest refresh; native GSM smoke usually wants CUDA | yes for repo-native smoke, but the local manifest wrapper does not need it | exact pair supported and already resolved locally |
| `KVComm` | direct communication baseline / ported control | yes | no | yes via our MPS wrapper path | yes if you want pristine upstream repo-native validation; our wrapper patches vendored code | exact pair supported through our ported harness |
| `KVPress` | same-model cache compression control | no | yes | yes via our wrapper on MPS | no for the wrapper; yes if you want pristine upstream repo-native execution | exact pair is not a communication claim; target-model control only |
| `KVzip` | same-model cache compression control | no | yes | no, upstream path is CUDA/flash-attn oriented | yes | exact pair is only meaningful as a target-model control |
| `Quest` | query-aware sparsity control | no | yes | no, upstream path is CUDA/flash-attn oriented | yes | exact Qwen pair is not supported upstream without a port |

## Blocker Checks

- `C2C`: verify the published Qwen fuser resolves locally and that the native GSM smoke still matches the same parser as our LatentWire table.
- `KVComm`: verify the calibration sweep does not collapse to one layer set and that the dirty vendored patch still isolates cleanly from the upstream repo.
- `KVPress`: verify `none` remains the same-model control floor and `expected_attention` stays a compression control, not a communication win.
- `KVzip`: verify the abbreviated model name `qwen3-0.6b` resolves to `Qwen/Qwen3-0.6B` and that flash-attn/CUDA are present before any real run.
- `Quest`: exact Qwen-pair benchmarking is blocked upstream. Do not force it into the paper table without a port or a new model-family-aligned control.

## Command Snippets

### 1) C2C manifest refresh on the exact pair

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --output-json results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json
```

### 2) KVPress same-model control floor: `none`

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress:/Users/sujeethjinesh/Desktop/LatentWire \
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_none_gsm30.jsonl
```

### 3) KVPress same-model control floor: `expected_attention`

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
PYTHONPATH=/Users/sujeethjinesh/Desktop/LatentWire/references/repos/kvpress:/Users/sujeethjinesh/Desktop/LatentWire \
./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm30.jsonl
```

### 4) KVComm exact-pair ported smoke

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/run_kvcomm_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file data/gsm8k_eval_70.jsonl \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25,0.5,0.75,1.0 \
  --prediction-output results/competitor_bootstrap_20260421/kvcomm_qwen_gsm30_ported.jsonl
```

### 5) KVzip target-model compression control on Qwen3

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
../../venv_arm64/bin/python eval.py \
  -m qwen3-0.6b \
  -d gsm \
  --idx 0 \
  --num 5 \
  --tag latentwire_qwen3_gsm5
```

### 6) Quest blocker check for the exact Qwen pair

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest
grep -nE 'Llama-3.1|Mistral-v0.3|flash-attn|CUDA' README.md
```

## What To Put In The Table

- Put `C2C` and `KVComm` in the direct cross-model competitor block.
- Put `KVPress`, `KVzip`, and `Quest` in the table-control block.
- Keep `Quest` out of the exact-Qwen-pair leaderboard unless we first port it to the pair and remove the CUDA/runtime blocker.
- Keep the `KVPress` rows labeled as same-model compression controls, not communication wins.

## Source Anchors

- C2C: https://github.com/thu-nics/C2C, https://arxiv.org/abs/2510.03215
- KVComm: https://github.com/FastMAS/KVCOMM, https://arxiv.org/abs/2510.12872
- KVPress: https://github.com/NVIDIA/kvpress, https://arxiv.org/abs/2510.00636
- KVzip: https://github.com/snu-mllab/KVzip, https://arxiv.org/abs/2505.23416
- Quest: https://github.com/mit-han-lab/Quest, https://arxiv.org/abs/2406.10774
