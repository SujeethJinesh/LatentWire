# Competitor Smoke Matrix

Date: 2026-04-21

Scope: `C2C`, `KVComm`, `kvpress`, `KVzip`, `Quest`.

This note is a smoke-run map, not a final table. Keep GSM-capable baselines separate from non-GSM fallbacks.

## Split Legend

- `GSM5` = `data/gsm8k_5.jsonl`
- `GSM10` = `/tmp/gsm8k_eval_10.jsonl`
- `GSM30` = `data/gsm8k_gate_search_30.jsonl`
- `GSM30` is the local 30-example control slice used in current writeups; it is not a held-out eval split.
- `C2C` native evaluation uses the repo's `gsm8k` loader, not the local JSONL slices above.

Create `GSM10` once if needed:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
head -n 10 data/gsm8k_eval_70.jsonl > /tmp/gsm8k_eval_10.jsonl
```

## C2C

Primary sources:
- `references/repos/C2C/README.md`
- `references/repos/C2C/script/evaluation/unified_evaluator.py`
- `https://github.com/thu-nics/C2C`
- `https://arxiv.org/abs/2510.03215`

Next runnable now, without downloads:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python scripts/bootstrap_c2c.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --output-json results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json
```

Native smoke after the published fuser checkpoint is available. The only change across the three runs is `limit` and `output_dir`:

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

python - <<'PY'
from pathlib import Path
p = Path('/tmp/c2c_gsm5.yaml')
text = p.read_text()
text = text.replace('c2c_gsm5_native', 'c2c_gsm10_native').replace('limit: 5', 'limit: 10')
Path('/tmp/c2c_gsm10.yaml').write_text(text)
text = text.replace('c2c_gsm10_native', 'c2c_gsm30_native').replace('limit: 10', 'limit: 30')
Path('/tmp/c2c_gsm30.yaml').write_text(text)
PY
python script/evaluation/unified_evaluator.py --config /tmp/c2c_gsm10.yaml
python script/evaluation/unified_evaluator.py --config /tmp/c2c_gsm30.yaml
```

Fair-comparison caveats:
- Native C2C is a direct cross-model baseline, but it is not byte-matched to LatentWire unless the payload accounting is normalized.
- Native C2C reads HF `gsm8k`, so this is a smoke, not a local JSONL-matched table row.
- CUDA box only, after the published fuser checkpoint exists.
- Do not download the published fuser in this planning pass.

Expected artifacts:
- `results/competitor_bootstrap_20260421/c2c_qwen25_05b_to_qwen3_06b.json`
- `results/competitor_bootstrap_20260421/c2c_gsm{5,10,30}_native/pred/<model>/*.jsonl`
- `results/competitor_bootstrap_20260421/c2c_gsm{5,10,30}_native/*_summary.json`
- optional `bad_samples/`, `*_length.json`, and `*_cot.csv` depending on config

## kvpress

Primary sources:
- `references/repos/kvpress/evaluation/README.md`
- `scripts/run_kvpress_eval.py`
- `https://github.com/NVIDIA/kvpress`
- `https://arxiv.org/abs/2510.00636`

Run the floor and the query-aware comparator on all three GSM smoke slices:

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
  --prediction-output results/competitor_bootstrap_20260421/kvpress_none_gsm5.jsonl

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

./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file /tmp/gsm8k_eval_10.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_none_gsm10.jsonl

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

./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press none \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_none_gsm30.jsonl

./venv_arm64/bin/python scripts/run_kvpress_eval.py \
  --model Qwen/Qwen3-0.6B \
  --eval-file data/gsm8k_gate_search_30.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --press expected_attention \
  --compression-ratio 0.5 \
  --no-enable-thinking \
  --prediction-output results/competitor_bootstrap_20260421/kvpress_expected_attention_gsm30.jsonl
```

Fair-comparison caveats:
- This is a same-model compression control, not a cross-model communication result.
- The vendored clone is dirty; use the wrapper because it patches compatibility without editing repo files.
- `ExpectedAttentionPress` may need a CUDA fallback if MPS is unstable.

Expected artifacts:
- six `*.jsonl` prediction files under `results/competitor_bootstrap_20260421/`
- six companion `*.jsonl.meta.json` sidecars
- printed JSON summary per run

## KVzip

Primary sources:
- `references/repos/KVzip/README.md`
- `references/repos/KVzip/eval.py`
- `references/repos/KVzip/results/parse.py`
- `https://github.com/snu-mllab/KVzip`
- `https://arxiv.org/abs/2505.23416`

Use the repo-native GSM evaluator for the 5/10/30-example smoke:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVzip
python -B eval.py -m qwen2.5-7b -d gsm --kv_type retain --num 5 --tag latentwire_gsm5_retain06
python -B eval.py -m qwen2.5-7b -d gsm --kv_type retain --num 10 --tag latentwire_gsm10_retain06
python -B eval.py -m qwen2.5-7b -d gsm --kv_type retain --num 30 --tag latentwire_gsm30_retain06
```

Fair-comparison caveats:
- KVzip is a same-model query-agnostic compression control, not a source-target transport method.
- The repo is CUDA/flash-attn oriented; do not treat this as a Mac/MPS baseline.
- `eval.py` sweeps ratios internally; this smoke limits the sample count, not the ratio grid.

Expected artifacts:
- per-example JSON under `references/repos/KVzip/results/gsm/<idx>_qwen2.5-7b_latentwire_gsm*_retain06/output-pair.json`
- optional aggregate `references/repos/KVzip/results/gsm/result.json` if `python -B -m results.parse -m qwen2.5-7b -d gsm` is run after the smoke

## KVComm

Primary sources:
- `references/repos/KVComm/README.md`
- `references/repos/KVComm/com.py`
- `https://github.com/Zephyroam/KVComm`
- `https://arxiv.org/abs/2510.03346`

KVComm is not GSM-native in this clone. Use the supported QA task smoke instead:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/KVComm
python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 5

python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 5

python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 10

python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 10
```

Fair-comparison caveats:
- Do not call these GSM numbers; the repo is QA-task native, not GSM-native.
- The local clone is dirty in `model_attn.py` and `models.py`, so this should be treated as a compatibility-lift smoke, not a pristine reproduction.
- If you need structured artifacts, use `scripts/run_kvcomm_eval.py` instead of the stock repo entrypoint, but keep that separate from the stock competitor smoke.

Expected artifacts:
- stdout metrics and calibration summaries from `com.py`
- no canonical on-disk prediction artifact from the stock command unless stdout is redirected

## Quest

Primary sources:
- `references/repos/Quest/README.md`
- `references/repos/Quest/evaluation/passkey/passkey.py`
- `references/repos/Quest/evaluation/LongBench/pred.py`
- `references/repos/Quest/evaluation/LongBench/eval.py`
- `https://github.com/mit-han-lab/Quest`
- `https://arxiv.org/abs/2406.10774`

Use a cheap passkey smoke first, then a single LongBench task if the kernel stack is already built:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/evaluation/passkey
mkdir -p /Users/sujeethjinesh/Desktop/LatentWire/results/competitor_bootstrap_20260421
python passkey.py \
  -m meta-llama/Llama-3.1-8B-Instruct \
  --fixed-length 100000 \
  --iterations 5 \
  --quest \
  --token_budget 512 \
  --chunk_size 16 \
  --output-file /Users/sujeethjinesh/Desktop/LatentWire/results/competitor_bootstrap_20260421/quest_passkey_quest512.csv

cd /Users/sujeethjinesh/Desktop/LatentWire/references/repos/Quest/evaluation/LongBench
python -u pred.py \
  --model longchat-v1.5-7b-32k \
  --task qasper

python -u pred.py \
  --model longchat-v1.5-7b-32k \
  --task qasper \
  --quest \
  --token_budget 512 \
  --chunk_size 16

python -u eval.py --model longchat-v1.5-7b-32k
```

Fair-comparison caveats:
- Quest is a long-context query-aware sparsity control, not a GSM competitor.
- The repository is CUDA/FlashAttention oriented; do not treat the passkey smoke as a model-aligned apples-to-apples result for the current Qwen pair.

Expected artifacts:
- `results/competitor_bootstrap_20260421/quest_passkey_quest512.csv`
- `references/repos/Quest/evaluation/LongBench/pred/<model>/*.jsonl`
- `references/repos/Quest/evaluation/LongBench/pred/<model>/result.json`

## Guardrails

- Do not merge direct communication peers and same-model compression controls into one leaderboard.
- Do not compare bytes across C2C and LatentWire unless payload, metadata, and checkpoint costs are normalized.
- Do not publish a GSM score for KVComm or Quest without a validated GSM adapter.
- Do not mutate the vendored competitor repos in this planning step.
