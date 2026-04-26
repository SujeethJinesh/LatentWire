# Qwen2.5-Math -> Qwen3 SVAMP32 Source Soft-Prefix Logprob Gate

- date: `2026-04-26`
- status: `killed_before_generation`
- scale-up rung: `micro smoke / strict-small pre-generation diagnostic`
- live branch entering run: source-conditioned soft-prefix summary connector
- next branch: token/layer-local gated cross-attention objective

## Question

Can a tiny source-conditioned soft-prefix, trained directly on target-model
gold-vs-distractor continuation logprob, recover clean C2C-headroom SVAMP32
examples while beating source-destroying and target-only controls?

This was the next gate after target-query source bottleneck and process-repair
controls failed. The branch is deliberately pre-generation: if the target model
cannot prefer the gold continuation under a source-conditioned prefix more than
under controls, no generation run is justified.

## Method

Added `scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py`.

The diagnostic:

- extracts frozen Qwen2.5-Math source features and Qwen3 target prompt features
- fold-standardizes summary features using train examples only
- trains a tiny MLP connector into Qwen3 input-embedding prefix vectors
- optimizes gold-vs-distractor continuation logprob
- cross-fits heldout examples
- scores matched source against:
  - zero source
  - shuffled source
  - same-norm random source
  - train-mean projected soft prompt
  - target-only learned prefix
  - slots-only learned prefix
  - label-shuffled learned prefix

The decisive calibrated smoke used source-only matched prefixes, numeric-only
distractors, and mean-token continuation logprob.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl \
  --teacher-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --feature-layers last \
  --prefix-len 2 \
  --hidden-dim 16 \
  --epochs 1 \
  --outer-folds 2 \
  --matched-use-target false \
  --length-normalize true \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --output-json .debug/qwen25math_svamp32_source_soft_prefix_20260426/source_only_numeric_meanlogprob_smoke.json \
  --output-md .debug/qwen25math_svamp32_source_soft_prefix_20260426/source_only_numeric_meanlogprob_smoke.md
```

The durable readout was promoted to
`results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/`.

## Results

Promotion rule:

- at least `2/6` clean C2C-headroom IDs must be matched-only positive
- no clean control leaks
- target-correct/self-preservation must not collapse

Observed:

- clean IDs scored: `6`
- matched-only clean IDs: `1/6`
- control-leak clean IDs: `4/6`
- matched-positive clean IDs: `5/6`
- mean matched margin on clean IDs: `1.319601`
- mean best-control margin on clean IDs: `2.090727`
- mean matched-minus-best-control clean margin: `-0.771126`
- target-self/target-correct IDs scored: `8`
- target-self/target-correct matched-positive count: `6/8`

Clean row attribution:

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |
|---|---:|---:|---:|---|---:|---:|---|
| `3e8a5691f5443495` | 1 | 3 | 1.159 | label_shuffled | 2.362 | -1.203 | control explained |
| `1d50b408c8f5cd2c` | 949 | 1 | 3.142 | target_only_prefix | 5.260 | -2.118 | control explained |
| `de1bf4d142544e5b` | 57 | 2 | 2.646 | label_shuffled | 3.250 | -0.604 | control explained |
| `47464cc0b064f172` | 24 | 2 | 4.618 | label_shuffled | 4.101 | 0.517 | matched-only positive |
| `6e9745b37ab6fc45` | 61 | 600 | -3.758 | label_shuffled | -3.410 | -0.348 | negative/control |
| `575d7e83d84c1e67` | 2 | 24 | 0.111 | target_only_prefix | 0.981 | -0.870 | control explained |

## Decision

Kill pooled-summary soft-prefix connectors on this SVAMP32 surface before
generation. The calibrated source-only variant still recovers only `1/6` clean
IDs and is dominated by target-only or label-shuffled controls on most clean
examples.

This is not a hard kill of learned communication. It is a kill of the current
summary-vector soft-prefix formulation. The next highest-value branch should
use token/layer-local source access: a small gated cross-attention or local
source-token query objective trained on the same gold-vs-distractor surface,
with identical source-destroying and target-only controls.

## Artifacts

- analyzer: `scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py`
- tests: `tests/test_analyze_svamp32_source_soft_prefix_logprob_probe.py`
- result JSON:
  `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/source_only_numeric_meanlogprob_smoke.json`
- readout:
  `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/source_only_numeric_meanlogprob_smoke.md`
- hashes:
  `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/sha256.txt`

SHA256:

- analyzer:
  `913a0a8f4ae971d90fe47c9ed49f8a05ff83080e64eea4fb7b7ebe8c24bfc573`
- test:
  `2651ad7f8ec8d9f6e1ba8b5c94e8dc0a3ea7c87176b400cc8c0fda78fd2f0ac9`
- result JSON:
  `f89c0a8759a94574de9e5a52eb50af800fab352c13550efdf1660f85d33778c9`
- readout:
  `cd67391b2c87a449a2096fb04942a8232cc7f8035bcbb3aa989b4e1aeae94169`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_soft_prefix_logprob_probe.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py
```

Result: `5 passed`.

## Next Exact Gate

Implement a token/layer-local gated cross-attention pre-generation gate on the
same Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom surface:

- matched source tokens queried by target prompt state
- zero-source, shuffled-source, same-norm, target-only, slots-only,
  projected-soft-prompt, and label-shuffled controls
- numeric-only distractors
- mean-token continuation logprob
- fold-local feature normalization
- promotion rule unchanged: at least `2/6` clean matched-only IDs and no clean
  control leaks before generation
