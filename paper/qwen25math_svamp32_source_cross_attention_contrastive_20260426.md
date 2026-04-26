# Qwen2.5-Math -> Qwen3 Source-Control Contrastive Cross-Attention Gate

- date: `2026-04-26`
- status: `fails_gate`
- rung: strict-small pre-generation diagnostic
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- surface: SVAMP32 C2C-headroom exact-ID slice

## Start Status

- current ICLR readiness: not ready
- current paper story: decoded sidecars, source readouts, target-safe selectors,
  and tiny prefix emitters do not survive source-destroying controls
- exact blocker: target-only, label-shuffled, shuffled-source, and other
  controls dominate the matched-source logprob margin
- live branch: source-control contrastive training for a token-local
  cross-attention prefix connector

## Method Change

The previous token-local cross-attention gate trained only the matched-source
connector, then discovered after training that controls had higher
gold-vs-distractor continuation margins.

This run adds an optional source-control contrastive training penalty to
`scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`: during
matched-source training, the connector is penalized when zero-source,
shuffled-source, same-norm-noise, or projected-source controls match or exceed
the real-source margin.

Configuration:

- prefix length: `2`
- hidden dim: `16`
- epochs: `1`
- outer folds: `2`
- feature layers: `last`
- contrastive weight: `0.25`
- contrastive margin: `0.25`
- controls in training penalty: zero-source, shuffled-source, same-norm-noise,
  projected-soft-prompt

## Result

| Metric | Value |
|---|---:|
| clean IDs scored | `6` |
| matched-only clean IDs | `0/6` |
| matched-positive clean IDs | `4/6` |
| clean control leaks | `4/6` |
| mean matched margin on clean IDs | `0.070074` |
| mean best-control margin on clean IDs | `0.452928` |
| mean matched-minus-control clean margin | `-0.382854` |
| target-preservation IDs scored | `8` |
| target-preservation matched-positive count | `5/8` |

The control-dominance failure remains. The matched-source connector recovers no
matched-only clean IDs, and four clean IDs are still better explained by
label-shuffled, same-norm, target-only, or shuffled-source controls.

## Decision

Prune this source-control contrastive variant. Do not spend more compute on
this exact tiny prefix-emitting cross-attention architecture by tuning the
contrastive weight, margin, epochs, or hidden width. It directly targeted the
known failure mode and did not change the clean-ID attribution.

The next method branch needs a larger architectural change rather than another
small objective tweak: either a true target-side next-token-loss resampler with
generation-time evaluation and matched C2C-fuser baseline, or a new
source/surface pair that first clears source/text/C2C headroom gates.

## Artifacts

- result JSON:
  - `results/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.json`
  - sha256: `2f6e2a38f6b1685b7a571f30f53dd1587fa03532560aa7fe04f4f515a15cb4a1`
- readout:
  - `results/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.md`
  - sha256: `e009da82d298ade4e65aa0c76709f67f77066654485170fe05a27ca3e9918637`
- analyzer:
  - `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
  - sha256: `884005c8b61e41fb908eed6759fe484b90cc8060ef42850a515a2b8327be7d75`

## Command

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_cross_attention_logprob_probe.py \
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
  --length-normalize true \
  --source-control-contrastive-weight 0.25 \
  --source-control-contrastive-margin 0.25 \
  --source-control-contrastive-control zero_source \
  --source-control-contrastive-control shuffled_source \
  --source-control-contrastive-control same_norm_noise \
  --source-control-contrastive-control projected_soft_prompt \
  --device mps \
  --train-device mps \
  --dtype float32 \
  --date 2026-04-26 \
  --output-json .debug/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.json \
  --output-md .debug/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.md
```

## Tests

- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py -q`
