# SVAMP32 Value Transport Screen

Date: 2026-04-23

## Paper Status

- readiness: not ICLR-ready
- current story: target self-repair is the decoder-side-information floor; source communication must add clean residual innovation above it
- blocking gap: no live candidate reaches `>=2/6` clean source-necessary SVAMP32 residual IDs under exact-ID matched scoring
- turn gate: test whether value-side transport on the prior ID-weighted query-innovation checkpoint exposes a second clean residual ID

## Decision Context

Alive before this run:

- value-side runtime selection on the prior ID-weighted checkpoint
- target-self-preserving learned query bottleneck / Q-Former-style connector

Saturated before this run:

- scalar fixed-gate retuning
- scalar clean-ID sample over-weighting
- naive zero/shuffle source-control contrastive retraining
- one-gate sidecar/routing wrappers around failed matched rows

Blocked:

- source-destroying controls, seed repeats, cross-family pairs, and benchmark widening until a matched row clears `>=2/6` clean residual IDs

## Top Moves Considered

1. V-full `both` transport screen on the prior ID-weighted checkpoint. Matters because the prior live row was `k_only`, so value-side answer evidence might expose another clean ID. It could fail by injecting V noise or target-cache wins. Cost: one focused decode. Helps same-pair and efficiency.
2. Sparse source-attention V follow-up. Matters because full V could be too noisy while sparse V could preserve K-route behavior and add answer-side source evidence. It could fail by recovering only the same known clean ID. Cost: one smaller three-gate decode. Helps same-pair and efficiency.
3. Conditional query-bottleneck connector. Matters because local failures and external priors point away from scalar bridge tweaks toward learned connectors. It could fail by memorizing six clean IDs or leaking target repair. Cost: medium-high implementation and calibration. Helps interpretability and reproducibility if it works.

Chosen move: V-full screen, followed by one sparse-V falsification because V-full failed but did not by itself kill the selective-value hypothesis.

## External Priors

- C2C uses learned KV projection/fusion rather than raw value injection, which motivates connector-style fusers over more scalar transport screens: https://arxiv.org/abs/2510.03215
- KVComm makes selective KV sharing a direct communication baseline and reinforces that selection policy matters under bandwidth constraints: https://arxiv.org/abs/2510.03346
- BLIP-2 / Q-Former is the canonical frozen-backbone learned query bridge: https://arxiv.org/abs/2301.12597
- Perceiver IO supports fixed latent-query bottlenecks for arbitrary structured inputs/outputs: https://arxiv.org/abs/2107.14795

## Reproducibility

Prior ID-weighted checkpoint:

- path: `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`
- sha256: `4e67fdf2b6ea2c962036aad080ec3fe6c64a4083c627a282b925bdf546b90831`

Analyzer verification:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_gate_sweep_clean_targets.py -q
```

Result: `4 passed`

## V-Full Both Transport

Matched sweep:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport both \
  --kv-route-selection-ratio 0.5 \
  --kv-value-selection-ratio 1.0 \
  --kv-route-selection-metric attention \
  --kv-value-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.10 0.125 0.15 0.175 0.20 \
  --methods rotalign \
  --prediction-output .debug/svamp32_vfull_both_transport_20260423/preds/idweighted_both_vfull_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Prediction hashes:

- JSONL sha256: `d9bbed502ba4293120362b396188bf12458f1d10bcd482718c23184eed440dff`
- meta sha256: `849e318a9c00be1a86951b66f435986d2b1ba664b4c0dc8068dbc81432cb2744`

Readouts:

- `results/svamp32_vfull_both_transport_20260423/idweighted_both_vfull_attention_clean_targets.json`
- `results/svamp32_vfull_both_transport_20260423/idweighted_both_vfull_attention_clean_targets.md`

Result:

- status: `no_matched_gate_candidate_for_controls`
- best row: `rotalign_kv_gate_0.20`, `9/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`
- target losses: `2`
- delta vs target_self_repair: `-5`
- transport bytes: `1,193,918.25`
- verdict: full V is noisier, larger, and worse than the prior K-only live row

## Sparse Source-Attention V

Matched sweep:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport both \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --kv-route-selection-ratio 0.5 \
  --kv-route-selection-metric attention \
  --kv-value-selection-ratio 0.25 \
  --kv-value-selection-metric source_attention \
  --gate-mode sweep \
  --gate-values 0.125 0.15 0.175 \
  --methods rotalign \
  --prediction-output .debug/svamp32_vfull_both_transport_20260423/preds/idweighted_sparse_sourcev_attention_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Prediction hashes:

- JSONL sha256: `988820878a2a3f3659a3f8448e2d4e0b1c44ec815667219ac8bf69e1a1dfb1ab`
- meta sha256: `463c66f7b6f43f533df0abe509e797bec1263981892a623ad9f4f724739f389d`

Readouts:

- `results/svamp32_vfull_both_transport_20260423/idweighted_sparse_sourcev_attention_clean_targets.json`
- `results/svamp32_vfull_both_transport_20260423/idweighted_sparse_sourcev_attention_clean_targets.md`

Result:

- status: `no_matched_gate_candidate_for_controls`
- best rows: `rotalign_kv_gate_0.15` and `rotalign_kv_gate_0.17`, `10/32`
- clean residual recovered: `1/6`
- clean ID: `aee922049c757331`
- teacher-only recovered: `2`
- target losses: `1` at gate `0.15`, `2` at gate `0.17`
- delta vs target_self_repair: `-4`
- transport bytes: `597,337.671875`
- verdict: sparse V restores the known one clean ID but does not expose a second

## Interpretation

Value-side runtime selection on this checkpoint is saturated:

- V-full is worse than K-only and recovers no clean residual IDs.
- Sparse V recovers only the already-known clean ID.
- Both are below target_self_repair and below the live clean-ID promotion gate.
- No source-zero, shuffle, translated-zero, or sidecar controls are justified.

Hypothesis update:

- killed for now: full value transport can rescue the prior ID-weighted checkpoint
- weakened: runtime sparse V selection can expose a second clean residual ID
- promoted: target-self-preserving conditional query bottleneck / residual query codec

## Next Gate

Implement or train the conditional residual query codec described in
`paper/svamp32_next_method_conditional_residual_query_codec_20260423.md`.

Promotion criterion remains unchanged:

- `>=2/6` clean residual IDs in matched exact-ID scoring
- preserve target_self_repair floor within at most one target-correct loss
- only then run source-destroying controls
