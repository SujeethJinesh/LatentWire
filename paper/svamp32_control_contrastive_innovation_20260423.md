# SVAMP32 Source-Control Contrastive Innovation

Date: 2026-04-23

## Paper Status

- readiness: not ICLR-ready
- current story: target self-repair is the decoder-side-information floor; a publishable positive method must add source-conditioned innovation above it
- blocking gap: no live method has reached `>=2/6` clean source-necessary SVAMP32 residual IDs under exact-ID matched scoring
- turn gate: train one source-control query-innovation candidate and run the cheapest decisive matched gate sweep

## Decision Context

Alive:

- source-control contrastive innovation objective under the existing query-innovation resampler
- richer connector / Q-former-style bottlenecks if the cheap control objective fails

Saturated:

- runtime scalar gate retuning around the prior ID-weighted checkpoint
- scalar clean-ID over-weighting of the existing query-innovation objective
- runtime sidecar/router work before a matched candidate clears `>=2/6`

Blocked:

- benchmark widening, seed sweeps, and zero/shuffle controls until a matched row clears the live paper gate

Hypothesis updates before this run:

- weakened: source information is recoverable by scalar gate search alone
- weakened: simply increasing clean-ID sample weight yields useful innovation
- promoted: source-control negatives may force the module to encode matched-source innovation rather than target-cache repair

## Top Moves Considered

1. Source-control query-innovation objective. Matters because it directly targets source necessity; may fail by suppressing all residual innovation. Evidence gained: matched clean-ID count after zero/shuffle control training. Cost: one calibration plus one matched sweep. Helps same-pair, robustness, and interpretability.
2. V-full `both` transport screen on the prior ID-weighted checkpoint. Matters because it cheaply tests whether value transport exposes another clean ID; may fail by adding value noise or target-cache wins. Evidence gained: matched clean-ID count without new training. Cost: one decode sweep. Helps same-pair and efficiency.
3. Larger Q-former / Perceiver connector. Matters because it is the strongest connector story if scalar objectives saturate; may fail by adding too many degrees before the live gate. Evidence gained: whether a stronger bottleneck can create source-necessary residual IDs. Cost: higher implementation and calibration. Helps cross-family and interpretability if it works.

Chosen move: source-control query-innovation objective, because it is the most direct test of the current blocker.

## Implementation

Patch scope:

- `latent_bridge/translator.py`
- `latent_bridge/calibrate.py`
- `tests/test_translator_core.py`
- `tests/test_calibrate_and_ablation.py`

New default-off config/CLI:

- `innovation_control_weight`
- `innovation_control_mode in {none, zero, shuffle, zero_and_shuffle}`
- `innovation_contrastive_margin`

Semantics:

- only valid for `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- zero and shuffled source controls are trained toward zero innovation delta
- shuffled controls require `sample_prompt_ids`
- contrastive margin separates matched-source innovation norm from control-source innovation norm

Design constraint from the repo audit: the module output is an innovation delta, not full translated KV, so controls must not be trained toward `target - control_base`.

## Verification

```bash
./venv_arm64/bin/python -m pytest tests/test_translator_core.py tests/test_calibrate_and_ablation.py -q
```

Result: `214 passed`

## Candidate

Checkpoint:

- `.debug/svamp32_control_contrastive_innovation_20260423/checkpoints/qwen25_to_qwen3_svamp32_control_zero_shuffle_w010_m001_r16_bank16_seed1.pt`

Calibration settings:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- rank: `16`
- bank size: `16`
- seed: `1`
- innovation target weights: positive `16`, default `1`
- source control: `zero_and_shuffle`
- control weight: `0.10`
- contrastive margin: `0.001`
- source reasoning: `brief_analysis`
- alignment: `grouped_subspace_transport`

Calibration result:

- alignment quality: K cosine `0.951`, V cosine `0.734`
- prompt IDs built for source controls: `1411` samples, `32` prompts

## Matched Sweep

Prediction artifact:

- `.debug/svamp32_control_contrastive_innovation_20260423/preds/control_zero_shuffle_w010_m001_attention_gate_sweep.jsonl`

Readout artifacts:

- `results/svamp32_control_contrastive_innovation_20260423/control_zero_shuffle_w010_m001_attention_clean_targets.json`
- `results/svamp32_control_contrastive_innovation_20260423/control_zero_shuffle_w010_m001_attention_clean_targets.md`

Result:

- status: `no_matched_gate_candidate_for_controls`
- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best row: `rotalign_kv_gate_0.12`, `9/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`
- delta vs target_self_repair: `-5`
- oracle `target_self_repair + clean candidate`: `14/32`

Interpretation:

- low-weight zero/shuffle source-control training suppresses useful residual innovation instead of exposing clean source-necessary IDs
- this branch is weakened, not promoted
- no translated-zero, source-zero, shuffle, or sidecar controls are justified because the matched candidate failed the live gate
- the seven-gate sweep is too expensive for broad iteration; future screens should use fewer gates unless a candidate is already near promotion

## Literature / Inspiration Notes

- C2C frames model-to-model communication as a useful baseline and teacher signal: https://arxiv.org/abs/2510.03215
- BLIP-2 motivates learned query bottlenecks for frozen-model transfer: https://arxiv.org/abs/2301.12597
- Flamingo motivates cross-attention connector structure: https://arxiv.org/abs/2204.14198
- InfoNCE / CPC motivates matched-vs-negative contrastive pressure, but the SVAMP32 result here suggests naive source-control pressure is not sufficient: https://arxiv.org/abs/1807.03748
- Wyner-Ziv coding motivates source messages useful only under target-side information: https://doi.org/10.1109/TIT.1976.1055508

## Next Gate

Do not continue source-control scalar tuning until there is a stronger architectural reason. The next exact gate is either:

- run the cheaper V-full `both` transport screen on the prior ID-weighted checkpoint, or
- implement a small learned query bottleneck that can preserve target self-repair while injecting source innovation.

Promotion criterion is unchanged: `>=2/6` clean residual IDs in matched exact-ID scoring before any control sweep.
