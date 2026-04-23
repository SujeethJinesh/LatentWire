# SVAMP32 ID-Weighted Query Innovation

Date: 2026-04-23

## Paper Status

Not ICLR-ready. The live SVAMP32 lane has source-necessary evidence for one
clean residual ID, but not enough breadth or robustness for a publishable
positive method.

## Current Story

The target already has strong side information (`target_self_repair` reaches
`14/32`), while C2C teacher reaches `16/32`. The useful method direction is
therefore conditional innovation: keep the target/self-repair path as decoder
side information and use a narrow source-conditioned residual only for missing
teacher-only IDs.

This turn tested the smallest implementation of that story: exact-ID sample
weighting for the existing
`bridge_ridge_qk_dynalign_query_innovation_resampler_replace` residual module.

## Blocking Gap

The paper gate still requires at least `2/6` clean residual C2C-only IDs that
are not retained by controls. The best row this turn recovers `1/6`.

## Top Moves Considered

1. ID-weighted query innovation module fit.
   - Why it matters: directly targets the six clean residual IDs without adding
     a new connector.
   - Why it might fail: can memorize prompt IDs or learn target-side repair.
   - Evidence gained: whether clean residual IDs are reachable by the existing
     residual bottleneck.
   - Cost: medium.
   - Helps: same-pair, robustness, reproducibility, interpretability.

2. Full source-necessity control runner.
   - Why it matters: prevents false promotion from target-cache effects.
   - Why it might fail: no candidate yet clears enough clean IDs to justify the
     full run cost.
   - Evidence gained: source-necessity under translated-KV-zero, source-zero,
     and shuffled-source controls.
   - Cost: medium/high.
   - Helps: robustness and reproducibility.

3. New target-side-information innovation connector.
   - Why it matters: closest to the Wyner-Ziv/Q-Former/Perceiver story surfaced
     by literature and lateral agents.
   - Why it might fail: higher implementation risk and more degrees of freedom
     before the existing residual path is exhausted.
   - Evidence gained: whether a new bottleneck can recover multiple clean IDs.
   - Cost: high.
   - Helps: same-pair, efficiency, interpretability.

Decision: run move 1, then the cheapest decisive control for any clean hit.

## Code Changes

- `latent_bridge/calibrate.py`
  - added JSONL-aware calibration prompt metadata loading
  - added stable generation example ID reconstruction matching
    `latent_bridge/evaluate.py`
  - added `--innovation-target-set-json`
  - added `--innovation-positive-weight`
  - added `--innovation-default-weight`
  - expands prompt-level clean residual target IDs into flattened bridge sample
    weights via `sample_prompt_ids`

- `latent_bridge/translator.py`
  - forwards optional sample weights into the query-innovation module fit
  - keeps the base ridge fit unweighted for this mode, so the new hook only
    biases the residual innovation head

## Verification

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_calibrate_and_ablation.py \
  tests/test_translator_core.py -q
```

Result: `211 passed`

## Calibration

Checkpoint:

- `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`

Key calibration telemetry:

- frozen JSONL calibration prompts: `32`
- dynamic token-mixture samples: `1411`
- clean residual target prompts matched: `6`
- default weight: `1.0`
- positive clean-ID weight: `16.0`
- average fit quality:
  - K cosine: `0.951`
  - V cosine: `0.734`

## Runtime Evidence

Artifacts:

- `results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate010_matched.jsonl`
- `results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl`
- `results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate020_matched.jsonl`
- `results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_translated_kv_zero.jsonl`
- `results/svamp32_idweighted_query_innovation_20260423/c2c_teacher_probe_gate015_targetself_translated_zero.json`
- `results/svamp32_idweighted_query_innovation_20260423/paper_gate_gate015_targetself_translated_zero.json`

Matched gate sweep:

| Gate | Correct | Teacher-only recovered | Clean residual recovered |
|---:|---:|---:|---:|
| `0.10` | `8/32` | `1` | `0` |
| `0.15` | `10/32` | `2` | `1` |
| `0.20` | `8/32` | `1` | `0` |

Best row: `gate015`

- correct: `10/32`
- target-alone: `8/32`
- target_self_repair: `14/32`
- C2C teacher: `16/32`
- teacher-only recovered: `575d7e83d84c1e67`, `aee922049c757331`
- clean residual recovered: `aee922049c757331`
- translated-KV-zero retained: `575d7e83d84c1e67`
- translated-KV-zero did not retain clean residual ID `aee922049c757331`

Paper gate:

- status: `no_candidate_passes_target_self_repair_gate`
- clean residual recovered: `1/6`
- clean source-necessary recovered: `1/6`
- failing criteria:
  - `min_correct`
  - `beats_target_self_repair`
  - `min_teacher_only`
  - `min_clean_residual_recovered`
  - `min_clean_source_necessary`

## Interpretation

The branch is revived but not promoted. The clean win on
`aee922049c757331` is meaningful because it disappears under
`translated_kv_zero`, so it is not simply target-cache repair. However, one
clean source-necessary ID is below the gate and the row is still worse than
`target_self_repair`.

The next method should preserve the target_self_repair row and add source
innovation on top, rather than replacing it. The strongest next branch is a
target-self-conditioned residual composition: start from target_self_repair,
then allow an additive source innovation only on IDs/layers whose translated
packet beats translated-KV-zero.

## Next Gate

Recover at least `2/6` clean residual IDs while preserving
`target_self_repair`'s `14/32`, then rerun the full control suite:

- matched
- translated-KV-zero
- source-KV-zero
- two shuffled-source salts
- strict target-set paper gate
