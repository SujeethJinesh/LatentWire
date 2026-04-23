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

## Target-Self Sidecar Bound

To test whether this exact candidate is worth wrapping in a deployable
target-self-preserving sidecar, I added an oracle-bound analysis:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_sidecar_bound.py \
  --probe-json results/svamp32_idweighted_query_innovation_20260423/c2c_teacher_probe_gate015_targetself_translated_zero.json \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --candidate-label gate015 \
  --source-control-label translated_kv_zero \
  --output-json results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.json \
  --output-md results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.md
```

Result:

- status: `oracle_sidecar_bound_fails_gate`
- oracle `target_self_repair + clean source sidecar`: `15/32`
- delta versus `target_self_repair`: `+1`
- target losses versus `target_self_repair`: `0`
- clean source-necessary IDs: `1/6`
- clean source-necessary ID: `aee922049c757331`
- failing criteria:
  - `min_correct`
  - `min_clean_source_necessary`

Analyzer verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_source_sidecar_bound.py \
  tests/test_analyze_svamp32_paper_gate.py \
  tests/test_analyze_c2c_teacher_innovation.py \
  tests/test_build_svamp32_innovation_target_set.py -q
```

Result: `20 passed`

Interpretation: the target-self-preserving sidecar idea remains alive, but
wrapping this exact `gate015` candidate is not worth a runtime implementation.
Even a perfect oracle router around it cannot clear the current paper gate.

## Fine Attention Gate Search

After the sidecar bound failed, I tested whether the same checkpoint could
expose a second clean residual ID through a finer matched-only attention-gate
sweep before spending controls.

Command:

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
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.05 0.10 0.125 0.15 0.175 0.20 0.25 \
  --methods rotalign \
  --prediction-output .debug/svamp32_candidate_search_20260423/idweighted_attention_fine_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Readout artifacts:

- `.debug/svamp32_candidate_search_20260423/idweighted_attention_fine_gate_sweep.jsonl`
- `results/svamp32_idweighted_query_innovation_20260423/fine_gate_sweep_clean_targets_attention.json`
- `results/svamp32_idweighted_query_innovation_20260423/fine_gate_sweep_clean_targets_attention.md`

Result:

- status: `no_matched_gate_candidate_for_controls`
- best row: `rotalign_kv_gate_0.17`, `11/32`
- clean residual recovered: `1/6`
- clean residual ID: `aee922049c757331`
- oracle `target_self_repair + clean candidate` bound: `15/32`
- numeric extraction coverage: `32/32` for every row

Matched-only gate ranking:

| Gate | Correct | Teacher-only recovered | Clean residual recovered |
|---:|---:|---:|---:|
| `0.05` | `8/32` | `1` | `0` |
| `0.10` | `8/32` | `1` | `0` |
| `0.125` | `8/32` | `1` | `0` |
| `0.15` | `10/32` | `2` | `1` |
| `0.175` | `11/32` | `2` | `1` |
| `0.20` | `8/32` | `1` | `0` |
| `0.25` | `7/32` | `0` | `0` |

Interpretation: fixed attention-gate retuning of this checkpoint is saturated.
It improves the matched row from `10/32` to `11/32` but still only recovers the
same single clean residual ID, so translated-KV-zero, source-zero, and shuffle
controls are not justified for the fine-gate rows.

## Interpretation

The branch is revived but not promoted. The clean win on
`aee922049c757331` is meaningful because it disappears under
`translated_kv_zero`, so it is not simply target-cache repair. However, one
clean source-necessary ID is below the gate and the row is still worse than
`target_self_repair`.

The fine attention-gate sweep weakens the hypothesis that runtime thresholding
alone can rescue the existing ID-weighted checkpoint. The next candidate should
change training or the bottleneck objective, not just the runtime gate.

The next method should still preserve the target_self_repair row and add source
innovation on top, rather than replacing it. But the sidecar-bound result
and fine-gate result promote a new candidate/search step before runtime router
work: the candidate must first expose at least two clean source-necessary IDs
under matched versus source-destroying controls.

## Next Gate

Recover at least `2/6` clean residual IDs while preserving
`target_self_repair`'s `14/32`, then rerun the full control suite:

- matched
- translated-KV-zero
- source-KV-zero
- two shuffled-source salts
- strict target-set paper gate
