# Query-Resampler Answer-Likelihood Gate

- date: `2026-04-26`
- live branch: `latent_bridge` query-innovation/source-memory resampler
- scale-up rung: smoke/strict-small harness gate
- status: tooling blocker before MPS execution

## Start-Of-Cycle State

1. ICLR readiness: not ready; the repo still lacks a deployable positive method
   with seed-stable source-destroying controls, systems metrics, and
   cross-family falsification.
2. Current paper story: C2C and source-sidecar audits show real
   source-complementary headroom, while deployable LatentWire rows remain
   vulnerable to target-cache, selector, target-prior, or seed-stability
   explanations.
3. Exact blocker to submission: no source-derived method beats target/text
   baselines while surviving zero-source, shuffled-source, target-only, and
   slots-only controls.
4. Current top branch: query-innovation/source-memory resampler; fixed
   source-sidecar guards are now a surface clue rather than the live method.
5. Highest-priority gate: matched-vs-source-destroyed answer-likelihood scoring
   on the existing finite query-innovation checkpoint before any larger run.
6. Scale-up rung: smoke/strict-small selection.

## Historical MD/Results Audit

The audit covered the latest ledger, readiness review, SVAMP/GSM memos,
`rotalign`, `latent_bridge`, and relevant `results/` / `.debug/` directories.
The useful conclusions are:

- `dynalign_module_replace_residrank16` remains only a mechanism clue: seed 0
  is positive, but seeds 1/2 are nonfinite and finite repeats do not preserve a
  positive method.
- SVAMP70 one-byte source-sidecar rows are surface/headroom evidence, not a
  deployable live branch: the live CV guard has clean source wins, but fixed
  holdout guards lose clean source-only wins and leak controls.
- Query-resampler/query-innovation rows are the best live architecture clue:
  they are finite and target-safe enough to test, but prior GSM8K32 accuracy
  wins were retained by zero/shuffled source controls.
- Tiny prefix-emitter/cross-attention families are killed on the current
  SVAMP surfaces after target-CE generation still underperforms controls.

## Implementation

Added eval-only generation answer scoring to `latent_bridge/evaluate.py`.

- Each RotAlign generation record now scores answer aliases as continuations
  from the constructed prefix state.
- Per-record fields include `answer_logprob`, `answer_nll`,
  `answer_mean_logprob`, `answer_mean_nll`, `answer_tokens`,
  `answer_scored_text`, and `answer_reference`.
- Generation summaries now report `rotalign_*_answer_mean_logprob` and
  `rotalign_*_answer_mean_nll` when prediction output is enabled.
- This does not change decoding, training, checkpoint format, or translator
  behavior.

Focused unit coverage was added in `tests/test_evaluate_helpers.py`.

## Blocker

An exploratory redundant MPS capacity sweep was launched:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank16_seed1_20260426 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_resampler_capacity_20260426 \
  --seed 1 \
  --run-source-controls
```

The parent sweep was stopped, but its child calibration process remained
orphaned under PID `31103`:

```text
STAT=UE
/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python \
  /Users/sujeethjinesh/Desktop/LatentWire/scripts/calibrate.py ... \
  --device mps --dtype float32 --seed 1 ...
```

`SIGTERM` and `SIGKILL` did not terminate the process. No checkpoint was
materialized; the only produced file is:

- `.debug/gsm8k32_query_resampler_bank16_seed1_20260426/_artifacts/gsm8k_eval_32.jsonl`

This blocks further MPS-backed experiments because the process is stuck in
uninterruptible device/kernel wait. The exact next action is to restart the
machine or otherwise clear PID `31103`, then run the resume command below.

## Existing Artifacts For Resume

- finite query-innovation checkpoint:
  `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
  - sha256: `b1f0cfa62c67ffcbdbce631c6cfd80df3240e132e252b0775aef355940a557b8`
- GSM8K32 eval slice:
  `.debug/gsm8k32_query_innovation_resampler_seed1_20260423/_artifacts/gsm8k_eval_32.jsonl`
  - sha256: `04d3006a6b37aa691347f290d442279bca23bbe119cf9a9b86002263fded20e1`

## Next Exact Gate

After clearing PID `31103`, run matched and source-destroyed answer-likelihood
controls with the existing checkpoint:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
  --eval-file .debug/gsm8k32_query_innovation_resampler_seed1_20260423/_artifacts/gsm8k_eval_32.jsonl \
  --task-type generation \
  --methods rotalign \
  --device mps \
  --dtype float32 \
  --max-new-tokens 16 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --kv-transport k_only \
  --fixed-gate 0.15 \
  --gate-mode fixed \
  --prediction-output results/gsm8k32_query_innovation_answer_likelihood_20260426/matched.jsonl
```

Repeat the same command with:

- `--source-kv-control zero`, output `zero_source.jsonl`
- `--source-prompt-control shuffle_examples --random-salt 1`, output
  `shuffled_source_salt1.jsonl`
- `--innovation-memory-control target_only`, output `target_only.jsonl`
- `--innovation-memory-control slots_only`, output `slots_only.jsonl`

Promotion rule for the next rung: matched source must improve answer
mean-logprob on target-wrong/source-headroom IDs over every source-destroying
control, with no target-correct regression if generation is also inspected.

## Tests

- `./venv_arm64/bin/python -m py_compile latent_bridge/evaluate.py`
- `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -q`
  - `103 passed`
- `./venv_arm64/bin/python -m pytest tests/test_translator_core.py::test_query_innovation_perceiver_connector_fit_and_runtime_are_finite tests/test_translator_core.py::test_query_innovation_anti_memory_control_fit_is_finite tests/test_translator_core.py::test_fit_from_pairs_query_innovation_forwards_source_controls -q`
  - `3 passed`

## File Hashes

- `latent_bridge/evaluate.py`
  - sha256: `f143d4c301f783a607e2647fbc2f1efc9e0097d590d37ed28ea6964e1d7268b7`
- `tests/test_evaluate_helpers.py`
  - sha256: `5c2c03120642487cd2b1ec96e98f4ff91e12732b4ffa6dcc4c2069d82a28e3ea`

## Decision

The query-innovation branch is still the live branch, but this cycle ends at a
hard tooling blocker rather than a scientific pass/fail. Do not start another
MPS run until PID `31103` is gone. Once cleared, the answer-likelihood gate
above is the next highest-value command.
