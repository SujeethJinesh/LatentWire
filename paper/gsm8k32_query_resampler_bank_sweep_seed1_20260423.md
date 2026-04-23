# GSM8K32 Query-Resampler Bank Sweep Seed1

Date: 2026-04-23

## Status

Paper readiness remains not ICLR-ready. The live same-family method surface is
finite and reproducible on GSM8K32 seed1, but it still has no target-safe
positive row.

Current story: guarded query-resampler replacement made the seed1 learned
connector finite. A true no-private-slot row can change target outcomes, so the
surface is source-conditioned rather than a pure inert target cache. Increasing
private slot capacity to 16 suppresses that perturbation back to exact target
parity instead of producing net gains.

Blocking gap: candidate-only wins must exceed losses under full artifact
validity, numeric coverage, exact ID parity, and source/null controls. This
sweep does not clear that gate.

## Decision Surface

Top next moves considered:

- Capacity/null sweep: tests whether the guarded query-resampler is simply
  under-capacity. Failure mode is target parity or noisy wins/losses. Evidence
  gained is direct same-pair robustness and reproducibility at low compute cost.
- Innovation/residual-only query-resampler: transmit source innovation relative
  to target side information instead of replacing the full target-like cache.
  Failure mode is clipping to target parity or adding regressions. Evidence
  gained would directly test a cleaner positive-method contribution.
- Full null harness upgrade: add explicit zero-source/shuffled-source rows and
  artifact hashes/manifests. Failure mode is infrastructure work displacing the
  method gate. Evidence gained is reviewer-grade reproducibility.

Selected move this turn: capacity/null sweep plus minimal provenance upgrade.

## Commands

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 0 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank0_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_resampler_bank_sweep_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_resampler_bank_sweep_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/analyze_gsm8k_contract_diagnostics.py \
  --candidate-prediction-output .debug/gsm8k32_query_resampler_bank0_seed1_20260423/dynalign_query_resampler_replace_residrank16_bank0.jsonl \
  --candidate-label dynalign_query_resampler_replace_residrank16_bank0 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank0_seed1_20260423 \
  --output-tag bank0_seed1_20260423

./venv_arm64/bin/python scripts/analyze_gsm8k_contract_diagnostics.py \
  --candidate-prediction-output .debug/gsm8k32_query_resampler_bank16_seed1_20260423/dynalign_query_resampler_replace_residrank16_bank16.jsonl \
  --candidate-label dynalign_query_resampler_replace_residrank16_bank16 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank16_seed1_20260423 \
  --source-output-name ../gsm8k32_query_resampler_bank0_seed1_20260423/gsm8k32_source_alone.jsonl \
  --output-tag bank16_seed1_20260423
```

## Results

| Bank | Status | Accuracy | Paired vs target | Numeric coverage | Empty | Diagnostic status |
|---:|---|---:|---|---:|---:|---|
| 0 | ok | 2/32 | 1 win / 1 loss / 30 ties | 31/32 | 0 | invalid_artifact |
| 4 | ok | 2/32 | 0 wins / 0 losses / 32 ties | 32/32 | 0 | target parity |
| 16 | ok | 2/32 | 0 wins / 0 losses / 32 ties | 32/32 | 0 | target_parity_or_negative |

Bank0 produced one candidate-only non-copy win, but also one target loss and
only 31/32 numeric extraction coverage. It is useful mechanistic evidence that
live-memory source conditioning can perturb target behavior, but it is not a
valid promotable row.

Bank16 is fully valid and finite, but exact target parity. Extra private slots
do not recover the bank0 win; they suppress the perturbation back to target.

## Artifact Hashes

- bank0 checkpoint:
  `29d568668cf0bfb2d3d6638293b937b36dc89685d6bf79edab558cd8e203f543`
- bank0 checkpoint health:
  `3fceee4edf48304e3e52dc6747cb843a193da2cd57ab6952183f4f68c19bfeff`
- bank0 sweep JSON:
  `eb1652f76a58dc16047b83b4ff6fe1d1859bb6ccb3c02753ede4e4ce98e95545`
- bank0 prediction JSONL:
  `be0a22cd165ce2bb5b6dd97fd8194336b2b12a0768432d66e68e513fa88219ed`
- bank0 prediction sidecar:
  `09184f620661452037bd11c6ad08ee7b7c9436828b3b3ae85d93c89bdcd0bb03`
- bank0 diagnostics JSON:
  `1d9a4ac1257c79395fb62ae95b9b463ebf7adafdfd4864ec0cdc85c67cc5c473`
- bank16 checkpoint:
  `900193a8f035b79b4cc4c247d205693b4d99f15a28b44a63b7eab376d56b4a3e`
- bank16 checkpoint health:
  `95aeaaebaa1c3229ab7c9161a8af637f16e233f7993e4cdb0b499b1736a09b1e`
- bank16 sweep JSON:
  `80a70bb6bd8067488f3196653577abc1b0af86ae2f9ac03efb770d943afd9c9e`
- bank16 prediction JSONL:
  `fa37f36b9aa264a23cec26ad3da0ca475553ca66c6aba24efe258c6ff35c49c5`
- bank16 prediction sidecar:
  `39098ebce985876c3d74477053501c25302330c17d8b99e81e3625cd4740f195`
- bank16 diagnostics JSON:
  `40eece8d88af2d3d40a4045b7abec83d4040292c274bb7ea905ddc51bed1cc1c`

## Interpretation

Promoted: target-safe innovation/residual framing is now the highest-value next
branch. The observed bank0 win/loss pair suggests the source-conditioned live
memory can move answers, but full replacement lacks a target-safe mechanism.

Weakened: plain capacity scaling of the guarded full-replacement query
resampler. Bank16 is reviewer-clean but target-parity; bank0 is not valid and
not target-safe.

Blocked: GSM70/cross-family widening. The current live branch has not cleared
seed-stable positive same-family behavior on the frozen smoke slice.

Next exact gate: implement `dynalign_query_innovation_resampler_replace` or an
equivalent target-safe residual branch with a small residual-scale sweep, then
run GSM8K32 seed1 rank16 bank16 with matched zero/shuffled-source controls.
Promotion requires finite checkpoint, 32/32 numeric coverage, exact ID parity,
at least one candidate-only win, zero target losses, and the win disappearing
under shuffled/zero-source controls.
