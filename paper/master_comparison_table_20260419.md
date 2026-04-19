# Master Comparison Table (2026-04-19)

This table is the current paper-facing snapshot for the main Qwen same-pair setting.
It is intentionally narrow: exact held-out settings that have direct baseline
comparisons and a tracked readout in `latent_bridge/current_readout_20260418.md`.

## Qwen2.5-0.5B -> Qwen3-0.6B, GSM8K

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_eval_70` | `target-alone` | `0.0429` | `0` | no source communication |
| `gsm8k_eval_70` | `text-to-text` | `0.1000` | `-` | text communication baseline |
| `gsm8k_eval_70` | fixed head prior | `0.0857` | `151,163.7` | best current internal same-pair branch |
| `gsm8k_eval_70` | shuffled fixed prior | `0.0429` | `151,163.7` | query-blind null |
| `gsm8k_eval_70` | grouped signature transport | `0.0429` | `147,812.6` | best current transport-only branch |
| `gsm8k_eval_70` | grouped subspace transport | `0.0429` | `147,812.6` | tied grouped signature transport |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual | `0.0571` | `145,508.8` | best current transport-plus-correction branch |
| `gsm8k_eval_70` | grouped canonical transport | `0.0286` | `149,496.2` | low-rank canonical basis shortcut |
| `gsm8k_eval_70` | `C2C` | `0.1286` | `-` | strongest external baseline so far |
| `gsm8k_eval_70` | lifted `KVComm` replay | `0.0000` | `-` | compatibility-lifted heterogeneous replay |

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_100` | `target-alone` | `0.0400` | `0` | no source communication |
| `gsm8k_100` | `text-to-text` | `0.1000` | `-` | text communication baseline |
| `gsm8k_100` | fixed head prior | `0.0700` | `-` | best current internal branch on larger slice |
| `gsm8k_100` | shuffled fixed prior | `0.0400` | `-` | matched null |
| `gsm8k_100` | `C2C` | `0.1100` | `-` | strongest external baseline so far |

## Qwen2.5-0.5B -> Qwen3-0.6B, SVAMP

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `svamp_eval_70` | `target-alone` | `0.0714` | `0` | no source communication |
| `svamp_eval_70` | `text-to-text` | `0.4143` | `-` | text communication baseline |
| `svamp_eval_70` | grouped CCA fixed prior | `0.1714` | `-` | best current internal SVAMP branch |
| `svamp_eval_70` | grouped CCA shuffled null | `0.1286` | `-` | matched query-blind null |
| `svamp_eval_70` | `C2C` | `0.4429` | `-` | strongest external baseline so far |

## Current Read

- Best internal same-pair GSM branch is still the fixed head-prior branch, not transport-first.
- Best external baseline is still `C2C`, and it beats us on both GSM and SVAMP.
- Transport-only branches improved from `grouped_transport` to `grouped_signature_transport`, but they plateaued below the fixed-prior branch and well below `C2C`.
- The first transport-plus-correction branch improves over the pure transport family, but it still does not catch the fixed-prior branch or `C2C`.
- The paper is currently strongest as a **blocker / mechanism** story:
  head-space mismatch and transport quality matter, but the current transport
  family does not yet produce a competitive positive method on the main split.
