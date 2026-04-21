# Master Comparison Table (2026-04-19)

This table is the current paper-facing snapshot for the main Qwen same-pair setting.
It is intentionally narrow: exact held-out settings that have direct baseline
comparisons and a tracked readout in `latent_bridge/current_readout_20260418.md`.

## Qwen2.5-0.5B -> Qwen3-0.6B, GSM8K

### GSM30 Stochastic-Route Smoke

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_gate_search_30` | target-alone | `0.0667` | `0` | current stochastic-reranker baseline |
| `gsm8k_gate_search_30` | strict stochastic selector | `0.1667` | `-` | best current non-oracle selector over three random route/value candidates |
| `gsm8k_gate_search_30` | target-model listwise verifier | `0.0667` | `-` | selected target-alone on all 30 examples; useful negative selector control |
| `gsm8k_gate_search_30` | target-or-seed oracle | `0.3000` | `-` | candidate-quality ceiling, label-leaking |
| `gsm8k_gate_search_30` | C2C native smoke | `0.0667` | `-` | exact Qwen pair through published C2C artifact |
| `gsm8k_gate_search_30` | KVPress none | `0.0667` | `-` | same-model compression control |
| `gsm8k_gate_search_30` | KVPress expected-attention `0.5` | `0.0667` | `-` | same-model compression control |

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_eval_70` | `target-alone` | `0.0429` | `0` | no source communication |
| `gsm8k_eval_70` | `text-to-text` | `0.1000` | `-` | text communication baseline |
| `gsm8k_eval_70` | fixed head prior | `0.0857` | `151,163.7` | best current internal same-pair branch |
| `gsm8k_eval_70` | shuffled fixed prior | `0.0429` | `151,163.7` | query-blind null |
| `gsm8k_eval_70` | grouped signature transport | `0.0429` | `147,812.6` | best current transport-only branch |
| `gsm8k_eval_70` | grouped subspace transport | `0.0429` | `147,812.6` | tied grouped signature transport |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual | `0.0571` | `145,508.8` | best current transport-plus-correction branch |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual + bridge-ridge correction | `0.0429` | `295,614.9` | first bridge branch that survives held-out slices, but still below the live internal bars |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual + QK-fidelity budget | `0.0429` | `157,989.2` | query-conditioned per-head budget on top of the best transport-plus-correction checkpoint |
| `gsm8k_eval_70` | grouped covariance transport + rank-4 residual | `0.0143` | `146,417.7` | covariance-aware transport-plus-correction failure |
| `gsm8k_eval_70` | grouped template transport + rank-4 residual | `0.0429` | `150,038.8` | attention-template transport-plus-correction probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | grouped template-subspace transport + rank-4 residual | `0.0143` | `149,129.8` | stacked grouped-penalty failure (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast template transport + rank-4 residual | `0.0000` | `149,129.8` | rectangular `2 -> 8` head transport probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast template OT transport + rank-4 residual | `0.0000` | `149,129.8` | rectangular Sinkhorn-style `2 -> 8` head transport probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast peak-template OT transport + rank-4 residual | `0.0143` | `149,129.8` | rectangular Sinkhorn-style `2 -> 8` transport using peak-location templates (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast retrieval-spectrum OT transport + rank-4 residual | `0.0143` | `625,463.7` | rectangular Sinkhorn-style `2 -> 8` transport using retrieval-weighted key spectra under matched sparse `K-only` evaluation (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast QK-template OT transport + rank-4 residual | `0.0143` | `625,463.7` | rectangular Sinkhorn-style `2 -> 8` transport using last-token QK logit templates under matched sparse `K-only` evaluation (`64`-prompt calibration slice) |
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
- On the newer GSM30 stochastic-route smoke, the strict selector is the first
  non-oracle internal method to beat target-alone and same-slice C2C/KVPress
  controls (`0.1667` vs `0.0667`), but it is not yet a held-out paper result.
- The naive target-model listwise verifier is a negative control: it chose the
  target candidate on every GSM30 example, so future verifier work needs
  calibration, position randomization, or process-level checks.
- Transport-only branches improved from `grouped_transport` to `grouped_signature_transport`, but they plateaued below the fixed-prior branch and well below `C2C`.
- The first transport-plus-correction branch improves over the pure transport family, but it still does not catch the fixed-prior branch or `C2C`.
- The first bridge-style correction branch that actually survives beyond tiny smokes is `bridge_ridge`, but it still trails the grouped-subspace-plus-rank4 checkpoint and the fixed-prior branch.
- A genuinely query-conditioned QK-fidelity budget on top of that same best transport-plus-correction checkpoint recovers only to `0.0429`, so live query-conditioning alone is still not enough.
- A covariance-aware version of that same transport-plus-correction branch falls back to `0.0143`, so covariance geometry is not the next shortcut here.
- A calibration-time attention-template version of that same branch lands at `0.0429`, so light behavior matching inside the current grouped solver is also not enough.
- A hybrid template-plus-subspace version falls further to `0.0143`, so stacking the two best grouped penalties is not the right fix either.
- A finer rectangular `2 -> 8` broadcast-template transport branch falls all the way to `0.0000`, so the grouped family was not failing only because of coarse grouped transport.
- A richer rectangular Sinkhorn-style OT plan in that same attention-template space still lands at `0.0000`, so the remaining issue is not just transport granularity or many-to-many mass assignment.
- Replacing mean attention templates with simple peak-location templates lifts that OT branch to `0.0143`, so representation matters a bit, but the gain is still far below the fixed prior and `C2C`.
- Replacing the retrieval-spectrum descriptor with simple last-token QK logit templates does not move that frontier at all: it ties the retrieval-spectrum OT branch at `0.0143` while staying far less byte-efficient than the live sparse branches.
- The paper is currently strongest as a **blocker / mechanism** story:
  head-space mismatch and transport quality matter, but the current transport
  family does not yet produce a competitive positive method on the main split.
