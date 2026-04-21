# Competitor Next Runnable Matrix

Date: 2026-04-21

This memo ranks the next paper-facing comparison rows by paper value versus cost, using the local bootstrap/readout state and the current competitor literature. The current evidence says the right next move is not another smoke; it is to extend the exact Qwen pair onto the held-out GSM/SVAMP rows while keeping the same prompt, scorer, and decode budget.

## Normalization Rules

- Direct communication rows must keep the same source/target pair, prompt template, answer extractor, and `max_new_tokens`.
- Same-model compression rows must keep the same target model, prompt family, eval split, and decode budget.
- Report accuracy, latency, tokens/sec, and any transfer or repair cost together.
- If a row uses a calibration sweep or a published artifact download, freeze that artifact in the sidecar before comparing to anything else.

## Ranked Matrix

| Rank | Comparison | Exact command(s) | Required artifacts | Fairness normalization | Blockers |
| --- | --- | --- | --- | --- | --- |
| 1 | `C2C` on `gsm8k_eval_70` for the exact `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` pair. Highest paper value because it is the cleanest direct semantic-communication bar on the main GSM held-out split. | `./venv_arm64/bin/python scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --max-new-tokens 64 --prediction-output results/competitor_next_runnable_20260421/c2c_gsm70_qwen25_to_qwen3.jsonl` | `c2c_gsm70_qwen25_to_qwen3.jsonl` and `.meta.json`; C2C bootstrap manifest from `scripts/bootstrap_c2c.py` if you want the artifact header to be explicit. | Same source/target pair, same GSM split, same chat template path, same decode budget, same scorer. Do not compare raw bytes unless the published fuser/config metadata is normalized too. | First use may download the published C2C fuser bundle. |
| 2 | `C2C` on `svamp_eval_70` for the exact pair. This is the best second direct-communication row because it checks whether the GSM result is task-specific or stable across arithmetic phrasing. | `./venv_arm64/bin/python scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device mps --max-new-tokens 64 --prediction-output results/competitor_next_runnable_20260421/c2c_svamp70_qwen25_to_qwen3.jsonl` | `c2c_svamp70_qwen25_to_qwen3.jsonl` and `.meta.json`. | Same source/target pair, same SVAMP split, same decode budget, same extractor. Keep the scoring contract identical to the GSM row. | Same C2C artifact download caveat as row 1. |
| 3 | `KVComm` on `gsm8k_eval_70` with the exact pair. This is the closest multi-agent direct communication peer and the main test for whether the ported KV-sharing path generalizes beyond the small smoke rows. | `./venv_arm64/bin/python scripts/run_kvcomm_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --calibration-file data/gsm8k_100.jsonl --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --source-reasoning-mode brief_analysis --top-layers-grid 0.25,0.5,0.75,1.0 --prediction-output results/competitor_next_runnable_20260421/kvcomm_gsm70.jsonl` | `kvcomm_gsm70.jsonl` and `.meta.json`; the sidecar includes the calibration sweep, chosen layer fraction, and selected layers. | Same source/target pair, same calibration split, same `brief_analysis` source reasoning mode, same decode budget, same answer extractor. Freeze the calibration sweep once chosen. | The port still depends on local compatibility patching inside the vendored KVComm clone. Compare bytes only after normalizing the selected-layer and anchor metadata. |
| 4 | `KVPress` on `gsm8k_eval_70`, run as `none` versus `expected_attention`. This is the cheapest fairness control for deciding whether a gain is communication or just same-model cache policy. | `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_none.jsonl` and `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/gsm8k_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --prediction-output results/competitor_next_runnable_20260421/kvpress_gsm70_expected_attention_c050.jsonl` | Two prediction JSONL files plus two `.meta.json` files. | Same target model, same GSM split, same prompt family, same decode budget, same tokenizer, same device/dtype. Treat `none` as the floor and `expected_attention` as the compression comparator. | Not a cross-model communication baseline; this is a same-model control only. |
| 5 | `KVPress` on `svamp_eval_70`, again `none` versus `expected_attention`. This is worth running because the GSM result alone is not enough to rule out prompt-family sensitivity. | `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press none --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_none.jsonl` and `./venv_arm64/bin/python scripts/run_kvpress_eval.py --model Qwen/Qwen3-0.6B --eval-file data/svamp_eval_70.jsonl --device mps --dtype float32 --max-new-tokens 64 --press expected_attention --compression-ratio 0.5 --prediction-output results/competitor_next_runnable_20260421/kvpress_svamp70_expected_attention_c050.jsonl` | Two prediction JSONL files plus two `.meta.json` files. | Same as row 4, but on SVAMP. Keep the prompt budget fixed so the row is a true compression control rather than a prompt rewrite. | Same-model control only; useful for fairness, not for a direct semantic-transfer claim. |
| 6 | `C2C` on `gsm8k_100`. This is the strongest larger-sample anchor after the held-out GSM70/SVAMP70 rows and gives the direct peer a slightly less noisy comparison point. | `./venv_arm64/bin/python scripts/run_c2c_eval.py --source-model Qwen/Qwen2.5-0.5B-Instruct --target-model Qwen/Qwen3-0.6B --eval-file data/gsm8k_100.jsonl --device mps --max-new-tokens 64 --prediction-output results/competitor_next_runnable_20260421/c2c_gsm100_qwen25_to_qwen3.jsonl` | `c2c_gsm100_qwen25_to_qwen3.jsonl` and `.meta.json`. | Same source/target pair, same GSM family, same decode budget, same scorer. | Same C2C artifact-download caveat as rows 1 and 2. |

## What The Web Adds

The current direct-competitor watchlist is broader than the local runnable set, but most of the newer items do not yet have a clean local harness here. The most relevant sources are:

- `Latent Space Communication via K-V Cache Alignment` https://arxiv.org/abs/2601.06123
- `Q-KVComm` https://arxiv.org/abs/2512.17914
- `HCAttention` https://arxiv.org/abs/2507.19823
- `KQ-SVD` https://arxiv.org/abs/2512.05916

These are worth tracking next, but they are not top-6 runnable rows in this repo today because the local code or benchmark path is not yet as clean as C2C, KVComm, or KVPress.

## Bottom Line

- Run rows 1 to 3 first if you want the fastest paper-relevant update.
- Run rows 4 and 5 in the same batch so the direct-communication rows are interpreted against the correct same-model floor.
- Run row 6 when you want a larger C2C anchor without changing the model pair or scorer.
