# Current Bytes / Accuracy Frontier (2026-04-20)

| Split | Method | Family | Accuracy | Avg bytes | Notes |
|---|---|---|---:|---:|---|
| gsm8k_eval_70 | target-alone | control | 0.0429 | 0 | no source communication |
| gsm8k_eval_70 | text-to-text | control | 0.1000 | - | text communication baseline |
| gsm8k_eval_70 | shuffled fixed prior | selector-null | 0.0429 | 151,163.7 | query-blind prior null |
| gsm8k_eval_70 | fixed prior | selector | 0.0857 | 151,163.7 | best internal same-pair branch |
| gsm8k_eval_70 | grouped signature transport | transport | 0.0429 | 147,812.6 | best transport-only branch |
| gsm8k_eval_70 | grouped subspace + rank-4 residual | transport+correction | 0.0571 | 145,508.8 | best transport-plus-correction branch |
| gsm8k_eval_70 | bridge_ridge | bridge | 0.0429 | 295,614.9 | best bridge branch that survives held-out slices |
| gsm8k_eval_70 | QK-fidelity budget | query-conditioned selector | 0.0429 | 157,989.2 | runtime query-conditioned selector on top of best transport+correction checkpoint |
| gsm8k_eval_70 | KVComm-compatible replay | external comparator | 0.0000 | - | adjacent heterogeneous replay baseline |
| gsm8k_eval_70 | C2C | external comparator | 0.1286 | - | main external fair bar |
| gsm8k_eval_10_controlled | target-alone | control | 0.1000 | 0 | shared chat serialization + enable_thinking=false |
| gsm8k_eval_10_controlled | bridge_ridge | bridge | 0.1000 | 340,376.0 | fair controlled bridge baseline |
| gsm8k_eval_10_controlled | shared-plus-private asym adapter | modular bridge | 0.1000 | 681,668.4 | AsymLoRA-style shared bottleneck plus private K/V residual heads |
| gsm8k_eval_10_controlled | shared-plus-private dynmap adapter | modular bridge | 0.1000 | 681,668.4 | shared-plus-private bridge with context-reweighted top-k teacher |
| gsm8k_eval_10_controlled | xattn adapter | attention bridge | 0.1000 | 681,668.4 | tiny query-conditioned cross-attention bridge over live K/V-side memory signals |
| gsm8k_eval_10_controlled | grouped rotational transport | geometry | 0.1000 | 681,668.4 | first geometry-side branch to survive the controlled slice |
| gsm8k_eval_10_controlled | grouped fitted rotation transport | geometry | 0.1000 | 681,668.4 | calibration-fit gauge-fixing follow-up |
| gsm8k_eval_10_controlled | grouped shared-basis transport | geometry | 0.1000 | 681,668.4 | shared-basis coefficient-space transport |
| gsm8k_eval_10_controlled | KVPress no-press | external comparator | 0.1000 | - | exact external KVPress harness floor |
| gsm8k_eval_10_controlled | KVPress ExpectedAttentionPress | external comparator | 0.1000 | - | exact external Expected Attention comparator |
| gsm8k_5_controlled_smoke | shared-plus-private asym adapter | modular bridge | 0.2000 | 686,026.6 | AsymLoRA-style shared-plus-private bridge survives smoke and controlled slice |
| gsm8k_5_controlled_smoke | shared-plus-private dynmap adapter | modular bridge | 0.2000 | 686,026.6 | shared-plus-private bridge with context-reweighted top-k teacher |
| gsm8k_5_controlled_smoke | xattn adapter | attention bridge | 0.2000 | 686,026.6 | tiny query-conditioned cross-attention bridge over live K/V-side memory signals |
| gsm8k_5_controlled_smoke | shared-plus-private asym projector | projector bridge | 0.0000 | 686,026.6 | shared-plus-private post-transport projector combining full-rank query projector with the paired K/V interface |
| gsm8k_5_controlled_smoke | readout adapter | stronger-teacher bridge | 0.2000 | 686,026.6 | stronger prompt-local teacher survives smoke only |
| gsm8k_5_controlled_smoke | prediction-KL adapter | stronger-teacher bridge | 0.0000 | 722,107.7 | first prediction-level bridge teacher |
| gsm8k_5_controlled_smoke | prediction-KL bank | stronger-teacher bridge | 0.0000 | 722,107.7 | small modular bank follow-up to prediction-level teacher |
