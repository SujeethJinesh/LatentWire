# Bytes vs Accuracy Table (2026-04-19)

| Split | Method | Accuracy | Avg bytes | Notes |
|---|---|---:|---:|---|
| gsm8k_eval_70 | target-alone | 0.0429 | 0 | no source communication |
| gsm8k_eval_70 | text-to-text | 0.1000 | - | text communication baseline |
| gsm8k_eval_70 | fixed prior | 0.0857 | 151,163.7 | best current internal same-pair branch |
| gsm8k_eval_70 | shuffled fixed prior | 0.0429 | 151,163.7 | query-blind null |
| gsm8k_eval_70 | grouped signature transport | 0.0429 | 147,812.6 | best current transport-only branch |
| gsm8k_eval_70 | grouped subspace transport | 0.0429 | 147,812.6 | ties grouped signature transport |
| gsm8k_eval_70 | grouped subspace transport + rank-4 residual | 0.0571 | 145,508.8 | best current transport-plus-correction branch |
| gsm8k_eval_70 | grouped subspace transport + rank-4 residual + bridge-ridge correction | 0.0429 | 295,614.9 | first bridge branch that survives held-out slices, but still below the live internal bars |
| gsm8k_eval_70 | grouped subspace transport + rank-4 residual + QK-fidelity budget | 0.0429 | 157,989.2 | query-conditioned per-head budget on top of the best transport-plus-correction checkpoint |
| gsm8k_eval_70 | grouped covariance transport + rank-4 residual | 0.0143 | 146,417.7 | covariance-aware transport-plus-correction failure |
| gsm8k_eval_70 | grouped template transport + rank-4 residual | 0.0429 | 150,038.8 | attention-template transport-plus-correction probe (64-prompt calibration slice) |
| gsm8k_eval_70 | grouped template-subspace transport + rank-4 residual | 0.0143 | 149,129.8 | stacked grouped-penalty failure (64-prompt calibration slice) |
| gsm8k_eval_70 | broadcast template transport + rank-4 residual | 0.0000 | 149,129.8 | rectangular 2->8 head transport probe (64-prompt calibration slice) |
| gsm8k_eval_70 | broadcast template OT transport + rank-4 residual | 0.0000 | 149,129.8 | rectangular Sinkhorn-style 2->8 head transport probe (64-prompt calibration slice) |
| gsm8k_eval_70 | broadcast peak-template OT transport + rank-4 residual | 0.0143 | 149,129.8 | rectangular Sinkhorn-style 2->8 transport using peak-location templates (64-prompt calibration slice) |
| gsm8k_eval_70 | broadcast retrieval-spectrum OT transport + rank-4 residual | 0.0143 | 625,463.7 | rectangular Sinkhorn-style 2->8 transport using retrieval-weighted key spectra under matched sparse K-only evaluation (64-prompt calibration slice) |
| gsm8k_eval_70 | broadcast QK-template OT transport + rank-4 residual | 0.0143 | 625,463.7 | rectangular Sinkhorn-style 2->8 transport using last-token QK logit templates under matched sparse K-only evaluation (64-prompt calibration slice) |
| gsm8k_eval_70 | grouped canonical transport | 0.0286 | 149,496.2 | low-rank canonical basis shortcut |
| gsm8k_eval_70 | C2C | 0.1286 | - | strongest external baseline so far |
| gsm8k_eval_70 | KVComm-compatible replay | 0.0000 | - | compatibility-lifted heterogeneous replay |
| gsm8k_100 | target-alone | 0.0400 | 0 | no source communication |
| gsm8k_100 | text-to-text | 0.1000 | - | text communication baseline |
| gsm8k_100 | fixed prior | 0.0700 | 152,953.4 | best current internal branch on larger slice |
| gsm8k_100 | shuffled fixed prior | 0.0400 | 152,953.4 | matched null |
| gsm8k_100 | C2C | 0.1100 | - | strongest external baseline so far |
| svamp_eval_70 | target-alone | 0.0714 | 0 | no source communication |
| svamp_eval_70 | text-to-text | 0.4143 | - | text communication baseline |
| svamp_eval_70 | grouped CCA fixed prior | 0.1714 | - | best current internal SVAMP branch |
| svamp_eval_70 | grouped CCA shuffled null | 0.1286 | - | matched query-blind null |
| svamp_eval_70 | C2C | 0.4429 | - | strongest external baseline so far |
