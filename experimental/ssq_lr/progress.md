# SSQ-LR Progress

## 2026-05-06

Status: **NEW / Mac gates pending**.

Added and ran a deterministic synthetic S1 packet:

- script: `phase2/ssq_lr_synthetic_s1_gate.py`
- packet: `phase2/results/ssq_lr_synthetic_s1/`
- decision: `SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT`
- late/early max-abs ratio: `8.461`
- late/early std ratio: `3.640`
- late/early kurtosis ratio: `3.141`

Interpretation: synthetic-only artifact validation. It fixes the S1 readout
format but does not promote the branch or replace real hybrid SSM state dumps.

Next exact gate: S1 state distribution heterogeneity on the smallest available
hybrid model traces using shared activation/state dump utilities.
