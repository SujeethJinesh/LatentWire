# Restricted KLLOOK Residual Oracle

Status: `NOT_EXECUTED_NO_ROTATED_RESIDUAL_KLLOOK_RUNNER`.

This packet does **not** claim a KLLOOK oracle result. The existing `run_om_phase9_mkllook.py` code is an original-basis protected-channel oracle. The requested test is different: post-ParoQuant rotated-basis residual correction, with columns selected by output-distribution benefit. Reusing the original-basis runner would mix two mechanisms and would not answer the gate.

What we do know from the valid K-RES endpoint proxy:

- Granite tail trace I_4 top-8x32 residual correction recovery: `-12.294`.
- Tight ParoQuant reference recovery on the same trace/window: `5.940`.
- Margin: `-18.233`.

Decision: the residual family is **not promoted**. It can only be reopened by writing a new rotated-basis residual KLLOOK runner with per-candidate logits/KL and a calibration/confirmation split.
