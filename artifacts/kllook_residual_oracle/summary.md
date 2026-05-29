# Restricted KLLOOK Residual Oracle

Status: `NO_RUN_EXCEEDS_ONE_HOUR_RUNNER_GATE`.

This packet does **not** claim a KLLOOK oracle result. The existing `run_om_phase9_mkllook.py` code is an original-basis protected-channel oracle. The requested test is different: post-ParoQuant rotated-basis residual correction, with columns selected by output-distribution benefit. Reusing the original-basis runner would mix two mechanisms and would not answer the gate.

I inspected the available runner and residual subset code. A correct runner would need to apply post-ParoQuant residual sidecars per candidate or group, collect BF16/current/corrected logits, compute forward KL, and then verify on held-out representative traces. That is not a <=1 hour implementation-and-verification change. Per the gate, this is a NO_RUN.

The pass bar remains intentionally high: an oracle result would need large tail improvement, no representative-trace harm, a selected set materially different from the killed K-RES proxy, and acceptable HBM/energy overhead. A small calibration-tail gain would not reopen selector search.

What we do know from the valid K-RES endpoint proxy:

- Granite tail trace I_4 top-8x32 residual correction recovery: `-12.294`.
- Tight ParoQuant reference recovery on the same trace/window: `5.940`.
- Margin: `-18.233`.

Decision: the residual family is **not promoted**. It can only be reopened by a new preregistered rotated-basis residual KLLOOK runner with per-candidate logits/KL and a calibration/confirmation split.
