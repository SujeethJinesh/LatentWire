# Figure Plan

## Figure 1: Drift

Purpose: show the empirical problem. Plot strict top-channel set-leaving over
decode position for Granite, Nemotron, DeepSeek, and Falcon.

Message: channel identity changes during long reasoning, so static top-K
protection is not a safe default.

## Figure 2: Channel Failure

Purpose: convert failed methods into design logic. A compact forest/bar plot
shows recovery and CI for static, M2/M10, M11/M11b, M-PRED, M26, DecDEC, and
ParoQuant+M11b.

Message: discontinuous switching, under-budgeted EMA, prediction, and naive
composition do not solve the drift problem.

## Figure 3: Rotation

Purpose: highlight the new positive baseline. Plot ParoQuant-style rotation on
Granite and Nemotron against M11b and static controls.

Required data:
- Granite ParoQuant: median 0.754, CI95 [0.477, 1.004], 8 included traces.
- Nemotron ParoQuant: median 1.047, CI95 [1.007, 1.292], 10 included traces.
- Nemotron M11b top-10: median 0.815, CI95 [0.254, 0.923].

Message: rotation is currently stronger than channel tracking on the measured
positive regimes.

## Figure 4: Adaptive Rotation Headroom

Purpose: show where DriftRot could add something new. Visualize the decision
tree from static ParoQuant to scale/clip retuning, pairing refresh,
surface/branch rotation, or residual correction.

Message: DriftRot is only a paper method if it beats or robustifies ParoQuant
on held-out traces. Otherwise, the figure becomes a protocol figure showing
why static rotation is surprisingly robust.

