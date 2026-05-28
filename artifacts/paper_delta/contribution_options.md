# Contribution Options

## A. DriftRot Positive Method

1. We measure decode-time channel-set drift across long reasoning traces and
   show why static and hard-switch channel protection are fragile.
2. We show that ParoQuant-style rotation is a stronger baseline than dynamic
   channel tracking on Granite and Nemotron, with Nemotron recovery 1.047 and
   CI95 [1.007, 1.292].
3. We introduce DriftRot, a drift-aware rotation calibration family that
   retunes scale/clip settings, pairing, surface choice, or residual
   correction against long-decode tail risk.
4. We validate DriftRot only when it beats static ParoQuant or improves
   ParoQuant tail/CI behavior on held-out traces.

## B. Rotation-Dominant Mechanism / Protocol Paper

1. We establish that important activation channel identities move during
   long reasoning across measured architectures.
2. We provide a controlled failure catalog: static, position bins, hard
   switches, top-1% EMA, prediction, stable core, and naive ParoQuant+M11b
   composition do not produce a universal channel-set remedy.
3. We identify rotation as the robust positive baseline in the measured
   positive regimes: ParoQuant-style rotation recovers 0.754 on Granite and
   1.047 on Nemotron.
4. We turn the results into a prospective calibration protocol, evaluated
   descriptively on the current study: test rotation first, use drift/no-gap/
   tail diagnostics to reject fragile channel policies, then run branch or
   surface remedies only where rotation fails.

## C. Falcon Branch / Surface Rescue

1. We show Falcon-H1 is the current stress test for channel-set methods:
   M11b top-10 median recovery is 0.044 with CI95 [-0.144, 0.203].
2. We map branch and internal projection surfaces in parallel hybrid mixers
   and test whether drift is lower before post-mixer aggregation.
3. We introduce branch/surface-local rotation or protection only when the
   diagnostic finds materially lower drift than block output.
4. We integrate the result into a regime-aware rule: rotation where global
   rotation works, branch/surface remedies where architecture-local drift
   makes global channel policies fail.

