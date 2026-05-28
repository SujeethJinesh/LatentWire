# Abstract Options

## A. DriftRot Positive Method

Long-reasoning W4A16 inference exposes a mismatch between static channel
protection and decode-time activation structure: top activation channels leave
their initial protected set across hybrid and Transformer models, and hard
position switching or simple prediction often performs worse than static or
random controls. We show that rotation-based conditioning is a stronger
starting point than channel identity on the two measured positive regimes:
ParoQuant-style rotation recovers 0.754 on Granite-4.0-H-Small and 1.047 on
Nemotron-3-Nano, exceeding Nemotron M11b top-10 by 0.232. Building on this, we
introduce DriftRot, a long-decode rotation-calibration protocol that selects
rotation surfaces, retunes scale/clip ranges under drift, and optionally adds
matched-cost residual correction. DriftRot improves static rotation on
held-out traces by [fill after confirmation], while failed channel-set methods
explain why the gain comes from reducing basis dependence rather than tracking
unstable channel identities.

Use only if a DriftRot candidate beats or robustifies static ParoQuant on
confirmation traces.

## B. Rotation-Dominant Mechanism / Protocol Paper

Long-reasoning W4A16 inference challenges a common assumption behind
channel-protection quantization: the identity of important activation channels
is not stable over decode. Across four model families, strict top-channel
set-leaving is large, and static, hard-switching, EMA, predictive, and
composition policies either fail, remain model-local, or show wide
trace-level uncertainty. The positive result is rotation: ParoQuant-style
rotation recovers 0.754 on Granite-4.0-H-Small and 1.047 on
Nemotron-3-Nano, beating Nemotron M11b top-10 by 0.232 on the same
BF16/static baseline. We propose a rotation-first calibration protocol for
long reasoning: test rotation before channel-set protection, use drift and
tail diagnostics to reject fragile channel policies, and reserve branch or
surface-local remedies for models where rotation fails. This turns the
negative method catalog into design logic: channel identity drifts, but
rotation can remove much of the basis dependence.

Use if ParoQuant passes DeepSeek/Falcon or DriftRot does not improve static
ParoQuant.

## C. Falcon Branch / Surface Rescue

Long-reasoning W4A16 protection is not uniformly solved by dynamic channel
tracking. Static, hard-switching, EMA, and predictive policies fail or remain
ambiguous on multiple architectures, with Falcon-H1 resisting M11b top-10
almost completely. Rotation is strong on Granite and Nemotron, but Falcon
tests whether parallel hybrid mixers require a more local treatment. We
measure decode-time drift at branch and internal projection surfaces and
introduce branch-local rotation/protection for surfaces whose drift is
materially lower than post-block activations. If successful, this yields a
regime-aware method: global rotation for dense/hybrid regimes where it works,
and branch/surface-local protection for parallel hybrid regimes where global
channel identity is too unstable.

Use only if Falcon ParoQuant is weak and branch/surface diagnostics improve
matched-cost recovery.

