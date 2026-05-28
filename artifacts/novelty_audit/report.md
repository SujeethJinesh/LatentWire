# C10 Novelty Audit: Rotation-First Positive-Method Sprint

Created: 2026-05-28T15:38:16Z

## Scope

This audit compares the candidate DriftRot-style claims against the closest
rotation, smoothing, and SSM quantization priors:

- ParoQuant: pairwise Givens rotations plus channel-wise scaling for efficient
  reasoning LLM inference.
- QuaRot: computationally invariant rotations across residual, feed-forward,
  attention, and KV-cache surfaces for 4-bit inference.
- SpinQuant: learned rotations for quantized network accuracy.
- Quamba2: selective-SSM PTQ using sorting/clustering and SSM activation
  persistence assumptions.
- SmoothQuant and AWQ: activation-aware smoothing/scaling and salient-weight
  protection.
- MambaQuant: KLT-enhanced and smooth-fused rotations for Mamba-family PTQ.
- Rotated Runtime Smooth: runtime activation smoothing plus rotation for INT4.

Two additional 2025--2026 guardrail sources are included because they affect
safe wording:

- OSCAR: offline spectral covariance-aware rotation for INT2 KV-cache
  quantization on long-context/reasoning workloads.
- KVLinC: Hadamard rotation plus linear correction for KV-cache quantization.

## Bottom Line

The safe contribution boundary is narrow. A broad claim like "we introduce
rotation-based quantization for long reasoning" is already scooped. The viable
claim is conditional:

> We test whether long-decode drift diagnostics can select or retune an existing
> rotation family, surface, branch, clip/scale configuration, or residual
> correction under a calibration/confirmation split; if it improves held-out
> recovery, CI lower bound, or CVaR tail over the fixed ParoQuant baseline, that
> drift-aware selector is the contribution.

If held-out gains do not materialize, the safest paper claim is mechanism-first:
channel identity drifts under long reasoning, while rotation-style conditioning
removes much of the basis dependence; static ParoQuant is a strong baseline, not
our method.

## Candidate Gates

### DriftRot Scale/Clip Retune

- **Novelty risk:** HIGH.
- **Closest prior:** ParoQuant already combines pairwise rotations with
  channel-wise scaling; Rotated Runtime Smooth adjusts runtime activation
  smoothing; SmoothQuant/AWQ cover activation-aware scaling; OSCAR derives fixed
  rotations and clipping thresholds from covariance structures, albeit for
  INT2 KV cache rather than W4A16 weights.
- **Safe claim:** "A drift-calibrated clip/scale retune for ParoQuant-style
  W4A16 long-reasoning evaluation, selected on calibration traces and confirmed
  on held-out traces."
- **Unsafe claim:** "New rotation/scaling quantization method."
- **Gate:** Proceed only if C1/C2 show late-vs-early range or covariance drift
  that ParoQuant's static config does not cover, and G2 improves ParoQuant
  held-out median by at least 0.05, CI lower by at least 0.10, or CVaR tail
  without median loss above 0.05.

### Drift-Aware Pairing

- **Novelty risk:** MEDIUM-HIGH.
- **Closest prior:** ParoQuant is directly pairwise rotation; SpinQuant learns
  rotations; QuaRot and MambaQuant cover rotation parameterization and variance
  alignment more broadly.
- **Safe claim:** "A decode-drift-aware pairing criterion for an existing
  pairwise-rotation baseline, with material pairing differences from ParoQuant
  and held-out gains."
- **Unsafe claim:** "Pairwise rotation quantization."
- **Gate:** Promote only if C4 proves pairings materially differ from the
  ParoQuant baseline and predicted range reduction is not a calibration-only
  artifact. Without that, kill as ParoQuant hyperparameter search.

### Rotated Residual Correction

- **Novelty risk:** HIGH.
- **Closest prior:** KVLinC combines Hadamard rotation and linear correction for
  KV-cache quantization; AWQ protects salient weight channels via equivalent
  scaling; QuaRot and ParoQuant already reduce rotation-induced quantization
  error.
- **Safe claim:** "A protected-column residual correction after ParoQuant-style
  W4A16 weight-only PTQ, selected by long-decode activation/error diagnostics
  and evaluated against ParoQuant on held-out traces."
- **Unsafe claim:** "Rotation plus residual/linear correction" without
  specifying that the correction is for W4A16 weight residual columns rather
  than KV-cache correction.
- **Gate:** Keep only if G3 fixes ParoQuant tail traces or wins at least two of
  three smoke traces with matched overhead. Do not implement a kernel until
  partial evaluation survives.

### BranchRot

- **Novelty risk:** MEDIUM.
- **Closest prior:** QuaRot applies rotations to several Transformer surfaces;
  MambaQuant and Quamba2 target Mamba/SSM-specific quantization; no checked
  primary source directly claims Falcon-H1 parallel-branch-local rotation under
  long-decode channel drift.
- **Safe claim:** "A branch-local drift diagnostic and optional branch-local
  rotation/protection for parallel hybrid heads, promoted only when branch
  surfaces drift materially less than post-mixer outputs."
- **Unsafe claim:** "Architecture-aware rotation for hybrid models" without
  branch-local evidence.
- **Gate:** Promote only if C6/G5 finds branch-local drift or covariance range
  at least 0.15 better than post-mixer and recovery improves under matched
  cost.

### M-SURFACE Rotation/Protection

- **Novelty risk:** MEDIUM-HIGH.
- **Closest prior:** QuaRot already rotates residual, FFN, attention, and KV
  cache surfaces; MambaQuant uses KLT and smooth-fused rotations for Mamba
  surfaces; Quamba2 rearranges SSM weights offline using persistence in SSM
  quantization internals.
- **Safe claim:** "A measured-surface selection protocol for long-decode
  W4A16 drift, identifying whether internal hybrid-model surfaces are more
  stable than block outputs before applying rotation/protection."
- **Unsafe claim:** "First surface-aware rotation/protection for LLMs."
- **Gate:** Promote only if G6 finds an internal surface with strict leaving
  below 0.30--0.40 or at least 0.15 below block output, then improves recovery
  on confirmation traces.

## Method-Level Recommendation

1. Run ParoQuant Falcon and DeepSeek smoke first. If ParoQuant passes both,
   DriftRot must beat/robustify a very strong static rotation baseline; the
   paper becomes mechanism/protocol unless G2/G3 finds a tail improvement.
2. Run C1/C2/C3/C4 offline gates before any DriftRot scale/clip/pairing smoke.
   If the only change is a retuned ParoQuant knob with no drift-conditioned
   confirmation, do not call it a method.
3. Prefer residual correction only as a tail-fixer, not as a broad rotation
   claim.
4. BranchRot and M-SURFACE are the least-scooped directions because they are
   architecture/surface diagnostics, but they require direct surface evidence.

## Source Notes

- ParoQuant is the direct baseline pressure: pairwise rotations plus
  channel-wise scaling for reasoning LLM inference.
- QuaRot and SpinQuant make generic "rotation improves quantization" claims
  unsafe.
- SmoothQuant/AWQ make generic "activation-aware scale/protect salient
  channels" claims unsafe.
- MambaQuant and Quamba2 make broad Mamba/SSM-specific rotation or persistence
  claims risky without measured-surface precision.
- Rotated Runtime Smooth makes runtime activation smoothing plus rotation
  adjacent to any ScaleRefresh/ClipRetune branch.
- OSCAR makes "covariance-aware rotation for long-context/reasoning" risky
  unless scoped away from KV cache and INT2.
- KVLinC makes "rotation plus correction" risky unless scoped to W4A16
  post-rotation weight residual columns.

