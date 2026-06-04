# LatentWire Claim Boundary

- Do not claim broad latent communication from narrow cached-source rows.
- Do not claim native GPU, HBM, energy, or throughput superiority.
- Use combine-not-replace decode by default.
- Treat every cache-screen positive as provisional until held-out confirmation.
- The powered fresh MMLU-Pro oracle ladder is dev/gate screening evidence only: `1500` scored dev/gate rows, `359` gate rows, confirm rows scored `0`.
- One-way negative is now held out: `381` confirm rows, deployable WZ delta `-0.023622` with CI `[-0.060367, 0.013123]`, full source-only oracle delta `-0.031496` with CI `[-0.062992, 0.002625]`, source+target-at-encoder upper bound delta `0.110236` with CI `[0.081365, 0.141732]`.
- Do not call the current score-packet path a positive method. Deployable WZ and full source-only fusion do not beat source-index+confidence; only the source+target-at-encoder upper bound helps. Mechanism shorthand: `2.10 -> 0.28` bits after receiver conditioning.
- L_Q1 is killed as a final query-conditioned two-way score-packet shot: delta `-0.022284` versus source-index+confidence, controls did not collapse, and ablations explain it.
- Do not claim the CacheWire/C2C deployable trace path is killed from the `n=32` overnight triage. That slice is `INCONCLUSIVE_UNDERPOWERED`; the dense C2C teacher remains a headroom hint (`16/32` vs target `8/32`), while current compact packet smokes are killed only where deterministic controls explain them.
- Deterministic LatentWire/C2C kills that are safe to cite as method evidence failures: KVComm matched cache equals zero-source predictions in the inspected smoke, generated-answer value/index packets equal same-byte answer-text leakage, C2C teacher-delta ties zero/target controls, and candidate-pool deltas are dominated by coefficient controls.
- Allowed LatentWire claim: bounded negative / diagnostic failure localization for source-copy leakage, receiver-conditioned controls, and non-deployable oracle headroom.
- Next possible positive paths are: powered CacheWire ceiling from real source/receiver hidden-cache features, and L-A2 only after a generated-solution candidate-pool cache with verifier/source/target scores exists. Both remain dev/gate only before confirmation.
