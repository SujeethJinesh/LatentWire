# LatentWire Claim Boundary

- Do not claim broad latent communication from narrow cached-source rows.
- Do not claim native GPU, HBM, energy, or throughput superiority.
- Use combine-not-replace decode by default.
- Treat every cache-screen positive as provisional until held-out confirmation.
- The powered fresh MMLU-Pro oracle ladder is dev/gate screening evidence only: `1500` scored dev/gate rows, `359` gate rows, confirm rows scored `0`.
- One-way negative is now held out: `381` confirm rows, deployable WZ delta `-0.023622` with CI `[-0.060367, 0.013123]`, full source-only oracle delta `-0.031496` with CI `[-0.062992, 0.002625]`, source+target-at-encoder upper bound delta `0.110236` with CI `[0.081365, 0.141732]`.
- Do not call the current score-packet path a positive method. Deployable WZ and full source-only fusion do not beat source-index+confidence; only the source+target-at-encoder upper bound helps. Mechanism shorthand: `2.10 -> 0.28` bits after receiver conditioning.
- L_Q1 is killed as a final query-conditioned two-way score-packet shot: delta `-0.022284` versus source-index+confidence, controls did not collapse, and ablations explain it.
- Allowed LatentWire claim: bounded negative / diagnostic failure localization for source-copy leakage, receiver-conditioned controls, and non-deployable oracle headroom.
- Next possible positive path is L-A2 only after a generated-solution candidate-pool cache with verifier/source/target scores exists, still dev/gate only before confirmation.
