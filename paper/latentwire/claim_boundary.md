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
- Paper drafting should cite `dashboard/latentwire_terminal_negative_sanitized.md` for aggregate one-way/L_Q1 negatives; do not expose or reuse raw held-out rows in future screening contexts.
- L_PC1 is a pending fresh-pair lead, not a scoped positive. The cached strict slice has `512` rows and a small selected-packet gain `+0.042969`, CI `[+0.015625, +0.068359]`, but it is one cached HellaSwag Qwen-to-Phi surface and lacks the requested second pair/task plus source-index and equal-byte text controls. See `dashboard/l_pc1_reconciliation.md`.
- L_PC2 is not a latent/source-private packet win: the private tool result is valuable, but the equal-byte visible tool control ties it.
- L_PC5 is oracle-only. The `+0.666` strict rerank gain uses candidate scores computed from `cand == answer`, so it is a candidate-pool ceiling until a gold-blind verifier cache clears the nondegenerate prompt floor and beats source-index/equal-byte text controls. See `dashboard/l_pc5_deployable_verifier_plan.md`.
- L_C2 is oracle-only. The `+0.832` strict fuser gain uses the SVAMP equation-derived `tool_answer` as the fuser prediction, so it is a gap-to-answer ceiling, not a deployable cache-fusion method. See `dashboard/l_c2_oracle_decomposition.md`.
- Next possible positive paths are: fresh L_PC1 cross-family pair/task matrix, powered CacheWire ceiling from real source/receiver hidden-cache features, and L-A2 only after a generated-solution candidate-pool cache with gold-blind verifier/source/target scores exists. All remain dev/gate only before confirmation.
- 2026-06-05 cheap-exhaustion pass did not authorize more MPS live-forward evidence. Any next LatentWire positive attempt should be a reviewed CPU/GGUF cache materialization or scorer with wrong-row, zero-source, source-index, and equal-byte text/tool controls pre-registered.
