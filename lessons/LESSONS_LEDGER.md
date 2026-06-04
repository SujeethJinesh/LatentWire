# Lessons Ledger

## 2026-06-03 - Stage-1 cached WZ/source-copy family

- Hypothesis update: cached `L_SCORECOMP_wz_bins_deployable` packet artifacts are killed on the row-safe dev/gate screen; 12 gate rows have negative CI-high versus equal-byte baselines and 0 rows promote.
- L-B1 cache update: fixed-packet matched rows behave as source copy on the admitted Stage-1 cache. Gate matched-equals-source is `0.9969`, damage on target-correct/source-wrong rows is `192/192`, and repair is source-driven (`380/382` on target-wrong/source-correct rows).
- Ruled out: treating old 4-way source-copy packet wins as a deployable positive method or GPU foreground trigger.
- Still alive: fresh high-entropy LatentWire WZ/L-B1 on Mac with explicit risk/confidence/leakage controls; Channel-Set C_A1/C_F only as GPU backfill candidates until offline controls are stronger.

## 2026-06-03 - Fresh MMLU-Pro Mac continuation

- Fresh split guard: MMLU-Pro rows were split before scoring; confirm rows scored `0`.
- Fresh I_beyond: `2.231310` bits on scored dev/gate rows.
- WZ result: status `MAC_DONE`, gate delta vs best baseline `-0.043478` with CI-low `-0.304348`.
- L-B1 result: AURC delta vs source-index+confidence `-0.222541`, paired delta vs source-index `-0.086957`.
- Decision: `BOUNDED_NEGATIVE` for this Mac continuation; no foreground GPU unless a future row earns `PROVISIONAL_PROMOTE_TO_GPU`.

## 2026-06-03 - Powered LatentWire oracle ladder

- Power update: scaled fresh MMLU-Pro CPU scoring to `1500` dev/gate rows with `359` gate rows; achieved delta_beyond_score MDE half-width `0.030641`, below the `0.05` target.
- Receiver-conditioned diagnostic: `I(source_scores; correct | source_top1)=2.098527` bits and `I(source_scores; correct | source_top1, target_scores)=0.281933` bits.
- Oracle ladder: deployable WZ and full source-score fusion both fail to beat source-index+confidence on gate; only the source+target-at-encoder upper bound is positive.
- Hypothesis update: current LatentWire score-packet path is `not-deployable` on this screening surface, not merely underpowered. Future LatentWire work should require L-A2 generated-solution caches or a genuinely source-only codec before any confirm/GPU spend.
