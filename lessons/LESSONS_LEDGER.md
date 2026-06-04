# Lessons Ledger

## 2026-06-03 - Stage-1 cached WZ/source-copy family

- Hypothesis update: cached `L_SCORECOMP_wz_bins_deployable` packet artifacts are killed on the row-safe dev/gate screen; 12 gate rows have negative CI-high versus equal-byte baselines and 0 rows promote.
- L-B1 cache update: fixed-packet matched rows behave as source copy on the admitted Stage-1 cache. Gate matched-equals-source is `0.9969`, damage on target-correct/source-wrong rows is `192/192`, and repair is source-driven (`380/382` on target-wrong/source-correct rows).
- Ruled out: treating old 4-way source-copy packet wins as a deployable positive method or GPU foreground trigger.
- Still alive: fresh high-entropy LatentWire WZ/L-B1 on Mac with explicit risk/confidence/leakage controls; Channel-Set C_A1/C_F only as GPU backfill candidates until offline controls are stronger.
