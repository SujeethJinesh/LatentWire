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

## 2026-06-03 - Held-out one-way closeout and L_Q1 final shot

- One-way confirm closeout: on `381` held-out confirm rows, deployable WZ remains below source-index+confidence (`-0.023622`, CI `[-0.060367, 0.013123]`) and full source-only fusion remains null (`-0.031496`, CI `[-0.062992, 0.002625]`); source+target-at-encoder remains a non-deployable upper bound (`0.110236`, CI `[0.081365, 0.141732]`).
- L_Q1 two-way query packet: killed on `359` gate rows. It loses to source-index+confidence by `-0.022284` with CI `[-0.055710, 0.011142]`, controls do not collapse, and query-only/reply-only ablations explain it.
- Hypothesis update: one-way and query-conditioned score-packet LatentWire are locked negative on MMLU-Pro. Do not chase additional score-packet variants; only L-A2 generated-solution caches remain as a parked LatentWire positive shot.

## 2026-06-03 - Overnight CPU-only screening closeout

- CacheWire/C2C trace oracle: current cached trace features are receiver/control-limited. Matched generation-summary trace accuracy is `13/32`, target-only and zero-source are `14/32`, label-shuffle is `15/32`, and the source-necessary clean count is `0`.
- C2C tractability: local C2C repo and Qwen2.5-0.5B to Qwen3-0.6B checkpoint metadata are present; SVAMP MPS replay reaches `0.5000` versus target `0.2500`, but this clone has no completed MMLU-Redux reproduction artifact.
- Existing KVComm smoke is killed for method evidence: matched predictions collapse to zero-source predictions in every inspected ARC/OpenBookQA diagnostic.
- L-A2 remains parked: the cache has only a 3-row generated-text smoke and lacks source, target, and verifier score surfaces, so the high-entropy ceiling cannot be estimated.
- Channel-Set update: C_A1 is the first GPU backfill target, but C_F is control-contaminated (`M10` random-control delta `-0.761487`) and cannot move to foreground.
- Next branch order: C_A1 native replay with DeepSeek/Falcon sentinels, then L-A2 generated-solution cache materialization, then C-F identical-row denominator cleanup.

## 2026-06-04 - Overnight v2 adjudication correction

- Corrected EXP1 state: the `n=32` CacheWire/C2C trace readout is `INCONCLUSIVE_UNDERPOWERED`, not a deployable-path kill. The dense C2C teacher hint (`16/32` vs target `8/32`) keeps CacheWire ceiling/headroom alive until a powered source/receiver hidden-feature probe replaces it.
- Deterministic kills retained: KVComm matched equals zero-source in inspected smokes, generated-answer value/index packets are answer-text leakage, teacher-delta ties zero/target controls, candidate-delta is dominated by coefficient controls, and C_F is killed by random/control contamination.
- C_F branch state updated to `KILLED_CONTROL_CONTAMINATED`; do not queue C_F again without a fresh preregistration and identical-row denominator that beats random/static controls.
- Required next evidence: `dashboard/overnight_v2.md` must report wall-clock seconds, achieved dev/gate n, and MDE for EXP1 and EXP4. A fast report without generated/extracted rows is not completion.
