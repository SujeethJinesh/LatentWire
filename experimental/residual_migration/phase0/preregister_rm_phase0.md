# Residual Migration in Hybrid Reasoners — Phase 0 Preregistration

**Frozen on**: 2026-05-07
**Frozen by**: human author (before any data observed)

## Hypothesis

"Rethinking the Outlier Distribution in Large Language Models" (May 2025)
showed that massive activations propagated through residual streams in
standard transformers do not affect accuracy when clipped. We test
whether this holds in current 2026 hybrid Mamba-Transformer reasoners.

## Falsifiable predictions

- **PASS (replicates)**: AIME-2025 accuracy drop <1.5% after clipping
  residual-stream tokens >95th percentile to the 95th percentile.
  Decision string: PASS_RM_PHASE0_RETHINKING_REPLICATES
- **PASS (rejects for hybrids)**: drop >3%. Hybrids depend on residual
  outliers more than transformers do. Decision string:
  PASS_RM_PHASE0_HYBRIDS_DEPEND_ON_RESIDUAL
- **KILL**: 1.5-3% drop (ambiguous, neither finding strong enough).
  Decision string: KILL_RM_PHASE0_AMBIGUOUS_DROP

## Model

- ibm-granite/granite-4.0-h-tiny

## Promptset

- Source: AIME-2025
- Count: 12 traces (same as OutlierMigrate; reuses shared dump pass for
  efficiency)
- Selection: deterministic, prompts 0-11
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

1. Baseline: run inference on all 12 prompts. Record AIME-2025 accuracy
   (correctness on each problem, scored by exact match against canonical
   answers).
2. Ablation: run inference with a forward hook that, at every transformer
   block, identifies residual-stream tokens with magnitude > 95th
   percentile (computed per layer per token position) and clips them to
   the 95th percentile.
3. Record ablation accuracy.
4. Compute: drop = baseline_accuracy - ablation_accuracy.
5. Apply decision rule above.

## Statistical readout

- Per-prompt accuracy is binary (0 or 1).
- Aggregate: mean accuracy across 12 prompts.
- Bootstrap (n=1000) over the 12 prompts for 95% CI on the drop.
- Pass-replicates: CI upper bound < 1.5%.
- Pass-rejects: CI lower bound > 3%.
- Kill: anywhere in between.

## Forbidden inputs

- Must not change the 95th percentile threshold after observing baseline.
- Must not select a different ablation target (e.g., 90th or 99th) after
  seeing 95th-percentile results.
- Must not exclude prompts post-hoc.

## On pass (replicates)

Append residual_migration_phase1 to queue.yml. Phase 1 validates at
scale on Granite-4.0-H-Small and adds layer-stratified ablation.

## On pass (rejects)

Append residual_migration_phase1_negative to queue.yml. Phase 1 confirms
hybrid-dependent finding at scale and isolates attention vs SSM
contribution to the effect.

## On kill

Write experimental/KILLED_residual_migration_ambiguous/README.md.
