# Historical Positive Branch Audit

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; the project still lacks
   a positive method with live/holdout source-control evidence, uncertainty,
   seed stability, and systems accounting.
2. Current paper story: historical RotAlign/latent_bridge/results artifacts
   contain real mechanism clues, but every old positive-looking method is
   either seed-unstable, control-explained, weaker than C2C without a systems
   win, or only an oracle/bound.
3. Exact blocker to submission: no source-derived communication method has yet
   cleared strict controls on a disjoint validation surface.
4. Current live branch: `source_likelihood_sketch` on SVAMP70 live/holdout,
   with source-likelihood side information over target/text/source candidates.
5. Highest-priority gate: clear the stuck MPS process and run the live/holdout
   sketch gate recorded in `paper/svamp70_source_likelihood_sketch_20260427.md`.
6. Scale-up rung: strict-small implementation; scientific run blocked by MPS.

## Audit Scope

The audit re-read the positive-looking branches the user flagged:

- `rotalign` and early RotAlign progress/readout memos
- `latent_bridge/HANDOFF.md` and blocker maps
- `paper/*.md` SVAMP/GSM memos
- `results/*/manifest.md` and positive-looking result readouts

This memo focuses on branches that ever looked promotable or provided a useful
mechanism clue.

## Branch Ranking

### 1. Source-Contrastive Sidecar, SVAMP32/SVAMP70

Status: weakened but still the best historical source-derived positive clue.

Key files:

- `paper/qwen25math_svamp32_source_contrastive_sidecar_20260426.md`
- `paper/qwen25math_svamp70_source_contrastive_sidecar_20260426.md`
- `results/qwen25math_qwen3_svamp70_source_surface_20260426/`
- `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/`

Evidence:

- SVAMP32 strict-small cleared:
  - matched `11/32`
  - target/text `8/32`
  - source `6/32`
  - clean source-necessary `3/4`
  - control clean union `0/4`
  - one-byte source residue sidecar
- SVAMP70 medium live was positive versus target/text:
  - target `21/70`
  - text `22/70`
  - C2C `31/70`
  - guarded sidecar `25/70`
  - textless sidecar `26/70`
  - clean source-necessary `4/6`
  - control clean union `0/6`
  - one-byte sidecar
- Disjoint holdout weakened the fixed decoded guard:
  - len-ratio/textless guard matched `10/70`
  - clean source-necessary `0/2`
  - control clean union `2/2`
  - text relay `18/70`, C2C `37/70`

Decision:

- Do not revive fixed decoded guards.
- Keep the side-information formulation: target has candidate context, source
  should provide a low-rate disambiguating signal.
- This directly supports the current `source_likelihood_sketch` branch, which
  replaces brittle decoded source-quality guards with source-model candidate
  likelihoods.

Next decisive gate:

- Run the source likelihood sketch live/holdout gate. Promote only if it keeps
  the live clean wins while removing holdout control leakage.

### 2. GSM70 DynAlign Residual Rank-16

Status: useful mechanism clue, killed as current live method.

Key files:

- `results/rotalign_progress_report_20260414.md`
- `results/rotalign_reaudit_20260414.md`
- `paper/gsm8k_contract_artifact_manifest_20260422.md`
- `paper/gsm8k70_source_controls_20260423.md`
- `paper/gsm8k70_seed4_dynalign_source_controls_20260426.md`

Evidence:

- Early RotAlign control suite found a fragile positive regime:
  sparse/low-gate/quantized row `0.06` vs target `0.04` on GSM8K100, but it was
  selected on the reported slice.
- GSM70 seed0 dynalign residual rank-16:
  - target `4/70`
  - live `8/70`
  - paired vs target `6W/2L/62T`
  - zero/shuffled controls erased `0/6` live wins under target-safe fallback
- Larger frozen same-pair slice:
  - candidate mean `0.1143`
  - target `0.0571`
  - C2C `0.1286`
  - bootstrap delta crossed zero
- Seed stability failed:
  - seed3 `2/70`
  - seed4 `4/70`
  - seed4 checkpoint finite, so the failure is not only nonfinite construction

Decision:

- Do not revive raw dynalign scale-up.
- Keep it as a mechanistic clue: sparse, low-gate, compressed, source-specific
  transfer can matter, but raw static transport is seed-unstable.
- Any revival must subtract target-predictable signal or add a target-safe
  conditional innovation objective before generation.

Next decisive gate if revived later:

- decoder-conditioned innovation codec with matched, predictor-only ghost,
  shuffled-message, zero-source, target-only, and same-rate dense-resid
  controls. Do not run more raw dynalign seed repeats first.

### 3. Syndrome Sidecar Bound

Status: positive feasibility bound, not a deployable method.

Key files:

- `paper/svamp32_syndrome_sidecar_probe_20260424.md`
- `results/svamp32_syndrome_sidecar_probe_20260424/manifest.md`

Evidence:

- Strict target-side candidate pool:
  - matched `14/32`
  - target-only fallback `14/32`
  - target-self matched `3/3`
  - clean source-necessary `2/6`
  - control clean union `0/6`
  - one-byte residue syndrome
- Augmented pool:
  - matched `15/32`
  - clean source-necessary `3/6`
  - control clean union `0/6`
- Caveat: used C2C-derived numeric residues, so it is an oracle/bound rather
  than source-derived communication.

Decision:

- Do not report as a method.
- Treat as strong support for conditional sidecar/candidate-pool decoding.
- The current source likelihood sketch is the closest source-derived follow-up:
  source scores the candidate pool instead of borrowing C2C numeric residue.

Next decisive gate:

- If source-likelihood sketch fails, train a source-latent syndrome predictor
  with the same candidate-pool controls rather than returning to dense KV
  transport.

### 4. ID-Weighted Query-Innovation

Status: best latent_bridge source-specific clue, not a promotable row.

Key files:

- `paper/svamp32_idweighted_query_innovation_20260423.md`
- `results/svamp32_idweighted_query_innovation_20260423/manifest.md`
- `results/svamp32_source_oracle_bound_20260424/source_oracle_bound.md`

Evidence:

- best controlled `gate015` row:
  - matched `10/32`
  - teacher-only `2`
  - clean residual/source-necessary `1/6`
- fine gate `0.175` reached `11/32`, but still only `1/6` clean residual.
- The key positive clue is that clean ID `aee922049c757331` was not retained
  by the translated-KV-zero control.
- The row still fails the promotion gate:
  - below target self-repair `14/32`
  - below the `>=2/6` clean source-necessary threshold
  - sidecar oracle still fails `min_correct` and `min_clean_source_necessary`

Decision:

- Do not revive `idweighted_query_innovation` as-is.
- Keep it as the best old latent_bridge clue that matched source can provide
  one clean ID not explained by a source-destroying control.
- Any revival should be a materially different target-self-preserving
  innovation bottleneck, not another gate/rank/seed sweep on the same row.

Next decisive gate if revived:

- preserve target self-repair `14/32`
- recover `>=2/6` clean residual IDs
- allow `<=1` target-correct loss
- require zero-source, shuffled-source, target-only, and slots-only controls
  to recover none of the same clean IDs

### 5. Query-Innovation / Query-Memory / Perceiver Resamplers

Status: killed as current family; keep only as architecture inspiration.

Key files:

- `paper/query_resampler_answer_likelihood_gate_20260426.md`
- `paper/query_memory_answer_likelihood_cpu_sweeps_20260426.md`
- `paper/qwen25math_svamp32_perceiver_c2c_residual_20260426.md`
- `paper/svamp32_perceiver_answer_teacher_contrastive_20260426.md`
- `paper/svamp70_perceiver_answer_teacher_contrastive_20260426.md`

Evidence:

- Non-target-conditioned query-innovation checkpoint failed because
  `target_only` controls were not supported and answer-likelihood controls did
  not prove matched-source dependence.
- SVAMP32 delta-memory CPU smoke:
  - matched mean `-7.673776`
  - best-control delta `-0.141012`
  - best-control wins `0/4`
- SVAMP70 Perceiver answer-teacher:
  - matched mean `-7.261671`
  - best-control delta `-0.112360`
  - best-control wins `0/4`
- Qwen2.5-Math Perceiver had one 4-ID pass:
  - best-control wins `3/4`
  - live-best delta `+0.080362`
- The required clean6 expansion failed:
  - matched mean `-8.195434`
  - live-best delta `-0.090384`
  - shuffled/target/slots controls matched or beat it

Decision:

- Do not tune fixed gate, query count, rank, anti-memory weight, or another
  target-memory checkpoint on these artifacts.
- Keep Perceiver/query slots only as an architecture shape for a future
  target-conditioned innovation codec with real target-only/slots/shuffle
  controls.

Next decisive gate:

- Only revive after defining an innovation target and a predictor-only ghost
  control; otherwise it repeats the same target/slot leakage pattern.

### 6. Process Repair

Status: target-side baseline/confound only.

Key files:

- `paper/svamp70_process_repair_source_controls_20260426.md`
- `paper/svamp32_process_repair_source_controls_20260426.md`
- `results/process_repair_holdout_20260421/`

Evidence:

- Old SVAMP70 process-repair row looked strong:
  - matched process repair `38/70`
  - target self-repair `35/70`
  - matched had `3` wins over target self-repair
- Source controls erased source specificity:
  - zero-source K/V control `35/70`, overlapping `1/3` matched-only IDs
  - shuffled-source prompt control `37/70`, overlapping `3/3`
  - strict source-specific matched-only IDs after controls: `0`

Decision:

- Do not call process repair communication.
- Keep it visible as a strong target-side repair baseline and as a confound
  for any candidate-selection method.

Next decisive gate:

- If used again, repair can only select/refine after a source-derived signal
  has already cleared controls.

### 7. Source-Trace And Source-Confidence Routers

Status: useful instrumentation, not promotable.

Key files:

- `paper/source_generation_diagnostics_artifact_20260426.md`
- `paper/qwen25math_svamp70_source_trace_router_20260426.md`
- `paper/qwen25math_svamp70_source_confidence_router_20260426.md`

Evidence:

- Source diagnostics collector works and reproduces source-alone correctness.
- Source-trace router:
  - live CV matched `20/70`
  - clean source-necessary `1`
  - accepted harm `2`
  - holdout clean win survived equation permutation, weakening trace causality
- Source-confidence router:
  - live CV matched `24/70`
  - clean source-necessary `2`
  - control clean union `0`
  - holdout matched `7/70`
  - holdout clean source-necessary `0`

Decision:

- Keep diagnostics as artifact support.
- Do not tune shallow confidence/trace routers on this surface.
- A source-likelihood candidate sketch is still higher value because it asks
  the source model to compare candidate answers directly, not infer correctness
  from generic confidence.

## Overall Decision

The audit does not change the live branch. The best old positives all point
toward the same formulation:

- target-side candidate/context is useful decoder side information
- source should send a small, example-specific innovation or syndrome
- controls must destroy source identity without preserving clean wins
- raw dense/static KV alignment and decoded guard tuning are saturated

Therefore the current branch remains:

`source_likelihood_sketch` on `svamp70_live_source` with frozen validation on
`svamp70_holdout_source`.

The next exact scientific command is still blocked by PID `31103`; after it is
cleared, run the collector/analyzer commands in
`paper/svamp70_source_likelihood_sketch_20260427.md`.

## Pruning Rules From The Audit

- Do not revive raw dynalign residual rank-16 without a new conditional
  innovation objective and predictor/shuffle controls.
- Do not revive fixed decoded source-quality or length guards.
- Do not revive query-memory/Perceiver target-memory checkpoints by tuning
  rank, gate, slot count, or answer-teacher weights.
- Do not revive `idweighted_query_innovation` as-is; only reuse its single
  clean source-specific ID as a design clue for a target-self-preserving
  innovation bottleneck.
- Do not treat process repair as communication unless a source-derived signal
  has already cleared controls.
- Do not spend large compute on adjacent SVAMP scouts that fail clean
  source-only surface gates.

## Resume Check

Before any MPS run:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Proceed only if PID `31103` is absent or no longer a stuck
`scripts/calibrate.py --device mps` process.
