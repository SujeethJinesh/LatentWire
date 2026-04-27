# Byte-Efficient Side-Information Branch Audit

Date: 2026-04-27 01:08 PDT

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; still missing a
   deployable positive method with live/holdout controls, seed stability,
   paired uncertainty, systems accounting, and cross-family falsification.
2. Current paper story: source-derived side information is still the strongest
   story, but historical sparse-K, RotAlign/DynAlign, sidecar, Perceiver, and
   semantic-predicate positives are mechanism clues rather than headline
   claims.
3. Exact blocker to submission: no method currently beats target-alone/text
   relay while surviving source-destroying controls and target-self
   preservation; MPS remains blocked by orphaned PID `31103`.
4. Current live branch or top candidates: no live branch. Top candidate is a
   byte-efficient learned syndrome/innovation sidecar decoded with target-side
   context. Second candidate is a target-safe sparse/dictionary atom sidecar.
5. Highest-priority gate: preserve the audit as durable state and define the
   next exact gate once MPS clears.
6. Scale-up rung: post-kill branch selection / next smoke.

## Blocker Recheck

Command:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: PID `31103` is still present with `PPID=1` and `STAT=UE`, running the
old `scripts/calibrate.py ... --device mps --dtype float32 ...` job. User-side
`kill -9` and `sudo kill -9` have not removed it. Do not start MPS jobs until
this process disappears or the OS/session is reset.

## Historical Positives Re-read

### Sparse K-only / cosine transport

Relevant artifacts:

- `results/gsm8k_query_attention_20260417/summary.md`
- `results/cosine_g010_controls_20260417/gsm8k/summary.md`
- `results/gsm8k_k_only_followup_20260416/summary.md`
- `latent_bridge/current_readout_20260418.md`

Evidence:

- Query-aware sparse K-only recovered a narrow GSM70 lift at ratio `0.50`
  (`0.057143` vs target `0.042857`) while random/zero selector controls were at
  target.
- Cosine `g=0.10` showed a larger historical lift (`0.0857` vs zero/random
  controls `0.0143`) but still trailed text-to-text and did not become stable.
- Later seed sweeps weakened the family; fixed-prior sparse K-only averaged
  below target across seeds.

Decision: keep as a systems/control clue for source-state sparsification, not a
live method branch.

### RotAlign / DynAlign

Relevant artifacts:

- `results/gsm8k_contract_residual_rank16_dynalign_20260421/dynalign_module_replace_residrank16_diagnostics_20260423.md`
- `paper/gsm8k70_seed_stability_full_20260422.md`
- `paper/gsm8k70_seed4_dynalign_source_controls_20260426.md`

Evidence:

- GSM8K32 reached `4/32` vs target `2/32`, and GSM70 seed 0 reached `8/70` vs
  target `4/70`.
- Seed stability failed: seeds 1/2 had nonfinite failures, seed 3 fell to
  `2/70`, and seed 4 was only `4/70`.

Decision: kill raw transport as a live branch. Revive only as target-safe
conditional innovation or sparse/dictionary side information with source-zero,
shuffle, and predictor-only controls.

### Source sidecar / candidate-pool syndrome

Relevant artifacts:

- `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/manifest.md`
- `results/qwen25math_qwen3_svamp70_source_surface_20260426/sidecar_manifest.md`
- `results/svamp32_syndrome_sidecar_probe_20260424/manifest.md`
- `paper/svamp32_syndrome_sidecar_probe_20260424.md`

Evidence:

- SVAMP32 source-contrastive sidecar reached `11/32` vs target/text `8/32`
  with clean source-necessary wins and no clean control union.
- SVAMP70 live textless sidecar reached `26/70` vs target `21/70`, with
  bootstrap lower bound at `0.0000`, but holdout/fixed guards failed.
- Candidate-pool syndrome probes showed a low-byte bound (`1` byte residue,
  `14-15/32`) but depended on C2C-derived residues, so they are not a source
  method yet.

Decision: promote the idea, not the old implementation. The next branch should
learn a source-derived syndrome/innovation sidecar and decode it with target
candidate/cache side information.

### Perceiver / query-memory / shallow predicates

Relevant artifacts:

- `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/manifest.md`
- `results/svamp70_perceiver_answer_teacher_contrastive_20260426/manifest.md`
- `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/manifest.md`
- `paper/noharm_source_predicate_pruning_20260427.md`

Evidence:

- Perceiver answer-teacher controls were stronger than matched on SVAMP32
  clean6 and leaked on SVAMP70.
- Semantic predicates passed the live surface (`25/70`, `3` clean, `0` harm)
  but failed holdout (`9/70`, `0` clean).
- No-harm source likelihood gates recovered `0` clean IDs.

Decision: kill shallow source-text/predicate/router variants on current
surfaces.

## Reference Update

New focused memo:

- `references/471_byte_efficient_source_sideinfo_refs.md`

Sources added to the active design contract:

- Q-KVComm, KVComm, C2C, DroidSpeak, latent K-V cache alignment.
- Slepian-Wolf/Wyner-Ziv and neural distributed source coding.
- Semantic communication between agents.
- Universal sparse autoencoders and cross-architecture crosscoders.

Design consequence: the method should be evaluated as decoder-side-information
coding. Quality must be plotted or at least tabulated against bytes, with
KVComm/Q-KVComm/C2C/text/self-repair baselines and strict source-destroying
controls.

## Branch Decision

Promoted:

- Learned source-derived syndrome/innovation sidecar over target-side candidate
  pools or cache context.

Weakened:

- Raw KVComm as a method; keep it as a baseline because the current smoke is
  byte-heavy (`530432` communicated bytes/example at a 0.25 layer fraction).
- Raw RotAlign/DynAlign; keep only target-safe innovation/dictionary variants.

Killed on current evidence:

- Shallow source likelihood, semantic predicate, and fixed source-text router
  branches.
- Perceiver/query-memory answer-teacher variants with current controls.
- Numeric hash-syndrome probe as deployed in
  `results/candidate_syndrome_decoder_20260427`.

## Next Exact Gate

First check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source MPS surface scout already
specified in `paper/postkill_historical_cpu_audit_20260427.md`:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Only if that scout has ordered ID parity, high numeric coverage, source-only
over target at least `6/70`, and target-or-source oracle at least target plus
`6/70`, implement the next learned syndrome/innovation sidecar gate with:

- matched sidecar
- zero-source sidecar
- shuffled-source sidecar
- random same-byte sidecar
- target-only and slots-only
- same-byte KVComm/Q-KVComm-style baseline where feasible
- paired sidecars against target-only and every source-destroying control

If PID `31103` remains present, the hard blocker persists. The repo is in a
resumable state, and no CPU-only artifact command remains that can promote a
positive method on the current frozen surfaces.
