# No-Harm Source Predicate Pruning

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; the project still
   needs one positive method that survives live/holdout controls, seed repeats,
   C2C/text/self-repair baselines, and cross-family falsification.
2. Current paper story: source-derived side information has isolated clues, but
   shallow numeric and source-text predicates are not yet a deployable
   communication method.
3. Exact blocker to submission: no branch recovers source-necessary wins while
   preserving target-correct examples; MPS remains blocked by orphaned PID
   `31103`, which remained in `STAT=UE` after both `kill -9` and
   `sudo kill -9`.
4. Current live branch: none. Top candidates after this cycle are learned
   semantic predicate decoding and zero-init target-preserving query
   bottlenecks, contingent on a stronger source surface.
5. Highest-priority gate: no-harm CPU replay over existing source-predicate and
   syndrome artifacts.
6. Scale-up rung: strict CPU smoke / branch pruning.

## Committee Synthesis

The literature and artifact audits agree on the failure mode: the source signal
is not absent, but the current decoders apply it too aggressively. The next
method must behave like an erasure-aware side-information decoder:

- source sends compact semantic predicates or an innovation, not an answer;
- target keeps ownership of final selection;
- decode abstains unless the source signal uniquely helps and controls do not;
- any learned connector starts at target-alone behavior through zero-init
  target-side gates.

Primary-source inspirations recorded in
`references/467_crossfield_noharm_predicate_refs.md` include Neural
Distributed Source Coding, DeepJSCC-WZ, side-information vending machines,
Semantic Entropy Probes, Q-Former connectors, C2C, and multi-way
representation alignment.

## CPU Gates

### Candidate Syndrome Bits4

Command:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir .debug/candidate_syndrome_bits4_audit \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --bits 4 \
  --run-date 2026-04-27
```

Result: fail. Live recovered `1` clean source-necessary ID but harmed `16`
target-self examples. Holdout recovered `4` clean source-necessary IDs but
harmed `14`. Control clean union stayed `0`, so there is some source signal,
but it is not target-safe.

### Source Predicate Router With No-Harm Pressure

Command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_sidecar_cv_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --accept-penalty 0.25 \
  --min-correct 25 \
  --min-clean-source-necessary 3 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 63 \
  --date 2026-04-27 \
  --output-json .debug/source_predicate_gates/source_predicate_router_penalty025.json \
  --output-md .debug/source_predicate_gates/source_predicate_router_penalty025.md \
  --output-predictions-jsonl .debug/source_predicate_gates/source_predicate_router_penalty025_predictions.jsonl
```

Result: fail. Best rows matched `23/70` with `3` clean source-necessary IDs
and control clean union `0`, but accepted harm stayed at `1` and the row failed
the matched-correct gate.

### Source Likelihood No-Harm Gate

Command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp70_source_likelihood_sketch_gate.py \
  --live-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_normpred_answer_template_sketch_cpu.jsonl \
  --live-candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --live-candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --live-candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_normpred_answer_template_sketch_cpu.jsonl \
  --holdout-candidate target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --holdout-candidate text=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --holdout-candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --accept-penalty 0.25 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/source_predicate_gates/source_likelihood_noharm_gate.json \
  --output-md .debug/source_predicate_gates/source_likelihood_noharm_gate.md \
  --output-predictions-jsonl .debug/source_predicate_gates/source_likelihood_noharm_gate_predictions.jsonl
```

Result: fail. The no-harm constraint worked but removed the useful signal:
live matched `21/70`, holdout matched `8/70`, clean source-necessary `0` on
both, accepted harm `0`, and control clean union `0`.

## Decision

Prune shallow CPU source-predicate decoding on current artifacts. Numeric/hash
syndromes and source-text threshold routers should not be rerun on these
surfaces. The branch is only worth reviving with:

- learned semantic predicates over target candidate pools;
- erasure-aware abstention;
- source-fault detection before decode;
- a stronger source surface that has enough clean headroom.

## Artifact Manifest

Durable artifacts were copied to
`results/noharm_source_predicate_pruning_20260427/`. The result directory is
ignored by default, so the tracked `manifest.md` records command provenance and
SHA256 hashes for the JSON/MD/prediction files.

## Next Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, the
hard blocker is OS/session-level cleanup or reboot before any MPS experiment;
normal and sudo `SIGKILL` attempts have not cleared it.
