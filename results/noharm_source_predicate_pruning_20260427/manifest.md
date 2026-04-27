# No-Harm Source Predicate Pruning

- date: `2026-04-27`
- status: `cpu_source_predicate_family_pruned_until_stronger_surface_or_learned_predicates`
- live branch entering run: numeric syndrome / shallow source-predicate decoding
- scale-up rung: strict CPU smoke over existing SVAMP70 artifacts
- device: CPU-only artifact replay; no MPS jobs were started
- MPS blocker: PID `31103` remained present with `STAT=UE` after user attempted
  both `kill -9 31103` and `sudo kill -9 31103`

## Commands

### Candidate Syndrome Bits4 Audit

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir .debug/candidate_syndrome_bits4_audit \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --bits 4 \
  --run-date 2026-04-27
```

### Source Predicate Router No-Harm Pressure

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

### Source Likelihood No-Harm Gate

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

## Results

- Candidate syndrome bits4: fail. Live clean source-necessary `1`, target-self harms `16`; holdout clean source-necessary `4`, target-self harms `14`; control clean union `0`.
- Source predicate router with `accept_penalty=0.25`: fail. Best rows match `23/70` with `3` clean source-necessary IDs, control clean union `0`, but accepted harm remains `1` and accuracy fails the gate.
- Source likelihood no-harm gate: fail. Live matched `21/70`, holdout matched `8/70`, both with `0` clean source-necessary IDs, `0` accepted harm, and `0` clean control union.

## Decision

Prune shallow CPU source-predicate decoding on current artifacts. The family is
only worth reviving with learned semantic predicates, erasure-aware abstention,
or a stronger source surface. Do not rerun numeric/hash syndrome or source-text
feature thresholds on these artifacts.

## Artifact Hashes

- `candidate_syndrome_bits4_probe.json`: `f304e16922f49bff9d909d73761681b32ee104b04bfbf80a3702254960802fa8`
- `candidate_syndrome_bits4_probe.md`: `99b8aa9f997120a906cf96da45cf747a750e933915732d8b89cded73f7c84f34`
- `source_predicate_router_penalty025.json`: `4f2e04497fc4964d7637badcedf66644c26c004408f9b8916c060e79841e9e2b`
- `source_predicate_router_penalty025.md`: `13625b933afbba6d92995ee7050768377a249f47900fe28e322c892684e4c578`
- `source_predicate_router_penalty025_predictions.jsonl`: `ef1409b5e5af758a484b62e9282016522ee957675663eb93c6c5735e1a56c4e6`
- `source_likelihood_noharm_gate.json`: `97182e0394c8c61d01da0baa1ab1b53eb3427cd33f379230b10f57e41a2e52db`
- `source_likelihood_noharm_gate.md`: `cfdd39a0ada6228f78e5591effb9fb3c1fd4c17e94cab2f472bcc296c94cc6e2`
- `source_likelihood_noharm_gate_predictions.jsonl`: `e8d12c2f5cff8480eafdb2e9392f4145020aa3c9779904c8149e425bfdad8633`

## Next Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, the
next action is OS/session-level cleanup or reboot before MPS experiments; a
normal or sudo `SIGKILL` is insufficient while the process remains in
uninterruptible `UE` state.
