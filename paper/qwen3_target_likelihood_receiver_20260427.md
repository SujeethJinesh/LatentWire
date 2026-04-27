# Qwen3 Target-Likelihood Receiver Smoke

- date: `2026-04-27`
- status: `fails_live_prune`
- scale-up rung: smoke / branch discovery
- git commit at run start: `5e28beaff290a2ebf1498793ce31fa90afbc3143`

## Question

Can the target model act as a no-harm receiver by scoring target/text/source
candidate answers and accepting source candidates only when its own likelihood
indicates the source answer is useful?

This is not sufficient as cross-LLM communication by itself. It is only a
candidate receiver gate if source-conditioned candidates produce clean wins that
target-only/text/self-selection controls cannot recover at the same budget.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen3-0.6B \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field normalized_prediction \
  --continuation-template 'Answer: {text}' \
  --resume \
  --device cpu \
  --dtype float32 \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --date 2026-04-27 \
  --output-jsonl results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl \
  --output-md results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.md
```

MPS was not used because PID `31103` remained in `STAT=UE`.

## Live Result

- rows: `70`
- target-alone correct: `21/70`
- source-alone correct: `13/70`
- text-to-text correct: `22/70`
- top-likelihood selection correct: `14/70`
- top label counts: source `64`, text `6`, target `0`
- clean live source-only IDs recovered by accepting all source-top rows:
  `14bfbfc94f2c2e7b`, `2de1549556000830`,
  `4d780f825bb8541c`, `41cce6c6e6bb0058`,
  `ce08a3a269bf0151`, `bd9d8da923981d69`
- accepted harm if accepting all source-top rows: `16`

Simple live-only no-harm threshold probes over source-top likelihood features
did not clear the smoke gate. Best no-harm settings recovered at most `1` clean
source-only ID and reached only about `22-23/70` correct, below the required
`25/70` live-correct threshold. Lower-margin thresholds could recover `2` clean
IDs but introduced target-correct harm.

## Decision

Prune this target-likelihood receiver variant on the current SVAMP70 live
surface. The target scorer strongly prefers the source label, but does not
separate correct source help from wrong source answers well enough to preserve
target-correct cases. Because the live no-harm gate fails before true
condition-specific controls, holdout collection would not change the promotion
decision.

The broader receiver-gate idea remains alive only if a stronger source surface
or a real condition-specific control harness is available. A fair future harness
must rescore mutated candidate pools for `zero_source`, `shuffled_source`,
`target_only`, and `slots_only`; the existing likelihood sketch analyzer only
shuffles sketches and forces some controls to fallback, so it is not sufficient
for a target-likelihood communication claim.

## Artifacts

- `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl`
  - sha256: `104ceba6676c752c2863347a2b201faa48f23f3964fee3cdcd22430b461e3ca0`
- `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.md`
  - sha256: `10fd305e89940ddb0c86b3a855524d4e24261629cd5f9cfd8893d23209c94f75`
- ordered example IDs sha256:
  `0292230b41840995d6c178c72b571f4f4441e631a6e7f1535a03106717010506`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
```

Result: `9 passed in 0.09s`.

## Next Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, resume MPS source-surface/interface reset work. If it
remains, continue CPU-only with a canonical exact-ID overlap audit across
SVAMP70 live/holdout surfaces rather than another threshold sweep.
