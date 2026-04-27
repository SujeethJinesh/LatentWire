# Qwen3 Target-Likelihood Receiver Smoke

- date: `2026-04-27`
- status: `fails_live_prune`
- git commit at run start: `5e28beaff290a2ebf1498793ce31fa90afbc3143`
- device: `cpu`
- dtype: `float32`
- scorer model: `Qwen/Qwen3-0.6B`
- dataset: `results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl`
- dataset sha256: `bc9178d043bc05f2d1d1dd4aa2c6ec1ed024643b1a2886dfd243c4b1eca3e131`

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py --source-model Qwen/Qwen3-0.6B --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone --reference-label target --candidate-text-field normalized_prediction --continuation-template 'Answer: {text}' --resume --device cpu --dtype float32 --prompt-mode direct --source-use-chat-template --source-enable-thinking false --date 2026-04-27 --output-jsonl results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl --output-md results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.md
```

## Metrics

- rows: `70`
- target-alone correct: `21/70`
- source-alone correct: `13/70`
- text-to-text correct: `22/70`
- top-likelihood selection correct: `14/70`
- top labels: source `64`, text `6`, target `0`
- clean source-only IDs under accept-all source-top: `6`
- accepted target-correct harm under accept-all source-top: `16`
- best simple no-harm live thresholds: at most `1` clean source-only ID,
  about `22-23/70` correct, below the `25/70` live gate

## Decision

Fail/prune this target-likelihood receiver variant before holdout. The target
receiver likelihood does not separate useful source answers from harmful source
answers on live SVAMP70.

## Artifact Hashes

- `live_target_model_normpred_answer_template.jsonl`:
  `104ceba6676c752c2863347a2b201faa48f23f3964fee3cdcd22430b461e3ca0`
- `live_target_model_normpred_answer_template.md`:
  `10fd305e89940ddb0c86b3a855524d4e24261629cd5f9cfd8893d23209c94f75`
- ordered example IDs sha256:
  `0292230b41840995d6c178c72b571f4f4441e631a6e7f1535a03106717010506`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
```

Result: `9 passed in 0.09s`.
