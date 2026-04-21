# 388. GSM8K 32-Example Smoke Contract

Date: 2026-04-21

This memo freezes the next benchmark smoke so the method-discovery track can
touch real rows without drifting into a mixed or unfair table.

## Exact smoke

- Task: `GSM8K`
- Slice size: `32`
- Slice source: first `32` rows of `data/gsm8k_eval_70.jsonl`
- Materialized path at runtime: `/tmp/gsm8k_eval_32.jsonl`
- Sender: `Qwen/Qwen2.5-0.5B-Instruct`
- Receiver: `Qwen/Qwen3-0.6B`

## Exact rows

1. `target_alone`
2. `text_to_text`
3. `rotalign_kv`
4. `c2c_generate`

## Why this smoke

- It is the smallest held-out slice where LatentWire and C2C can both run on
  the same local JSONL examples with the same offline scorer.
- It avoids pulling KVComm into the wrong task family.
- It avoids `lm-eval` and `OpenCompass` prompt drift on GSM8K.

## Metric contract

Primary metric:
- generation `accuracy` under the current LatentWire GSM matcher

Secondary metrics:
- paired `win/loss/tie` counts versus `target_alone`

Tertiary metrics:
- `generated_tokens_avg`
- `latency_sec`
- `examples_per_sec`

## Pass / fail checks

1. All four rows emit exactly `32` predictions.
2. The `example_id` sets are identical.
3. All rows are greedy-only:
   - `do_sample=false`
   - `temperature=0.0`
   - `max_new_tokens=64`
4. No row has empty predictions.
5. Numeric extraction coverage is at least `31 / 32`.
6. Rerunning `target_alone` on the same slice produces byte-identical
   predictions.
7. Offline rescoring of saved `target_alone` outputs matches the sidecar
   summary exactly.
8. `c2c_generate` should beat `target_alone` by at least `2` correct answers.
   If not, stop and debug prompt/scorer mismatch before widening.
9. `rotalign_kv` must at least tie `target_alone`. If it loses, stop benchmark
   expansion and return to method debugging.

## Why not other runners here

- `KVComm` is LongBench-oriented and should stay in the long-context suite.
- `LatentMAS` is same-backbone and belongs in an appendix/same-model table.
- `lm-eval` and `OpenCompass` are useful harnesses, but their local GSM8K
  prompt defaults are not the frozen zero-shot contract for this smoke.

## Immediate next use

- Keep this as a smoke-only checkpoint.
- Do not promote it to a paper table yet.
- Run it only after the current method branch is stable enough that a loss to
  `target_alone` would be informative rather than noise.
