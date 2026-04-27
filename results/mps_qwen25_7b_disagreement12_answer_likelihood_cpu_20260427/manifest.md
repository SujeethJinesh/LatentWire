# MPS Qwen2.5-7B Disagreement12 Answer-Likelihood CPU Smoke

- date: `2026-04-27`
- status: `fails_gate`
- rung: `smoke`
- source surface:
  `results/mps_qwen25_7b_disagreement12_discovery_20260427/`
- scorer: `Qwen/Qwen3-0.6B`
- device/dtype: `cpu` / `float32`
- rows: `12`
- candidate text field: `normalized_prediction`
- continuation template: `Answer: {text}`

## Artifacts

- candidate pools: `candidate_pools/`
- sketches: `sketches/`
- gate JSON: `gate.json`
- gate markdown: `gate.md`
- predictions: `predictions.jsonl`

## Conditions

- `matched`
- `target_only`
- `slots_only`
- `shuffled_source`
- `answer_only`
- `answer_masked_source`

## Readout

- matched source slot correct before scoring: `4/12`
- answer-only source slot correct before scoring: `4/12`
- answer-masked-source slot correct before scoring: `0/12`
- gate status: `condition_likelihood_receiver_fails_gate`
- live/CV clean source-necessary IDs: `0`
- holdout/frozen clean source-necessary IDs: `0`
- matched and answer-only sketch SHA256:
  `fbc34d474466922f3678f0615e2fab8a88e3f1ee90723279f1d3626267e891a7`
- matched top-label histogram: `source: 12`

## Decision

Prune normalized-answer receiver-likelihood variants on this surface. Matched
equals answer-only, so the recoverable signal is source final-answer relay, not
nontrivial cross-model communication.
