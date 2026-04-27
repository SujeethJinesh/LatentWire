# Source Likelihood Sketch Kill Note

Date: 2026-04-27

## Cycle Header

1. Current ICLR readiness and distance: not ICLR-ready; no positive method has
   cleared live/holdout source controls, seed stability, and cross-family
   falsification.
2. Current paper story: historical source sidecars show real headroom, but
   source likelihood over candidate answers does not produce a reliable
   deployable source signal on the frozen live surface.
3. Exact blocker to submission: no live branch now clears the strict-small
   gate; MPS remains blocked by PID `31103`.
4. Current live branch: `source_likelihood_sketch` is killed on this
   Qwen2.5-Math -> Qwen3 SVAMP70 live/holdout surface.
5. Highest-priority gate: move to source-surface discovery or a new
   source-controlled syndrome predictor rather than tuning likelihood sketch
   variants.
6. Scale-up rung: strict-small kill decision; no scale-up.

## Decision

Kill `source_likelihood_sketch` as the current live branch.

Reason:

- Four adjacent variants fail the live gate:
  - bare normalized answer, mean logprob
  - bare normalized answer, sum logprob
  - formatted `Answer: {normalized_prediction}`, mean logprob
  - formatted `Answer: {normalized_prediction}`, sum logprob
- All live variants stay at or below the target baseline and recover zero clean
  source-necessary IDs under the predefined live-CV rule.
- The only interesting partial result is formatted mean logprob on holdout:
  holdout `10/70`, clean source-necessary `2`, control union `0`, accepted harm
  `1`. It cannot be promoted because the live-CV training surface fails:
  live `20/70`, clean source-necessary `0`, control union `1`, accepted harm
  `1`.

Interpretation:

- The full-text smoke was style-confounded: the two-example full-continuation
  smoke top-ranked `text` twice, including a wrong text candidate.
- Answer-only scoring removes the obvious style confound, but the source model
  likelihood does not align with source-correct candidate selection on live.
- The holdout-positive formatted row is a scout signal, not a valid claim,
  because it appears only after the live training rule fails.

## Results

| Variant | Live matched | Live clean source-necessary | Live control union | Holdout matched | Holdout clean source-necessary | Holdout control union | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| normalized answer mean logprob | 21/70 | 0 | 0 | 8/70 | 0 | 0 | fail |
| normalized answer sum logprob | 20/70 | 0 | 0 | 8/70 | 0 | 0 | fail |
| `Answer: {text}` mean logprob | 20/70 | 0 | 1 | 10/70 | 2 | 0 | fail live |
| `Answer: {text}` sum logprob | 19/70 | 0 | 1 | 8/70 | 0 | 0 | fail |

## Source-Trace Router Scout

After killing likelihood sketches, the next selected branch was the existing
`source_trace_router` harness. It tests whether interpretable source trace
features can route source candidates while equation-permutation and
source-destroying controls check whether the signal is real trace consistency.

Result:

- status: `source_trace_router_fails_gate`
- live: `20/70`, clean source-necessary `1`, control union `0`, accepted harm
  `2`
- holdout: `10/70`, clean source-necessary `1`, control union `0`, accepted
  harm `0`
- holdout source-necessary ID survives equation permutation, so it is not a
  robust trace-consistency signal

Decision:

- Do not promote source-trace routing.
- Keep it as a source-surface diagnostic clue only.
- Next highest-value branch should change the source signal, not tune shallow
  numeric/trace routers.

## Artifact Hashes

- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_normpred_sketch_cpu.jsonl`
  - sha256: `b08b0f5eab854dd7da9f4099238e5f300a2c2508a66341f58ae40d0adaa91254`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_normpred_sketch_cpu.jsonl`
  - sha256: `932e5449d5174c1f484f1535575e5b06763bc797fd83b8fec804471d300d5f15`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/normpred_cpu_sketch_gate.json`
  - sha256: `431844070c012be613b98d64c1effa1b88a14a542d37cc8c2bdf6320942fca55`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/normpred_sumlogprob_cpu_sketch_gate.json`
  - sha256: `51893cebe1a2cbcb1e91788fbf98d8b721fbbdf858ead6787e63e9efe0ff2e3e`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/normpred_answer_template_cpu_sketch_gate.json`
  - sha256: `b14d22ceda15c5d83104becbe0645d078c5d47a3c9872bb98d4c0217ec3a92b1`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/normpred_answer_template_sumlogprob_cpu_sketch_gate.json`
  - sha256: `ad33fd03dac92f5f85c6fcfc24562b6a19c6ba065fb0f5e6c98ed1c2cee9724f`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/source_trace_router_after_sketch_kill.json`
  - sha256: `e4e5600e139efbf7bc068ff2117e172cba9f87055e9477f51839a90175c54c03`

## Next Gate

Do not run more source-likelihood sketch variants unless a new predictor
changes the hypothesis.

Next highest-value branch:

- source-controlled syndrome prediction from a richer source signal, with the
  same target-candidate decoder and strict source-destroying controls
- exact gate: train/evaluate a predictor that can recover the C2C syndrome
  bound's source-necessary IDs without using C2C final answers

MPS hard blocker remains:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Do not launch MPS jobs until PID `31103` is cleared.
