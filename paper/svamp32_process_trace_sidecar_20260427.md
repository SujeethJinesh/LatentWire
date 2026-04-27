# SVAMP32 Process-Trace Sidecar Smoke

Date: `2026-04-27`

## Cycle State

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives answer masking and source-destroying controls.
2. Current paper story: target/no-source sampling creates reachable target-side
   candidates, but both numeric and process-text source sidecars fail to select
   those candidates under controls.
3. Exact blocker: source-derived non-answer information must select reachable
   target candidates without final-answer leakage, target priors, random-sidecar
   wins, or representation collapse.
4. Current live branches: process/latent ranking is weakened; broader
   target/no-source candidate-pool discovery or a trained frozen-latent
   connector is now higher value than more hand-built sidecars.
5. Highest-priority gate: CPU process-trace similarity sidecar on the SVAMP32
   clean6 target-only pool.
6. Scale-up rung: smoke.

## Method

Implemented `scripts/materialize_svamp_process_trace_sidecars.py`, a CPU-only
diagnostic sidecar producer. It:

- loads a decoder-compatible target set
- masks numerals in source and candidate reasoning text
- scores source-process text against target-side candidate-process text using
  TF-IDF cosine over unigrams/bigrams
- emits sidecars compatible with
  `scripts/analyze_candidate_score_sidecar_top_select.py`
- logs collapse/leakage telemetry: feature count, variance, effective rank,
  zero vectors, zero margins, top-label counts, and whether the selected value
  appears among unmasked source numbers

Three variants were tested:

1. `process_trace`: all target-side labels included.
2. `process_trace_sample_pref`: when a value appears in target and sampled
   labels, prefer sampled candidate text.
3. `process_trace_predonly_no_t2t`: exclude text relay and score only each
   row's normalized prediction value.

## Results

| Variant | Matched Correct | Matched Accepted | Control Clean Union | Source-Necessary Clean | Key Failure |
|---|---:|---:|---:|---:|---|
| `process_trace` | 0/6 | 0 | 0 | 0 | target-label zero-margin collapse |
| `process_trace_sample_pref` | 0/6 | 0 | 1 | 0 | shuffled/random recover reachable ID |
| `process_trace_predonly_no_t2t` | 0/6 | 2 | 1 | 0 | matched selects wrong values; random recovers clean ID |

The strongest diagnostic variant, `process_trace_predonly_no_t2t`, had:

- matched accepted IDs: `1d50b408c8f5cd2c`, `3e8a5691f5443495`
- matched clean correct: `0/6`
- random-sidecar clean correct: `575d7e83d84c1e67`
- control clean union: `575d7e83d84c1e67`
- top value in unmasked source numbers: `5/6`
- effective rank: `31.270460`
- zero vectors: `0`

This is not a representation-collapse failure in the variance/effective-rank
sense. It is a source-signal failure: answer-masked process similarity mostly
tracks problem/candidate lexical overlap and unmasked source-number overlap, not
which reachable target candidate is correct.

## Decision

Fail and prune hand-built process-trace similarity sidecars on this slice. This
is the third adjacent hand-built sidecar family to fail source controls or
answer masking after numeric candidate sidecars and earlier query-bottleneck
syndrome probes. Do not tune TF-IDF/process-text similarity further.

The next exact gate should either:

- widen target/no-source candidate-pool discovery to get more than two
  reachable clean IDs before spending on selectors, or
- train a genuinely different frozen-latent/rate-capped connector with
  source-destroying controls and collapse telemetry, rather than another
  deterministic hand-built sidecar.
