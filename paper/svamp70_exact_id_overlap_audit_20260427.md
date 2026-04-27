# SVAMP70 Exact-ID Overlap Audit

- date: `2026-04-27`
- status: `exact_id_overlap_audit_complete`
- scale-up rung: smoke / branch selection
- git commit at run: `c3814de3d335d4996d9a05446363521db692f27b`

## Question

Do the recent SVAMP70 live/holdout artifacts contain a reusable positive
example structure across independent branches, or are the apparent wins
branch-local and therefore poor evidence for another threshold sweep?

## Result

No audited branch has reusable canonical live+holdout clean IDs under target
preservation.

Canonical live clean source-only IDs are:

- `14bfbfc94f2c2e7b`
- `2de1549556000830`
- `41cce6c6e6bb0058`
- `4d780f825bb8541c`
- `bd9d8da923981d69`
- `ce08a3a269bf0151`

Recovered live IDs cluster around a few examples:

- `2de1549556000830`: recovered by semantic predicates, trace routers,
  source-predicate routers, and the harmful target-likelihood accept-all
  diagnostic.
- `4d780f825bb8541c`: recovered by semantic predicates, source-predicate
  routers, and target-likelihood accept-all.
- `ce08a3a269bf0151`: recovered by source-predicate routers and
  target-likelihood accept-all.

Canonical holdout clean source-only IDs are:

- `ab1e71e8928661d0`
- `daea537474de16ac`

Only `daea537474de16ac` appears in audited branch recoveries, and only through
the trace-router family that fails the full gate. The semantic-predicate and
likelihood receiver variants recover no canonical holdout clean IDs.

Adjacent scout positives are separate surfaces, not canonical holdout evidence:

- `chal171-240`: syndrome recovers `4157958051c69d70` but harms target-correct
  examples.
- `chal241-310`: syndrome recovers all four clean IDs on that adjacent slice,
  but again with target-self harm.

## Decision

Stop spending CPU cycles on another threshold/router sweep over the current
canonical SVAMP70 artifacts. The useful information is now a branch-selection
constraint:

- keep canonical live/holdout exact IDs for future falsification
- do not promote live-only semantic or likelihood receiver clues
- revive receiver gates only with a true condition-specific control harness
- revive source surfaces only after MPS clears or a CPU-feasible stronger
  prompt/model surface exists

## Artifacts

- `results/svamp70_exact_id_overlap_audit_20260427/exact_id_overlap_audit.json`
  - sha256: `358cb6b6db2a76dcea074df91e8e755d03d8114649cce78e019ed4f5626c4f5c`
- `results/svamp70_exact_id_overlap_audit_20260427/exact_id_overlap_audit.md`
  - sha256: `92b688053c8948331b7df070538f645dfaa2746a456ada6c53347cc665bd9ec0`

## Tests

No code changed for this audit. The previous likelihood harness regression
tests in this cycle passed:

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
```

Result: `9 passed in 0.09s`.

## Next Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` remains, implement and test the fair condition-specific
receiver-control analyzer before collecting more target-likelihood sketches.
If PID `31103` clears, resume source-surface/interface reset on MPS.
