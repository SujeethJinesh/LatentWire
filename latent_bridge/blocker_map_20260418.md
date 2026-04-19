## Blocker Map

This is the current best decomposition of why cross-model latent communication
is not yet a stable positive result in this repo.

### Blocker 1: Head-space symmetry / permutation mismatch

Observed symptom:

- fixed head priors are pair-conditioned and unstable across seeds
- cross-pair transfer is asymmetric
- grouped CCA changes behavior by task, which suggests basis sensitivity

Interpretation:

- head identity is not canonical across models
- a useful head in one model may be represented in a permuted or mixed basis in
  another

Current status:

- not solved
- simple permutation-aware rank matching also failed on the exact Qwen GSM70
  branch (`0.042857`), so plain head-order remapping is too weak on its own
- a direct gauge-aware Procrustes head-overlap score also failed on the same
  branch (`0.028571`), so cheap orthogonal-invariant head scoring is too weak
  on its own as well
- the compatibility-lifted `KVComm` replay also collapsed on the same Qwen
  pair (`0.000000` on GSM70), which is consistent with head identity and raw
  cache geometry becoming too brittle once KV-head count and per-head
  dimensionality stop matching directly

Next fix:

- permutation- or OT-aware head matching before sparse routing
- lightweight evaluator-level soft transport is also too weak on its own, so a
  stronger transport map likely has to move deeper into the translation path

### Blocker 2: Attention geometry is more important than raw KV similarity

Observed symptom:

- several routing heuristics improve one weak branch but only back to matched
  null levels
- expected-attention position priors were a null on grouped CCA
- head-level expected attention also only tied the old shuffled-null branch

Interpretation:

- preserving raw cache values is not enough
- the effective invariant is likely the induced QK / attention-logit geometry

Current status:

- not solved

Next fix:

- attention-fidelity or QK-geometry-preserving ranking / correction
- a stronger linear correction layer is still only a bounded repair on GSM70,
  so correction likely has to sit on top of a better transport map rather than
  replace it

### Blocker 3: Head importance is query- and task-conditional

Observed symptom:

- live sparse routing works only in narrow regimes
- grouped CCA helps SVAMP more than GSM
- fixed priors do not generalize broadly

Interpretation:

- there may be no single globally correct sparse route
- the useful head and position budgets depend on the reasoning regime

Current status:

- partially identified, not stabilized

Next fix:

- query-aware routing after the head basis is made more canonical
- task-aware comparisons rather than pooled scores

### Blocker 4: Keys and values are not equally useful transport channels

Observed symptom:

- `k_only` is consistently more promising than `v_only`
- dense full-KV transport has repeatedly failed or become confounded

Interpretation:

- keys carry retrieval geometry
- values are noisier to transplant directly and may be better left target-side

Current status:

- strong mechanism clue, not a final method

Next fix:

- keep `k_only` central
- test structured corrections on top of sparse key transport, not symmetric KV

### Blocker 5: Heterogeneous KV-head geometry itself is a transport barrier

Observed symptom:

- `Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen3-0.6B` mismatches both KV-head
  count (`2 -> 8`) and per-head dimensionality (`64 -> 128`)
- stock `KVComm` could not run on this pair without an explicit compatibility
  lift
- even with that lift, held-out GSM70 collapsed to `0.000000`

Interpretation:

- some baselines implicitly rely on matched or near-matched KV geometry
- once KV-head layout itself differs, training-free raw selective sharing may
  fail before higher-level routing logic even has a chance to help

Current status:

- newly identified and partially validated by the lifted `KVComm` replay

Next fix:

- canonicalize or transport KV geometry before selective routing
- keep baseline notes explicit about which methods natively support
  heterogeneous head geometry and which require a lift or learned adapter

## Immediate Plan

### Today

1. bootstrap `C2C` on the exact Qwen pair
2. record a fair baseline replay path for our GSM split
3. stop investing in weak expected-attention variants that only tie nulls
4. use the lifted `KVComm` failure as further evidence that deeper transport,
   not evaluator-side routing alone, is the next method class

### Next 1-2 days

1. implement permutation-matched head prior
2. run it on:
   - `gsm8k_eval_70.jsonl`
   - `gsm8k_100.jsonl`
   - `svamp_eval_70.jsonl`
3. compare against:
   - target-alone
   - text-to-text
   - zero-byte attenuation
   - random translated
   - grouped CCA fixed prior
   - grouped CCA shuffled-prior null
   - C2C

### What would count as real progress

- beat grouped-CCA shuffled null by a nontrivial margin
- stay above target-alone on at least one reasoning split
- survive one repeat seed or one second split
- remain competitive with or above the first published baseline
