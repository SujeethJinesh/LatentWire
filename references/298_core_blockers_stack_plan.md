# Core Blockers Stack Plan

Scope: cross-model communication in LatentWire, as of the 2026-04-18 readout and the current blocker map.

Working conclusion:
- the remaining gap is not a single selector tweak
- the best surviving GSM8K result is still narrow, seed-fragile, and asymmetric across pairs
- the next useful fixes need to stack, not replace one another

## Rank-ordered blocker stack

### 1. Teacher objective

Why this is the top blocker:
- the current transport paths can be made to fit calibration data, but the fit does not reliably survive held-out reasoning
- the readout shows that richer supervision is more promising than another static routing heuristic
- latent regression alone is too weak a target for a cross-model bridge

Smallest next fix:
- move the teacher from latent MSE toward a span/readout/logit-style objective
- prefer prompt-local attention readouts, sequence likelihood matching, or interaction-structure targets over raw hidden-state regression

Composes with:
- every other blocker
- should sit on top of any geometry fix, not replace it

### 2. KV cache geometry

Why this is near the top:
- the exact Qwen2.5-0.5B -> Qwen3-0.6B pair is geometrically mismatched in KV heads and per-head dimension
- stock selective sharing can collapse even after a compatibility lift
- cheap soft transport and cheap orthogonal scoring were already too weak on the same pair

Smallest next fix:
- canonicalize KV geometry before routing
- try permutation-aware / OT-aware head matching, or a small gauge-aware transport map
- keep a lightweight residual correction on top, not as a substitute

Composes with:
- teacher objective
- query dependence
- layer matching

### 3. Query dependence

Why this is still a blocker:
- live query-aware sparse branches and retrieval-head heuristics are unstable
- fixed priors beat matched nulls in one regime but do not generalize cleanly
- the method looks query-conditioned, but not query-stable

Smallest next fix:
- make the bridge explicitly query-conditioned with pseudo-queries, per-example routing, or task-conditioned head budgets
- use a held-out calibration prior only as initialization, not as the whole method

Composes with:
- KV geometry canonicalization
- teacher objective
- layer matching

### 4. Layer mismatch

Why this matters:
- layer importance is not canonical across models
- grouped / layer-wise variants move behavior by task and split
- a single static layer policy is too brittle for a heterogeneous pair

Smallest next fix:
- match layers explicitly before head routing
- use a soft layer transport or a tiny layer-wise projector
- keep layer selection as a budget allocator, not the main method

Composes with:
- KV geometry
- query dependence
- teacher objective

### 5. Vocabulary / tokenizer mismatch

Why this is lower on the current Qwen pair but still important:
- it is not the first-order failure mode on the current Qwen family pair
- it becomes more important as we broaden to more heterogeneous model pairs
- span alignment and token remapping are still a likely future blocker

Smallest next fix:
- add dynamic span remapping before supervision
- use a tokenizer-transfer or token-basis adapter when token boundaries diverge
- supervise on aligned spans rather than raw token positions

Composes with:
- teacher objective
- query dependence
- any future broader hetero-pair benchmark

### 6. Benchmark variance

Why this is a blocker to making the method claim:
- the fixed-prior branch is seed-fragile
- the live sparse branch is also seed-fragile
- asymmetric transfer means a one-off positive run is not enough

Smallest next fix:
- standardize 3-seed replay
- keep paired bootstrap CIs and matched nulls
- freeze the calibration split and report only held-out scores

Composes with:
- every modeling fix
- must be applied before claiming a positive method

### 7. Bytes / latency

Why this is last:
- it is important for paper positioning, but it is not the core blocker to correctness
- a method can look efficient and still be wrong
- systems gains should be treated as a secondary axis until the method is stable

Smallest next fix:
- keep bytes, TTFT, and throughput in every table
- compare against a matched budget / matched token count setup
- report these as constraints on the method, not as a substitute for accuracy

Composes with:
- all of the above, but only as a reporting axis

## Stack order for the next method attempt

1. canonicalize KV geometry
2. add a stronger teacher objective
3. make routing query-conditioned
4. make layer assignment explicit
5. add span/tokenizer remapping if the pair broadens
6. lock seed / split variance controls
7. keep bytes / latency matched in every readout

## Practical implication

If the next transport-plus-correction branch still fails after geometry + teacher + query conditioning, the paper should stop claiming a universal cross-model communication method and lock to a blocker/mechanism contribution.
