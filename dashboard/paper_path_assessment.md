# Paper Path Assessment

- current answer: `BOUNDED_NEGATIVE_LOCKED_FOR_ONE_WAY_AND_L_Q1`
- target venue calibration: COLM Efficient Reasoning workshop; this is supportable as a bounded-negative/diagnostic LatentWire story, not a positive LatentWire method.
- powered LatentWire ladder: `1500` scored dev/gate rows, `359` gate rows, achieved MDE half-width `0.030641`.
- LatentWire deployable positive: `no`
- key LatentWire verdict: full source-only fusion does not beat source-index+confidence; only source+target-at-encoder upper bound helps.
- held-out one-way confirm: deployable WZ delta `-0.023622`, CI `[-0.060367, 0.013123]`; full source-only oracle delta `-0.031496`, CI `[-0.062992, 0.002625]`; upper bound delta `0.110236`, CI `[0.081365, 0.141732]`.
- L_Q1 two-way query packet: `KILLED`; delta `-0.022284`, CI `[-0.055710, 0.011142]`; controls did not collapse and query/reply ablations explain it.
- L-B1 AURC signal: `no`
- L-A2 status: no real generated-solution candidate-pool/verifier score cache exists yet.
- C_A1/C_F/CE13 offline signal: C_A1 limited and backfill-ready, C_F mixed/control-contaminated, CE13 no cache.
- exact result that makes a positive COLM submission viable: C_A1 native tail/CVaR replay beats matched controls with a non-regressing DeepSeek/Falcon sentinel, or a future L-A2 generated-solution cache produces a deployable dev/gate win without confirm access.
- fallback paper path: write LatentWire as a bounded-negative diagnostic about source-copy leakage and non-deployable oracle headroom.
