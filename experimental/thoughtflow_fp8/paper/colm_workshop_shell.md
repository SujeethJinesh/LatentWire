# ThoughtFlow-FP8 COLM Workshop Shell

Status date: 2026-05-06

## Current Policy Status

**Stopped for the current anchor/recent/phase/math policy family; demoted or
killed for the pre-registered `rdu_topk`, `psi_topk`, and `vwac_topk`
successors.** The current
interpretable retention family supports a diagnostic/falsification claim only,
and should not be tuned further on the available saved traces. The first
successor evaluation, `rdu_topk`, clears the original frozen sparse-cache gate
by using delayed prefix self-attention recurrence rather than token labels or
recent reserves, but a first alternate-surface reproduction check does not keep
it separated from a stopped same-family sparse row, and a larger independent
saved-trace surface fails cross-family separation. Two later successor
registrations are also consumed: `psi_topk` fails decisively on a fresh C2C GSM
surface, and `vwac_topk` fails on a fresh C2C SVAMP surface.

This shell is a scoped workshop-paper scaffold, not a positive-method
submission draft. The current evidence supports a falsification study: a
promising Mac-local sparse-cache signal was found, reproduced on the same
surface, and then killed by stricter same-family and cross-family reproduction
checks; two fresh-surface successor signals then failed without retuning.

A cached split diagnostic supports the 0.20 result at the level of mean
margins, but it is not an independent reproduction: all deterministic half-size
partitions keep `rdu_topk` as the best compressed row, while only 2/4 partitions
clear both paired CI highs below zero. A separate measured no-retuning rerun on
the same 74-trace surface reproduces the cached result exactly on this
deterministic local stack and adds same-family/cross-family plus oracle/headroom
readouts, but it is still same-slice evidence. A measured alternate-surface
check with longer prefix/continuation settings preserves cross-family wins over
R-KV-like and ThinKV-like, but the stopped sparse ThoughtFlow row is lower than
`rdu_topk` by 0.006 NLL. The larger independent saved-trace check scores 89
traces from the chat SVAMP surface and makes R-KV-like the best compressed row
at NLL 3.981 versus `rdu_topk` at 4.014. The successor is therefore demoted to
diagnostic rather than kept alive as a positive branch.

## Candidate Story Only If Revived

Most deployers need training-free KV compression for already released reasoning
models, not only new architectures trained around compressed attention. A
revived ThoughtFlow-FP8 story would be:

> Explicit anchor/fair-span/phase-transition retention plus FP8 byte budgeting
> can reduce KV footprint for existing reasoning models while avoiding the
> delayed-utility and span-bias failures that pure current-query or redundancy
> selectors miss.

This story only survives if the policy beats strong nominal-budget baselines on
quality or perplexity, not just on protected-token recall.

## Current Evidence

| Artifact | Evidence | Decision |
|---|---|---|
| `phase2/phase_eviction_analysis.md` | Synthetic traces: ThoughtFlow preserves phase and anchor labels at matched keep rate. | Alive only as synthetic policy evidence. |
| `phase2/real_trace_retention_sweep.md` | Saved real generated traces: ThoughtFlow never beats the strongest proxy across keep fractions 0.10-0.35. | Weakened. |
| `phase2/hidden_saliency_retention_probe.md` | Distilgpt2 attention, final-hidden, key, value, and KV-norm telemetry: ThoughtFlow beats `value_norm_topk` on phase recall by +0.508 paired mean, but math-state CI crosses zero and LongFlow-like ties it. | Mixed; diagnostic, not a revival. |
| `phase2/perplexity_impact_proxy.md` | Distilgpt2 retained-context NLL: ThoughtFlow-saliency-recent beats old ThoughtFlow, LongFlow-like, and ThinKV-like, but loses to R-KV-like. | Weakened but more diagnostic. |
| `phase2/policy_sweep.md` | Train-selected ThoughtFlow-family policy ties R-KV-like on 12 held-out traces, 3.480 vs 3.482 NLL. | Mixed tie-range result. |
| `phase2/kv_drop_quality_probe.md` | Actual CPU sparse-cache pruning plus a 24-config train-fixed sparse sweep. The selected row has held-out NLL 3.340 vs ThinKV-like 3.385 and R-KV-like 3.420; paired CI still crosses zero vs ThinKV-like. | Mixed/promising, not revived. |
| `phase2/frozen_sparse_cache_probe.md` | Larger no-retuning CPU sparse-cache slice on 74 traces. The stopped family still fails, but pre-registered `rdu_topk` reaches NLL 3.779 versus ThinKV-like 3.900 and R-KV-like 3.939, with paired CIs below zero against both. | Historical first-surface candidate only; later gates demote it. |
| `phase2/rdu_robustness_diagnostic.md` | Cached split/paired diagnostic over the same 74-trace 0.20 rows. `rdu_topk` is best on even, odd, first-half, and second-half partitions, with all split mean margins above 0.03 versus R-KV-like and ThinKV-like; 2/4 split partitions also clear both paired CI highs below zero. | Explains why the first surface looked promising, but not a fresh reproduction. |
| `phase2/rdu_no_retune_reproduction_check.md` | Measured same-slice rerun of the frozen probe on current Mac hardware. The measured result exactly matches the cached first-surface gate: `rdu_topk` NLL 3.779, margins +0.160 vs R-KV-like and +0.121 vs ThinKV-like, zero measured-cached NLL drift for all policies. Per-trace compressed oracle NLL is 3.634, leaving `rdu_topk` 0.145 above oracle and 0.931 above full cache. | Same-surface bookkeeping only; later independent surfaces demote the branch. |
| `phase2/rdu_alt_surface_reproduction_check.md` | Measured alternate surface with `max_length=112` and `continuation_tokens=32`. `rdu_topk` NLL is 3.594 and still beats R-KV-like by 0.087 and ThinKV-like by 0.256 with paired CIs below zero, but `tf_sparse_r0.55_p0.05_m0.12_a2` reaches 3.588. | Weakened/not reproduced because strict same-family separation fails. |
| `phase2/rdu_independent_trace_reproduction_check.md` | Larger independent saved-trace no-retuning surface. The 96-row chat SVAMP input surface yields 89 scored traces; R-KV-like is best compressed at NLL 3.981 and beats `rdu_topk` at 4.014, so cross-family separation fails. | Not reproduced; demote `rdu_topk` to diagnostic. |
| `phase2/psi_fresh_sparse_cache_check.md` | Fresh C2C GSM surface. `psi_topk` reaches NLL 7.899 versus ThinKV-like 3.906 and R-KV-like 3.960. | Killed. |
| `phase2/vwac_fresh_sparse_cache_check.md` | Fresh C2C SVAMP surface. `vwac_topk` reaches NLL 4.336 versus R-KV-like 4.096 and ThinKV-like 4.162. | Killed. |
| `phase2/stop_pivot_decision_20260506.md` | Stops current policy-family tuning on the available saved traces; allows only a future pre-registered new utility signal evaluated once. | Stop/pivot gate. |

The most recent proxy scored 24 saved traces at 0.20 retained-prefix budget:

| Policy | NLL | Delta NLL vs full | Mean PPL |
|---|---:|---:|---:|
| Full context reference | 2.101 | 0.000 | 9.7 |
| R-KV-like | 3.419 | 1.319 | 38.7 |
| ThinKV-like | 3.583 | 1.482 | 44.6 |
| LongFlow-like | 3.961 | 1.861 | 74.2 |
| ThoughtFlow | 3.961 | 1.861 | 74.2 |
| ThoughtFlow-recent | 3.562 | 1.461 | 44.2 |

Interpretation: a simple recent-token reserve repairs part of the failure mode,
which means pure phase-marker protection was too far from the continuation
objective. The branch is still not revived because the R-KV-like retained-prefix
proxy remains stronger under the nominal-budget accounting used here.

The real hidden/KV telemetry sharpens the diagnosis but does not change the
decision:

| Probe row | Keep rate | Anchor recall | Phase recall | Math-state recall |
|---|---:|---:|---:|---:|
| ThoughtFlow | 0.207 | 1.000 | 1.000 | 0.918 |
| LongFlow-like | 0.207 | 1.000 | 1.000 | 0.918 |
| ThinKV-like | 0.207 | 1.000 | 0.948 | 0.848 |
| best real hidden/KV proxy: value-norm top-k | 0.207 | 0.000 | 0.492 | 0.845 |
| train-selected ThoughtFlow sweep policy | 0.207 | 1.000 | 0.430 | 0.956 |

Paired against `value_norm_topk`, the best ThoughtFlow-family row has phase
recall delta +0.508 with 95% CI [+0.436, +0.579], but math-state recall delta
+0.073 with 95% CI [-0.078, +0.223]. This rules out using phase-recall telemetry
alone as a positive claim.

The CPU sparse-cache probe is the strongest Mac-feasible quality gate so far:
the model first builds the full prefix cache, each policy prunes that cache at
the same budget, and the continuation is scored from the sparse cache. It still
does not revive the branch:

| Policy | NLL | Paired delta vs R-KV-like | Paired delta vs ThinKV-like |
|---|---:|---:|---:|
| full cache | 2.142 | -1.296 [-1.533,-1.066] | -1.247 [-1.418,-1.061] |
| ThoughtFlow-saliency-recent | 3.372 | -0.067 [-0.151,+0.011] | -0.018 [-0.104,+0.072] |
| ThinKV-like | 3.389 | -0.049 [-0.192,+0.077] | 0.000 |
| ThoughtFlow-recent | 3.399 | -0.040 [-0.169,+0.074] | +0.010 [-0.037,+0.056] |
| R-KV-like | 3.438 | 0.000 | +0.049 [-0.078,+0.191] |
| LongFlow-like | 3.588 | +0.150 [-0.016,+0.300] | +0.199 [+0.105,+0.295] |

The bounded train-fixed sparse sweep selects
`tf_sparse_r0.55_p0.05_m0.12_a2` on 12 train traces. On 12 held-out traces it
gets NLL 3.340 versus ThinKV-like 3.385 and R-KV-like 3.420, clearing the mean
0.03 margin. The paired delta is -0.080 versus R-KV-like with 95% CI
[-0.152,-0.014], but only -0.045 versus ThinKV-like with 95% CI
[-0.226,+0.182]. The fixed `thoughtflow_saliency_recent` incumbent is even
better in held-out mean NLL at 3.304, but its paired CIs also cross zero.

Interpretation: the best train-fixed sparse policy is now a promising
falsification candidate, not a positive result. It needs a larger frozen
sparse-cache slice with no further policy tuning.

The larger frozen slice was then run without any retuning:

| Policy | Traces | NLL | Delta vs R-KV-like | Delta vs ThinKV-like |
|---|---:|---:|---:|---:|
| full cache | 74 | 2.848 | -1.091 [-1.254,-0.938] | -1.052 [-1.199,-0.919] |
| rdu_topk | 74 | 3.779 | -0.160 [-0.264,-0.050] | -0.121 [-0.211,-0.037] |
| ThinKV-like | 74 | 3.900 | -0.039 [-0.100,+0.015] | 0.000 |
| frozen sparse ThoughtFlow | 74 | 3.908 | -0.031 [-0.078,+0.020] | +0.008 [-0.060,+0.085] |
| ThoughtFlow-saliency-recent | 74 | 3.920 | -0.019 [-0.048,+0.006] | +0.020 [-0.030,+0.074] |
| R-KV-like | 74 | 3.939 | 0.000 | +0.039 [-0.015,+0.099] |
| LongFlow-like | 74 | 4.158 | +0.219 [+0.136,+0.308] | +0.258 [+0.178,+0.337] |

Interpretation: the stopped family remains ruled out, but `rdu_topk` initially
clears the first pre-registered successor surface before later demotion gates. The signal is not a
phase-marker policy in disguise: aggregate telemetry retains only 3.1% of
phase markers and 21.7% of math-state labels, while retaining 56.8% of tokens
whose strongest recurrence bucket is 16-31 tokens.

## Demotion Result And Next Gate

The current anchor/recent/phase/math policy family should not be tuned further
on the available saved traces. The pre-registered recurrence-distance successor
cleared the first gate, then failed reproduction:

- `rdu_topk`: NLL 3.779 at keep rate 0.213.
- Margin versus R-KV-like: 0.160 NLL; paired delta -0.160, 95% CI
  [-0.264,-0.050].
- Margin versus ThinKV-like: 0.121 NLL; paired delta -0.121, 95% CI
  [-0.211,-0.037].
- Telemetry: anchor retention 0.706, phase retention 0.031, math-state
  retention 0.217; recurrence-bucket retention is 0.226 for lag 8-15 and
  0.568 for lag 16-31.
- Cached deterministic split diagnostic: all four half-size partitions keep
  `rdu_topk` as the best compressed row and preserve >=0.03 mean margins versus
  R-KV-like and ThinKV-like; 2/4 partitions clear both paired CI highs below
  zero.
- Measured no-retuning same-slice rerun: zero measured-minus-cached NLL drift
  for all policies; `rdu_topk` remains best compressed with the same margins and
  paired intervals.
- Strict measured separation: stopped-family margins versus `rdu_topk` are
  +0.141 and +0.129 NLL; cross-family margins are +0.160 versus R-KV-like,
  +0.121 versus ThinKV-like, and +0.379 versus LongFlow-like.
- Measured oracle/headroom: per-trace compressed oracle NLL is 3.634,
  `rdu_topk` is 0.145 NLL above that oracle and 0.931 above full cache, and the
  `rdu_topk` oracle-hit rate is 0.419.
- Measured alternate-surface check: with `max_length=112` and
  `continuation_tokens=32`, `rdu_topk` still beats cross-family rows
  R-KV-like and ThinKV-like, but the stopped sparse ThoughtFlow row reaches
  NLL 3.588 versus `rdu_topk` at 3.594, so same-family separation fails.
- Independent saved-trace check: R-KV-like is best compressed at NLL 3.981
  versus `rdu_topk` at 4.014, so cross-family separation fails.
- Failure decomposition: long-prefix/high-RDU-density rows are the largest
  regression bucket (`rdu_topk - R-KV-like = +0.213` NLL), while short-prefix
  low-density rows still favor `rdu_topk` by `-0.049` NLL versus R-KV-like.

The highest-value method branch is no longer recurrence-distance utility,
prefix-surprisal utility, or value-weighted attention contribution as
implemented here. These consumed signals should not be retuned on the current
or fresh surfaces used above. Any revival requires a genuinely new
pre-registered utility signal evaluated once on a fresh/larger frozen slice
with strict same-family versus cross-family separation and oracle/headroom
diagnostics.

Saturated: synthetic marker-retention, text-prefix-only policy tuning, the
current anchor/recent/phase/math frozen candidates, `rdu_topk`, `psi_topk`, and
`vwac_topk` on the available Mac-local surfaces. Still alive: the sparse-cache
probe as diagnostic infrastructure and real KV/hidden telemetry to explain
eviction bias.

## Limitations

- The retained-context NLL proxy is not sparse-KV decoding; the newer CPU
  sparse-cache probe does drop KV entries, but it is still not GPU/FP8 serving
  evidence.
- Distilgpt2 is a small model and not a reasoning model; this is a Mac-local
  falsification proxy, not a benchmark result.
- Current saved traces are small and not seed-repeated; the measured
  same-slice reproduction rerun is deterministic, and the alternate-surface
  check weakens `rdu_topk` by failing same-family separation.
- The split diagnostic reuses the same 74 traces and should not be counted as a
  new independent reproduction.
- There is no real FP8 numerical drift measurement in this gate.
- There is no latency, throughput, or serving-system result.
- Current baselines are local proxies, not faithful implementations of LongFlow,
  R-KV/R-KVHash, ThinKV, PM-KVQ, or RaaS.

## Next Gate

Do not move to a broad GPU benchmark for the current `rdu_topk`, `psi_topk`, or
`vwac_topk` branches. The next exact gate is either stop/pivot, or one new
pre-registered utility signal targeting a specific observed failure bucket and
evaluated once on a fresh/larger frozen sparse-cache surface.

1. Do not tune further on the current saved traces.
2. Do not spend GPU time on the consumed successor signals unless a new
   pre-registered signal first clears same-family and cross-family CPU
   sparse-cache checks.
3. Report paired uncertainty, per-span keep telemetry, recurrence misses, and
   FP8 round-trip error separately.
4. Keep promotion only if the new signal beats the strongest proxy on quality
   at matched bytes while preserving interpretable keep-rate telemetry.

Until a new pre-registered signal clears that gate, ThoughtFlow-FP8 should
remain a diagnostic paper shell rather than a positive-method claim.
