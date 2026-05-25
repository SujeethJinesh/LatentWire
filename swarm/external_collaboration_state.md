# External Collaboration State Export

Generated: 2026-05-25 UTC  
Current commit at generation start: `6d92b39b`  
Share URL: `https://github.com/SujeethJinesh/LatentWire/blob/main/swarm/external_collaboration_state.md`

## 1. Project Snapshot

Current paper title: **Channel-Set Drift in Long-Reasoning W4A16 LLMs:
Mechanisms and a Budget-Tuned Remedy**. Near-term target is a COLM /
efficient-reasoning workshop submission; ICLR remains a follow-up only if a
stronger positive method or rotation+budget composition result lands. The
May 25 polish pass compressed the paper into a workshop-compliant main body
and moved detailed provenance into appendices/release documentation.

Problem statement: W4A16 channel protection assumes that high-magnitude
activation channels remain worth protecting through long autoregressive
reasoning traces. Across four reasoning models, top-1% channel membership
changes substantially by 20K-token horizons, undermining static or reactive
protected-channel policies.

Current state: the measurement story is strong and scoped. The positive-method
story is partial: M11b budget-tuned EMA has a Granite top-5 median recovery of
`0.449284091125` with wide CI, and a Nemotron top-10 median recovery of
`0.814739798903`; Nemotron top-5 is positive but does not beat static top-10.
ParoQuant is a stronger rotation baseline on Granite with median recovery
`0.753776848891`.

Blocker for a higher-ceiling paper: no channel-set method is yet a clean,
deployable, cross-model positive method. The current paper is best treated as a
mechanism and scoped partial-remedy workshop paper.

Last 10 commits:

```text
6d92b39b Audit: release reproducibility and paper claims
9ff24992 Release: document clean fast verification
a150edee Review: complete committee rounds 5 and 6
cf3b1a7d Release: minimal reproducibility package for COLM submission
6b87c88e Merge SA1 release framework
387891f0 Review: committee round 4 dual workshop
e78c40ff Paper: address committee round 3 load-bearing critiques
7b11d305 Paper: integrate Nemotron Path C and Phase 9 results
94afca9d Merge SA5 related work expansion
6c2b13a3 Merge SA4 figure audit and polish
```

## 2. Complete Experimental Inventory

| ID | Description | Models | Traces | Outcome | Key Statistic | Files |
|---|---|---|---:|---|---|---|
| Phase 0 | Granite-Tiny drift gate | Granite-4.0-H-Tiny | fixed AIME | PASS | migration `0.817838541667` | `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z` |
| Phase 1 | Granite-Small replication | Granite-4.0-H-Small | 24 | PASS | migration `0.843165650407`; strict set-leaving `0.566234756098` | `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z` |
| Phase 2 | Partial cross-family validation | Nemotron-3-Nano | 24 | PARTIAL_PASS | strict set-leaving `0.533713200380` | `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z` |
| Phase 3 | Granite-Tiny union protection | Granite-Tiny | 24 | KILL | union median recovery `0.0` | `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z` |
| Phase 4 | Granite-Small static union | Granite-Small | 24 | KILL | median recovery `0.0`; no-gap `0.375` | `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z` |
| Phase 5' | Pure Transformer control | DeepSeek-R1-Distill-Qwen-1.5B | 24 | DYNAMIC | strict set-leaving `0.670572916667` | `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z` |
| Phase 5'' | Qwen3.6 hook attempt | Qwen3.6-35B-A3B | 0 | FAIL_INFRA | no public hook path under constraints | `experimental/outlier_migrate/phase5_double_prime/results/om_phase5dp_qwen36_20260512T070500Z` |
| Phase 6 | RSPR | none | 0 | SKIPPED | blocked by Phase 4 measurable-gap gate | `swarm/blocked_phase6_testbed_selection.md` |
| Phase 7 | Falcon-H1 replication | Falcon-H1 | 24 | PASS_MEASUREMENT | strict set-leaving `0.673611111111` | `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z` |
| Experiment D | Set-leaving decomposition | Granite/Nemotron/DeepSeek | packets | COMPLETE | strict set-leaving dominates systems-relevant drift | `experimental/outlier_migrate/decomposition_analysis/` |
| Experiment E | Threshold sensitivity | four packets | packets | COMPLETE | top-1% stable through top-2% | `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.md` |
| Step 9.0 | Four-model replication | Granite/Nemotron/DeepSeek/Falcon | packets | PASS | set-leaving `0.5337-0.6736` | `experimental/outlier_migrate/phase9/step9_0_decomposition_replication.md` |
| M2 | Position-conditional switching | Granite-Small | 12 | KILL_RANDOM_CONTROL_BEATS | median `-0.866837313391`; random margin `-0.667548290168` | `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z` |
| M10 | Hard position-binned scales | Granite-Small | 12 | KILL_RANDOM_CONTROL_BEATS | median `0.234447927834`; random margin `-0.761487459046` | `experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z` |
| M11 | EMA smoothing | Granite-Small | 12 | KILL_AMBIGUOUS | median `0.048299284138`; random-walk much worse | `experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z` |
| M18 | Activation+K coupling | Granite-Small | 12 | KILL_AMBIGUOUS | median `-0.343590844126`; beats random coupled by large margin | `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z` |
| DecDEC | Algorithmic reactive baseline | Granite-Small | 12 | BASELINE_REPORTED | median `-0.070034781258` | `experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z` |
| FFT | Spectral analysis | Granite dense packet | packet | COMPLETE | entropy `0.853097829878`; autocorr `100` tokens | `experimental/outlier_migrate/phase9/spectral_analysis.md` |
| Per-component | Layer/block dissection | Granite/Nemotron/Falcon/DeepSeek | packets | COMPLETE | SSM/attention/MoE block outputs drift comparably | `experimental/outlier_migrate/phase9/per_component_dissection.md` |
| M11b | Budget-tuned EMA | Granite-Small | 12 | PASS_M11B_BUDGET_MATTERS | top-5 median `0.449284091125`; CI `[-1.300900018791, 1.000790626547]` | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z` |
| M26 | Stable core | Granite-Small | 12 | AMBIGUOUS_M26 | median `0.177615807648`; random control much worse | `experimental/outlier_migrate/phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z` |
| KL | Dense-grid KL accumulation | Granite-Small | 12 | COMPLETE | mean KL `0.150302750276`, `0.132701884446`, `0.131406585383`; AR decays `0.4387-0.5147` | `experimental/outlier_migrate/phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z` |
| ParoQuant | Algorithmic rotation baseline | Granite-Small | 12 | BASELINE_REPORTED | median `0.753776848891` | `experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z` |
| M11b Nemotron | Path C static baseline salvage | Nemotron-3-Nano | 12 | PARTIAL_BUDGET_SIGNAL | top-5 `0.456736183270`; top-10 `0.814739798903` | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z` |

## 3. Exact Empirical Findings

- Strict set-leaving: Granite-Small `0.566234756098`, Nemotron
  `0.533713200380`, DeepSeek `0.670572916667`, Falcon-H1
  `0.673611111111`.
- M11b Granite top-5: median `0.449284091125`, CI
  `[-1.300900018791, 1.000790626547]`, 8/12 positive-gap traces.
- M11b Nemotron top-5: median `0.456736183270`, CI
  `[0.345887792388, 0.794986745913]`, but below static top-10 median
  `0.594383743618`.
- M11b Nemotron top-10: median `0.814739798903`, CI
  `[0.254439288178, 0.922552264880]`, beating static top-10 by
  `0.220356055286`.
- ParoQuant Granite: median `0.753776848891`, CI
  `[0.477044452405, 1.003769554323]`.
- M26 stable core: median `0.177615807648`, CI
  `[-0.116448558090, 0.999915931324]`.
- KL dense grid: static mean `0.150302750276`, DecDEC proxy mean
  `0.132701884446`, M11 mean `0.131406585383`; fitted AR decays
  `0.5147`, `0.4461`, `0.4387`.
- FFT: normalized spectral entropy `0.853097829878`, low-frequency power
  `0.274561265367`, autocorrelation length `100` tokens.
- Phase 4 no-gap rate: `0.375`.

## 4. Three-Mechanism Framework Status

Mechanism 1, boundary discontinuities are harmful: supported by M2 and M10,
where hard switching/binning was beaten by random controls.

Mechanism 2, smoothing helps but is insufficient by itself: M11 is much better
than random-walk but recovers only `0.048299284138` median. M11b shows that
smoothing plus larger budget can recover nontrivial gap on some models.

Mechanism 3a, signal staleness: M18 and DecDEC indicate that current-step or
cross-tensor signals have information but do not recover enough at top-1%.

Mechanism 3b, budget insufficiency: supported by M11b. The caveat is that the
best budget shifts across Granite and Nemotron, so this is not a clean portable
policy yet.

Mechanism 3c, compound error: weakened in the measured Granite KL packet. KL
growth is flat/sublinear with moderate AR decay rather than superlinear.

## 5. Scoop Audit Results

The paper cannot claim first demonstration of set-leaving: DecDEC already shows
short-horizon static outlier recall degradation. The defensible novelty is
reasoning-scale horizon, reasoning model class, hybrid Mamba-2 / parallel
hybrid / pure-Transformer scope, and reasoning benchmarks.

The Quamba2 contradiction is scoped to block-output tensors: SSM/Mamba,
attention, and MoE block outputs drift at comparable rates in the measured
hybrid models. Internal SSM state tensors remain untested.

Terminology discipline: use "decode-position channel-set drift" operationally;
disambiguate SmoothQuant activation-to-weight migration, MoBiQuant
precision-dependent token sensitivity migration, HCP/CHON hot channels, and
test-time TTQ.

## 6. Current Paper Draft State

Paper source: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`.
The release copy is `release/paper/paper.tex`; the rebuilt PDF is
`release/paper/paper.pdf`.

Current abstract and paper body integrate:

- Quamba2 and DecDEC scoop-aware framing;
- four-model set-leaving;
- M2/M10/M11/M18/DecDEC/M11b/M26/ParoQuant/KL;
- M11b CI honesty and positive-gap trace counts;
- ParoQuant as stronger rotation baseline;
- W4A16, small-model, and block-output scope limitations.

Committee rounds 3-6 are complete. Scores stabilized at Efficient Reasoning
`7.5`, Context Beyond Window `8.0`, average `7.75`, with no new load-bearing
critiques in rounds 4-6.

## 7. Major Autonomous Decisions

- Reran Nemotron static-1% only after diagnosing substantive metadata/quantizer
  failure in the original packet.
- Chose hedged M11b framing after Path C: top-10 passes on Nemotron, top-5 is
  positive but does not beat matched static top-10.
- Merged citation, figure, and related-work branches during paper integration.
- Stopped additional GPU experiments after Nemotron, per timeline lock.
- Completed local audits after subagent audit threads failed due usage limits.
- Fixed release scripts to fail loudly outside dry-run/fast-verify rather than
  misrepresent frozen claim replay as full GPU reproduction.

## 8. In Flight Or Queued

No GPU experiment is currently running. The remaining work after this export is
final documentation and any human-requested polish. Full model-inference
reproduction inside `release/` is not implemented; FAST_VERIFY clean-clone
verification passed twice.

## 9. Open Questions

- Submit the current mechanism/partial-remedy paper to the workshop, or hold
  for ICLR after a stronger positive method?
- Should ICLR follow-up prioritize rotation+budget composition, M18b
  cross-tensor top-5, or M11c budget-resolution sweep?
- Should ThoughtFlow-FP8 be tied into the format-dependent story, or kept
  separate to avoid confounding?
- Should the Quamba2 contradiction be strengthened with internal SSM state
  hooks before a higher-tier submission?
- Is a full GPU release runner required before external sharing, or is
  FAST_VERIFY plus archived packets sufficient for the workshop artifact?

## 10. Methods Considered But Not Run

Deferred/not run in this sprint: M12, M15, M17, M25, M27, M30, M31, M33, M16,
M18b, M11c, streaming-PCA, RPCA-Q, Kalman scale predictors, WaveQ-Decode,
KAR-Q, InfoBit-Q, and M28 long-decode calibration. M13, M14, M19-M24 were
dropped by scoop audit. Non-authorized families include method cocktails,
sigma-delta/error-feedback quantization, dithered quantization, sparse
autoencoder protection, robust H-infinity protection, and random
projection/reservoir methods.
