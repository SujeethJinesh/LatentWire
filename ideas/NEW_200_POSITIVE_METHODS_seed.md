According to a document from 2026-06-04, the latest screening package teaches a very clear lesson: **the current LatentWire score-packet/cache-smoke path is not the positive route, but the campaign is not dead.** The overnight CPU-only report explicitly says the run was Mac CPU/MPS cached screening only, not a GPU or confirmation run; it killed current deployable CacheWire/KVComm/C2C packet smokes as method evidence, parked L-A2 until a real generated-solution score/verifier cache exists, and put **Channel-Set C-A1 native tail/CVaR backfill first** in the GPU queue. 

Yes: we can and should use these results to generate a **new 200-method positive-candidate pool**. But the important change is this: **the 200 methods should go into an idea backlog, not straight into the experiment queue.** The operating plan says ideas must pass math-vetting, adversarial review, scoop review, lineage audit, pre-launch review, and planted tests before entering the registry or GPU queue. 

## What we have learned

### 1. LatentWire score packets are basically exhausted as a positive route

The original LatentWire paper already drew the boundary: the packet beats target-only but does **not** beat explicit source-index; ARC source-index is 0.346 versus packet 0.344, OpenBookQA source-index and packet are both 0.378, and the packet follows the source choice at about 0.995–0.999. The paper therefore claims narrow same-family candidate transfer, not evidence synthesis beyond the source. 

The new Codex results strengthen that boundary. One-way WZ, full source-only score fusion, L-B1, and L-Q1 do not beat hard baselines. The only positive upper bound was **source+target-at-encoder**, which is not deployable. Mechanistically, that means the source has information, but a source-only packet cannot choose which bits the receiver needs.

So the next LatentWire positive candidates must **change the interface**:

```text
Do not keep trying:
source scores → tiny one-way packet → public decoder.

Try instead:
receiver/query/context/cache state → source-conditioned latent/cache/verifier evidence → receiver rerank or verify.
```

### 2. CacheWire/C2C/KVComm smokes are useful as controls, not yet methods

The overnight report says the current CacheWire oracle bound is alive, but the deployable trace was killed: dense teacher 16/32 versus target 8/32, but deployable trace 13/32 versus target/zero 14/32 and label-shuffle 15/32. It also says the destructive-control harness killed existing KVComm and C2C packet smokes as method evidence: zero-source agreement was 1.0, teacher-delta tied zero/target, candidate-delta matched was worse than control, and answer packet was an answer-text leak. 

That is valuable. It tells us that **dense/cache communication can still be a positive area**, but only if we stop trusting raw matched accuracy and instead build against wrong-row, zero-source, query-only, reply-only, and answer-leak controls from the start.

### 3. L-A2 is still alive, but the current cache is not enough

The report says L-A2 was parked because only 3 generated-text smoke rows existed and there were no source/target/verifier scores.  Later v3 apparently added a verifier field but still had too few useful prompts. So L-A2 is not dead; it needs a stronger generator and a real verifier-rerank cache.

This remains the best LatentWire positive route because generated-solution reranking has much higher entropy than 4-way or 10-way MCQ. The old plan already says L-A2 generated-solution verifier rerank is higher-ROI than MMLU-Pro because candidate entropy is larger and score shape has more room to matter. 

### 4. Channel-Set is now the main positive-method bet

The overnight report says **C-A1 backfill first**: 6 C-A1 gate rows, 5 positive medians, DeepSeek/Falcon checker medians 0.5180/0.3898, and C-F currently control-contaminated. 

But the Channel-Set source paper shows why we must be strict. Tight clipping improved Granite’s recoverable positive-gap tail from ParoQuant median 0.754 to 0.922 and moved the worst trace from -26.37 to 0.601, but the same setting lowered DeepSeek from 0.756 to 0.518 and regressed Falcon’s worst trace. 

So C-A1 can become a positive method only if it is **not just Granite tight-clip retuning**. It must pass DeepSeek/Falcon sentinels, matched ParoQuant parity, median non-regression, and CVaR/worst-trace improvement.

### 5. The next 200 methods should be generated from failure modes, not from generic creativity

The operating system already has the right architecture: every result wave should trigger ideation, failure-learning, anomaly mining, lineage audit, and hard baseline review; every candidate must beat hard baselines, survive controls, pass planted tests, have raw evidence, map to a figure, and be clean-room replicated before becoming a breakthrough candidate. 

So yes, we can generate 200 new positive-method candidates. But the right expected funnel is:

```text
200 positive-oriented ideas
→ 60 survive obvious killed-mechanism/scoop filters
→ 20 become registry cards
→ 8–12 run Stage-1 cached screens
→ 2–4 get GPU backfill/small live gates
→ maybe 1 positive method survives
```

That is the correct shape.

---

# A new 200-method positive-candidate pool

Below are **100 Channel-Set candidates** and **100 LatentWire / CacheWire candidates**. These are not all queueable. They are a new **idea bank** for Codex’s Ideation Lab. Each one must still pass math, adversary, scoop, lineage, planted-test, and baseline review before becoming a registry card.

## A. Channel-Set Drift: 100 positive-method candidates

### 1. Tail-risk / EVT / robust statistics

1. **C-T1 GPD-CVaR clip selector** — fit a generalized Pareto tail to per-trace ΔNLL and select clip by extrapolated tail risk.
2. **C-T2 model-local tail α selector** — learn per-model CVaR α, report α-transfer as result.
3. **C-T3 no-gap-aware tail selector** — optimize only on positive-gap traces, track no-gap fraction separately.
4. **C-T4 worst-trace repair budget** — allocate extra protection only to traces predicted to be catastrophic.
5. **C-T5 Huberized CVaR clip** — avoid one-trace overfit by Huberizing tail losses.

### 2. Survival analysis / hazard modeling

6. **C-S1 survival StableCore** — protect lowest-hazard channels by residence time.
7. **C-S2 competing-risk frontier** — model channels leaving because of magnitude drift versus layer-state drift.
8. **C-S3 Cox hazard warmup predictor** — predict channel leaving risk from first 512 tokens.
9. **C-S4 censored-trace survival core** — handle traces ending before 20K as censored survival observations.
10. **C-S5 hazard-weighted EMA** — update channels by leaving hazard, not raw magnitude.

### 3. Control theory / online adaptation

11. **C-C1 CUSUM drift trigger** — refresh policy only when set-leaving statistic crosses a control limit.
12. **C-C2 MPC budget router** — allocate top-K budget over the next horizon bucket from current drift state.
13. **C-C3 event-triggered clip retune** — retune clip only on KL spikes, not every position.
14. **C-C4 Kalman drift-state selector** — latent drift state controls policy library choice.
15. **C-C5 hysteresis policy switch** — prevent harmful hard switching by requiring sustained evidence before switching.

### 4. Rotation geometry / manifold methods

16. **C-G1 geodesic two-bin rotation** — prefill/decode rotations interpolated on SO(n), not linearly.
17. **C-G2 Cayley-path rotation library** — low-cost exact-orthogonal smooth rotation paths.
18. **C-G3 Grassmann subspace tracker** — protect a drifting subspace rather than individual channels.
19. **C-G4 rotation-distance regularized policy** — penalize large basis changes that break caches.
20. **C-G5 Procrustes warmup rotation selector** — choose closest library rotation from warmup activations.

### 5. Co-drift / graph methods

21. **C-D1 co-drift Givens pairing** — pair channels whose top-K membership drifts together.
22. **C-D2 channel graph community protection** — protect stable channel communities, not single channels.
23. **C-D3 graph-cut dynamic frontier** — choose frontier channels by min-cut between stable and drifting communities.
24. **C-D4 spectral hazard clustering** — cluster channels by hazard spectrum before protection.
25. **C-D5 pairwise drift-anti-correlated rotations** — rotate high-drift channels with compensating low-drift channels.

### 6. Information theory / rate-distortion

26. **C-I1 rate-distortion protection budget** — allocate protected channels by expected KL saved per bit.
27. **C-I2 unequal error protection** — give tail-risk horizons stronger protection.
28. **C-I3 syndrome residual correction** — encode correction relative to static protected set.
29. **C-I4 entropy-coded dynamic frontier** — transmit dynamic frontier IDs only when entropy is low enough.
30. **C-I5 channel-set mutual information selector** — protect channels with highest (I(c_t;c_{t+h})), not magnitude.

### 7. Optimization / submodular selection

31. **C-O1 submodular KL-gain greedy** — select channels by marginal KL gain, test diminishing returns.
32. **C-O2 budgeted matroid selector** — per-layer budget constraints with global tail objective.
33. **C-O3 min-cost flow layer budget** — route protection budget across layers by CVaR gain.
34. **C-O4 robust knapsack over traces** — maximize worst-trace benefit under channel budget.
35. **C-O5 Pareto selector median-vs-tail** — choose policies on the median/CVaR Pareto frontier.

### 8. Reliability engineering / fault tolerance

36. **C-R1 N-version policy voting** — choose only channels selected by multiple independent policies.
37. **C-R2 fail-safe static fallback** — reject adaptive policy when confidence is low.
38. **C-R3 redundancy-coded protection** — duplicate protection for catastrophic layers.
39. **C-R4 safety-margin clip** — clip selected by worst-case confidence interval, not point estimate.
40. **C-R5 burn-in reliability test** — require warmup policy to pass a synthetic stress before running.

### 9. Signal processing / temporal basis

41. **C-P1 wavelet drift basis** — represent channel drift over decode with wavelet coefficients.
42. **C-P2 DMD drift modes** — dynamic mode decomposition of top-K membership.
43. **C-P3 low-pass stable core** — protect low-frequency channel components only.
44. **C-P4 high-frequency drift rejection** — reject channels with noisy high-frequency membership.
45. **C-P5 spectral entropy router** — choose static versus adaptive policy from spectral entropy.

### 10. Bayesian / empirical Bayes

46. **C-B1 hierarchical tail prior** — share tail-risk strength across models while allowing model-local α.
47. **C-B2 posterior clip selector** — choose clip maximizing posterior probability of tail win.
48. **C-B3 Bayesian hazard shrinkage** — shrink channel hazards toward layer-level priors.
49. **C-B4 posterior no-regression gate** — require (P(\Delta_{\text{median}}<0)<\epsilon).
50. **C-B5 Thompson policy screen** — allocate Stage-1 rows adaptively to uncertain policies.

### 11. Causal inference / uplift modeling

51. **C-U1 treatment-effect trace router** — predict which traces benefit from C-A1 versus ParoQuant.
52. **C-U2 causal column ablation selector** — finite-difference output impact for candidate channels.
53. **C-U3 counterfactual horizon control** — enforce wrong-horizon collapse as a causal test.
54. **C-U4 uplift-only tail policy** — run tail policy only when predicted uplift positive.
55. **C-U5 mediation drift→KL→loss audit** — choose methods that actually reduce the drift-mediated loss path.

### 12. Hardware-aware policies

56. **C-H1 cache-resident protection set** — constrain dynamic frontier to L2/cache-resident working set.
57. **C-H2 HBM-cost-aware CVaR** — optimize CVaR improvement per HBM byte.
58. **C-H3 fused-rotation budget policy** — only choose rotations compatible with fused kernels.
59. **C-H4 prefetchable frontier** — choose future frontier channels predictable enough to prefetch.
60. **C-H5 profiler-in-loop selector** — demote policies whose tokens/s loss exceeds threshold.

### 13. Architecture-specific but not BranchRot-overclaim

61. **C-A1 attention-sink sentinel** — separate sink channels from drifting channels.
62. **C-A2 SSM-state persistence contrast** — protect SSM-stable channels only where persistence actually holds.
63. **C-A3 MoE expert-local tail gate** — run tail clip only in experts with positive local headroom.
64. **C-A4 branch-global hybrid selector** — choose global rotation unless branch-local headroom exceeds threshold.
65. **C-A5 expert-frequency-weighted CVaR** — weight tail risk by expert activation frequency.

### 14. Warmup / prompt adaptation

66. **C-W1 fixed-library warmup selector** — choose among ParoQuant, C-A1, C-F, reject.
67. **C-W2 warmup no-gap predictor** — detect unrecoverable traces early and avoid ratio artifacts.
68. **C-W3 warmup tail-risk classifier** — predict catastrophic ParoQuant tail.
69. **C-W4 warmup α predictor** — select CVaR α from early activation features.
70. **C-W5 warmup model-regime card** — output static/EMA/rotation/tail/reject protocol.

### 15. Benchmark/task conditioning

71. **C-K1 math-vs-code drift router** — choose policy by task class.
72. **C-K2 answer-length-conditioned policy** — longer expected decode gets stronger tail guard.
73. **C-K3 self-consistency trace router** — high disagreement among sampled chains triggers tail policy.
74. **C-K4 reasoning-phase selector** — use different policy for scratchpad versus final answer tokens.
75. **C-K5 benchmark-transfer audit** — promote only if AIME and MATH agree.

### 16. Error decomposition / diagnostics turned into method

76. **C-E1 KL-growth router** — sublinear KL gets static policy; spike-like KL gets adaptive policy.
77. **C-E2 AR-residual drift predictor** — use AR residual decay to choose smoothing.
78. **C-E3 error-budget decomposition selector** — separate basis error from channel identity error.
79. **C-E4 quantization-gap oracle router** — only intervene where static/BF16 gap is recoverable.
80. **C-E5 tail-cause classifier** — classify tail failures as clip, rotation, no-gap, or budget.

### 17. Ensemble / portfolio methods

81. **C-M1 robust portfolio over policies** — choose a convex policy mixture by minimax regret.
82. **C-M2 policy bagging** — bootstrap traces to produce stable policy choices.
83. **C-M3 ensemble tail guard** — run C-A1 only if multiple selectors agree.
84. **C-M4 disagreement-triggered reject** — reject adaptive intervention when policies disagree.
85. **C-M5 policy diversity regularizer** — avoid all policies collapsing to known tight clip.

### 18. Learning-to-rank over channel policies

86. **C-L1 rank policies by per-trace uplift** — learn pairwise ordering of policies on dev/gate.
87. **C-L2 LambdaRank channel selector** — rank channels by NDCG over future important sets.
88. **C-L3 pairwise trace-policy model** — predict which of ParoQuant/C-A1/C-F wins.
89. **C-L4 calibration-task meta-ranker** — rank methods using measured drift stats.
90. **C-L5 no-confirm meta-learning guard** — meta-ranker trained only on dev/gate families.

### 19. Numerical linear algebra

91. **C-N1 pivoted QR stable core** — choose low-condition-number protected basis.
92. **C-N2 leverage-score channel protection** — protect high leverage columns, not magnitude.
93. **C-N3 randomized SVD drift subspace** — low-rank dynamic subspace screen.
94. **C-N4 condition-number-aware rotation** — avoid rotations that amplify tail errors.
95. **C-N5 Bures/OT distribution alignment** — align early/late activation covariance with exact orthogonal constraints.

### 20. Systems / deployment protocols

96. **C-Y1 measure-then-choose deployment card** — formalize the calibration checklist into a method.
97. **C-Y2 rejection-first quantization protocol** — “do not use channel protection” as a positive safety method.
98. **C-Y3 tail-safe ParoQuant wrapper** — a wrapper that either passes CVaR sentinels or falls back.
99. **C-Y4 budget escalation protocol** — move from top-1 to top-10 only when survival/CVaR gate predicts value.
100. **C-Y5 reviewer-defense bundle** — combine OSC/DecDEC/no-gap/ParoQuant parity into an automated pass/fail protocol.

---

## B. LatentWire / CacheWire: 100 positive-method candidates

### 1. Receiver-conditioned communication

1. **L-R1 receiver-query cache delta** — receiver sends uncertainty query; source replies with cache delta.
2. **L-R2 ambiguous-pair query** — receiver sends top-2 candidates; source sends margin only for that pair.
3. **L-R3 target-margin-conditioned syndrome** — syndrome code conditioned on receiver margin bucket.
4. **L-R4 receiver-entropy gated packet** — send only when target entropy exceeds threshold.
5. **L-R5 target-error-type query** — receiver asks for arithmetic/retrieval/format evidence category.

### 2. Cache / latent delta methods

6. **L-C1 compressed cache delta** — send source cache residual relative to target cache approximation.
7. **L-C2 control-trained LCF-lite** — train cache adapter with wrong-row/zero-source negatives.
8. **L-C3 adaptive-width cache flow** — choose latent width by target uncertainty.
9. **L-C4 selected-layer cache delta** — send only layers with positive source-target disagreement value.
10. **L-C5 cache-delta verifier rerank** — use cache delta only to rerank generated candidates.

### 3. Teacher-to-byte distillation

11. **L-T1 C2C-teacher packet distillation** — distill dense C2C effect into tiny packet.
12. **L-T2 LCF-teacher byte code** — learn code approximating LCF output delta.
13. **L-T3 teacher-KL packet** — minimize KL between teacher receiver and packet receiver.
14. **L-T4 teacher-margin sketch** — transmit only teacher’s pairwise ranking margins.
15. **L-T5 teacher-control contrastive distill** — distill only effects that disappear under wrong-row controls.

### 4. Generated-solution verifier reranking

16. **L-G1 verifier margin packet** — source sends quantized verifier margins over candidate solutions.
17. **L-G2 pairwise verifier tournament** — compressed pairwise preferences among generated chains.
18. **L-G3 negative-evidence packet** — rule out bad generated candidates instead of choosing best.
19. **L-G4 critique-category packet** — send compact error type: algebra, retrieval, formatting, contradiction.
20. **L-G5 solution-cluster packet** — send preferred cluster, not candidate ID.

### 5. Distributional / information-theoretic coding

21. **L-I1 conditional entropy code** — encode only (H(S\mid R)) residual via bins.
22. **L-I2 interactive Slepian-Wolf** — one tiny receiver query, one source syndrome.
23. **L-I3 unequal error protection over candidates** — allocate bits to uncertain candidates.
24. **L-I4 rate-adaptive score sketch** — send 0, 1, 2, or 4 bytes by expected value.
25. **L-I5 information bottleneck packet** — maximize (I(Z;Y\mid R)-\beta I(Z;A_S)).

### 6. Damage avoidance / selective prediction

26. **L-D1 source-harm detector** — packet predicts when source will damage target.
27. **L-D2 abstention certificate** — source sends “do not use me” evidence.
28. **L-D3 risk-coverage packet** — optimize AURC rather than accuracy.
29. **L-D4 conformal disagreement packet** — send prediction-set size / trust set.
30. **L-D5 damage-repair split packet** — separate repair and harm bits.

### 7. Multi-source and ensemble packets

31. **L-M1 disagreement-haplotype packet** — encode which sources agree/disagree.
32. **L-M2 diversity-weighted source packet** — favor source evidence decorrelated from target.
33. **L-M3 source coalition sketch** — compact coalition vote over candidates.
34. **L-M4 anti-correlated expert packet** — use weak but complementary source only on target blind spots.
35. **L-M5 multi-source damage gate** — trust source only when independent sources concur.

### 8. Calibration and proper scoring

36. **L-P1 proper-scoring-rule packet** — train packet to improve log score, not accuracy.
37. **L-P2 Brier-risk packet** — optimize calibration improvement.
38. **L-P3 isotonic residual packet** — send source calibration residual after target confidence.
39. **L-P4 ECE-aware trust code** — packet reduces calibration error at fixed coverage.
40. **L-P5 probability simplex codebook** — code calibrated belief shapes, not labels.

### 9. Retrieval / RAG reranking

41. **L-A1 passage-rerank evidence packet** — source sends compressed passage relevance deltas.
42. **L-A2 negative-passage packet** — source flags distractor passages.
43. **L-A3 citation-support packet** — compact support-vs-contradict evidence.
44. **L-A4 retrieval-cluster sketch** — send relevant passage cluster ID.
45. **L-A5 RAG answer-verifier packet** — source verifies answer with retrieved evidence.

### 10. Program/math/code tasks

46. **L-X1 final-answer-format uncertainty** — packet says extraction confidence.
47. **L-X2 unit/dimension-check packet** — source sends compact unit-consistency evidence.
48. **L-X3 symbolic-step verifier packet** — source flags failed algebra step.
49. **L-X4 code-test outcome packet** — source sends compressed pass/fail pattern over hidden tests.
50. **L-X5 invariant-check packet** — send whether candidate satisfies problem invariant.

### 11. Learned query bottlenecks

51. **L-Q1 tokenwise Q-former connector** — frozen endpoints, learned query bottleneck.
52. **L-Q2 sparse crosscoder packet atoms** — learn shared sparse features, transmit atom IDs.
53. **L-Q3 SAE feature packet** — source sends interpretable feature activations.
54. **L-Q4 cross-family alignment packet** — learned relative representation bridge.
55. **L-Q5 query-bottleneck verifier** — query module reads source state only for candidate verification.

### 12. Adaptive communication policy

56. **L-B1 budget router** — choose 0/1/4/16 bytes by expected value.
57. **L-B2 two-stage micro-packet** — send byte 2 only if byte 1 leaves high risk.
58. **L-B3 active query selection** — receiver chooses source question that maximizes information gain.
59. **L-B4 cost-aware cascade policy** — packet chooses target-only/source/strong-model route.
60. **L-B5 regret-bounded packet policy** — learn communication policy with regret guarantee.

### 13. Robust / adversarial controls baked into training

61. **L-S1 wrong-row contrastive loss** — train packet to fail on wrong-row source.
62. **L-S2 candidate-derangement adversarial training** — enforce candidate-order sensitivity.
63. **L-S3 zero-source collapse regularizer** — method must not work with zero-source.
64. **L-S4 source-copy penalty** — penalize (I(Z;A_S)).
65. **L-S5 query-only/reply-only orthogonality** — train query and reply so neither alone explains gain.

### 14. Ranking geometry

66. **L-N1 Bradley-Terry delta code** — transmit low-rank pairwise utility.
67. **L-N2 Plackett-Luce packet** — code source ranking distribution compactly.
68. **L-N3 Kendall-syndrome code** — encode correction from target ranking to source ranking.
69. **L-N4 topological ranking clusters** — send cluster in ranking manifold.
70. **L-N5 margin-spectrum packet** — DCT/wavelet sketch of candidate margin spectrum.

### 15. Text/control hybrids

71. **L-H1 same-byte nonsemantic tag packet** — compare opaque category tags to visible text.
72. **L-H2 compressed critique code** — non-natural-language critique symbol.
73. **L-H3 source-text-distilled packet** — distill same-byte text advantage into bytes.
74. **L-H4 text-vs-cache ablation packet** — choose if text or latent wins per row.
75. **L-H5 answer-leak-resistant hint code** — hint class without candidate identity.

### 16. C2C/LCF improvements

76. **L-F1 LCF with byte-exact accounting** — convert latent width to literal bytes/query.
77. **L-F2 LCF destructive-control benchmark** — use controls as training/eval contribution.
78. **L-F3 receiver-conditioned LCF** — condition cache flow on target uncertainty.
79. **L-F4 adaptive-layer LCF** — dynamic selected layers based on row difficulty.
80. **L-F5 LCF-to-reranker instead of generator** — use cache flow only for verifier/rerank.

### 17. Small-model / same-family specialization

81. **L-O1 same-family affine score bridge** — calibrated source→target score map.
82. **L-O2 model-family packet dictionary** — learned packet atoms per model family.
83. **L-O3 source-quality-aware packet** — suppress weak-source families like Phi-3 case.
84. **L-O4 family-mismatch detector** — refuse cross-family packet when fidelity/source quality mismatch.
85. **L-O5 same-family cache checksum** — detect whether cache/state alignment is valid before fusing.

### 18. High-entropy task design

86. **L-E1 64-candidate math rerank** — scale candidate pool to force beyond-label signal.
87. **L-E2 self-consistency cluster rerank** — source packet chooses among reasoning clusters.
88. **L-E3 multi-hop QA evidence rerank** — packet selects evidence path.
89. **L-E4 code repair candidate rerank** — packet ranks candidate code patches.
90. **L-E5 proof-step ranking packet** — source ranks proof continuations.

### 19. Mutual-information diagnostics turned method

91. **L-MI1 conditional-MI task router** — run packet only where (I_{\text{beyond}}) clears threshold.
92. **L-MI2 source-target redundancy penalty** — avoid sending source information target already has.
93. **L-MI3 complementarity score packet** — encode only source features uncorrelated with target.
94. **L-MI4 oracle-gap decomposer** — route to cache/query method if source+target oracle gap is high.
95. **L-MI5 information-gain byte scheduler** — allocate bytes by marginal (I(Y;Z\mid R)).

### 20. Systems / exposure accounting

96. **L-Y1 exposure-minimized cache delta** — minimize state exposure, not just bytes.
97. **L-Y2 cacheline-aware packet packing** — optimize framed/cacheline-rounded bytes.
98. **L-Y3 latency-aware packet gate** — send packet only if latency-adjusted utility positive.
99. **L-Y4 byte-frontier profiler card** — every method gets literal bytes, min-code bits, latency proxy.
100. **L-Y5 state-exposure Pareto router** — choose text/packet/cache by privacy/exposure/accuracy frontier.

---

## Which of these are most likely to succeed?

The 200-method pool should immediately collapse into a short queue. I would prioritize these.

### Top Channel-Set candidates

1. **C-T1 / C-A1 native GPD-CVaR clip selector.** This is still the best positive route. It directly attacks tail failures and has a clear GPU backfill gate.
2. **C-S1 survival StableCore with clean controls.** This is the best non-rotation bet, but C-F must fix the control contamination first.
3. **C-W1 fixed-library warmup selector.** This can turn C-A1/C-F/ParoQuant into a regime protocol rather than a single brittle retune.
4. **C-D1 co-drift Givens pairing.** Strong because it modifies ParoQuant’s actual mechanism rather than fighting it.
5. **C-U1 uplift trace router.** Only apply tail policy where it is predicted to help; likely avoids DeepSeek/Falcon regression.
6. **C-Y3 tail-safe ParoQuant wrapper.** Even if it is not glamorous, a safety wrapper that catches tail regressions could be a credible positive.

### Top LatentWire / CacheWire candidates

1. **L-G1/L-G2 generated-solution verifier margin/tournament packet.** L-A2 remains the best true positive route if the cache has enough useful prompts.
2. **L-R1 receiver-query cache delta.** This directly attacks the source+target-at-encoder oracle gap.
3. **L-F3 receiver-conditioned LCF.** Improves on C2C/LCF style, not old score packets.
4. **L-T2 LCF-teacher byte-code distillation.** Uses a strong dense teacher to teach a small packet what effect to imitate.
5. **L-S1/S2 control-trained cache communication.** Turns destructive controls into a training objective; useful even if mean gain is modest.
6. **L-MI1 conditional-MI task router.** Stops wasting packets on rows/tasks where source information is redundant.

---

## The key math checks behind the next wave

### LatentWire: avoid the killed source-copy mechanism

If (Z) is only a function of the source’s selected answer (A_S), then:

[
I(Y;Z\mid X,A_S,R)=0.
]

So no new method can be just a disguised source-index or source-confidence code. It must increase:

[
I(Z;Y\mid X,R)
]

while controlling leakage:

[
I(Z;A_S).
]

That suggests an objective like:

[
\max_Z I(Z;Y\mid X,R) - \alpha I(Z;A_S) - \beta |Z|.
]

The recent result that only the source+target-at-encoder oracle helps means we need **receiver-conditioned** source messages, high-entropy reranking, or dense/cache teacher distillation—not one-way source-only score packets.

### Channel-Set: optimize drift cost and tail risk

The Channel-Set source paper’s error model is:

[
E[\mathrm{err}(t)] \approx K{\epsilon_p(1-L(t_0,t))+\epsilon_q L(t_0,t)}.
]

So C-F/survival methods try to reduce the integral of (L(t_0,t)) over decode time. C-A1/tail methods instead optimize the high-loss tail:

[
\widehat{\mathrm{CVaR}}_{\alpha}(\theta)
========================================

\min_\eta
\left[
\eta+\frac{1}{(1-\alpha)n}\sum_i(\ell_i(\theta)-\eta)_+
\right].
]

The key guardrail is: **do not choose the tail policy and evaluate it on the same traces**. The plan already calls for dev/gate/confirm discipline and no-gap filtering. 

---

## What I would tell Codex to do with this

Do not ask Codex to immediately implement all 200. Ask it to create a **ranked, filtered idea backlog**.

```text
Read the latest dashboard reports and lessons ledger.

Current state:
- LatentWire one-way WZ, L-B1, L-Q1, current CacheWire/KVComm/C2C smokes are killed or bounded-negative.
- L-A2 is parked until a stronger generated-solution verifier cache exists.
- Channel-Set C-A1 is the first GPU backfill priority.
- C-F is parked until clean controls exist.

Task:
Create ideas/NEW_200_POSITIVE_METHODS.md using the 200-method pool below.
For every idea, assign:
  paper
  mechanism_family
  nearest killed method
  how it avoids the killed mechanism
  closest hard baseline
  cheap Stage-1 proxy
  GPU requirement
  kill condition
  expected decision value
  contribution_role

Then filter:
  200 ideas -> top 60 survivors -> top 20 registry candidates -> top 8 immediate screens.

Do not create registry cards for all 200.
Only graduate ideas that pass math + adversary + scoop + lineage audit.
Keep gpu_foreground.yaml empty.
Keep C_A1 GPU backfill first.
```

## Bottom line

Yes, these results are exactly the right raw material for a new 200-method ideation wave. But the new wave should be **failure-shaped**:

```text
LatentWire:
  stop source-only score packets;
  move to receiver-conditioned cache/latent deltas,
  generated-solution verifier reranking,
  LCF/C2C improvements under destructive controls,
  and teacher-to-byte distillation.

Channel-Set:
  push C-A1 native tail/CVaR backfill first;
  repair C-F controls;
  build model-local tail routers,
  survival cores,
  warmup selectors,
  and co-drift/rotation-coupled methods.
```

The most promising positive-method path right now is still **Channel-Set C-A1**, with **LatentWire L-A2 / receiver-conditioned CacheWire** as the next best LatentWire revival route.

---

## Elevated supplement: complementary-source / private-computation communication

This family was under-weighted in the 200-bank above and is elevated before triage. It is not permission to queue experiments; each item must first pass the receiver-conditioned ceiling gate:

```text
I(source_signal; Y | receiver_state) >> 0.28 bits
```

and must explain why the source has information the receiver cannot recover from its own forward pass.

1. **L-PC1 cross-family specialist source ceiling** — a math/code specialist source supplies hidden/logit/verifier evidence to a small general receiver.
2. **L-PC2 tool-augmented source ceiling** — the source privately runs calculator/code execution and transmits a compact result/verifier signal.
3. **L-PC3 retrieval-augmented source ceiling** — the source privately sees retrieved documents and sends evidence deltas, not answer IDs.
4. **L-PC4 extra-compute source ceiling** — the source spends N times more samples/CoT budget and transmits a cluster/verifier signal the receiver did not compute.
5. **L-PC5 private verifier over receiver candidates** — the receiver generates candidates, the source privately verifies them with stronger tools or more compute.
6. **L-PC6 cross-family trained residual fuser** — frozen endpoints with a tiny trained fuser, only after L-PC1 shows conditional headroom.
7. **L-PC7 private code-test outcome packet** — the source runs tests hidden from the receiver and sends a compact pass/fail syndrome.
8. **L-PC8 retrieval contradiction packet** — the source privately identifies which candidate is contradicted by retrieved evidence.
