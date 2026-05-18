# External Collaboration State Export

Generated: 2026-05-18 UTC  
Current commit: `508d04d3`  
Share URL after push: `https://github.com/SujeethJinesh/LatentWire/blob/main/swarm/external_collaboration_state.md`

## 1. Project Snapshot

Current working title: **Decode-Position Channel Drift in Long Reasoning Traces**. The near-term target is a COLM workshop submission; the longer-term target is an ICLR follow-up only if a positive method replicates. The broader repo contains LatentWire and ThoughtFlow-FP8 artifacts, but the active sprint is OutlierMigrate.

Problem statement: W4A16 post-training quantization often relies on selecting and protecting salient activation channels. That is easiest if the top-magnitude channel set remains stable during long autoregressive decoding. Our measurements show that, in reasoning traces up to 20K decode positions, many top-1% channels leave the protected set. The paper now asks which mechanism explains this failure and whether any inference-time protection policy can recover the BF16-vs-static gap.

Current state: the measurement result is strong across Granite-4.0-H-Small, Nemotron-3-Nano, DeepSeek-R1-Distill-Qwen-1.5B, and Falcon-H1. Most interventions fail or are ambiguous. M11b budget scaling is the first mechanical PASS on Granite-Small: top-5 EMA protection median recovery is `0.4492840911245966`, beating static-top10 by `0.5010512676801815` median. The caveat is severe: CI95 is `[-1.3009000187907436, 1.00079062654749]` and only 8/12 traces have a recoverable static gap.

Blocker: to become a positive-method paper, M11b must replicate or another queued method must pass with controls. Without that, the defensible COLM paper is a mechanism/negative-result paper.

Last 10 commits:

```text
508d04d3 AUTO: outlier_migrate PASS + M11b budget scaling
52439527 Progress note: M11b static control halfway
3ad5f81f Progress note: M11b static control running
e43434b7 Progress note: M11b top10 running
7b619845 Progress note: M11b top5 complete
cba24211 Progress note: M11b top5 running
78d7704a Progress note: M11b top1 complete
c68b2479 Progress note: M11b top1 scoring
7da8181d Progress note: M11b active, queue updated
86212cb4 Paper: add latest KV and sensitivity prior work
```

## 2. Complete Experimental Inventory

| ID | Description | Models | Traces | Outcome | Key Statistic | Date | Files |
|---|---|---|---:|---|---|---|---|
| Phase 0 | Granite-Tiny drift gate | Granite-4.0-H-Tiny | fixed AIME set | PASS | migration `0.8178385416666667`, CI `[0.797265625, 0.8368489583333334]` | 2026-05-08 | `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z` |
| Phase 1 | Granite-Small scale replication | Granite-4.0-H-Small | 24 | PASS | migration `0.843165650406504`, CI `[0.8334349593495936, 0.8511432926829268]` | 2026-05-08 | `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z` |
| Phase 2 | Partial cross-model validation | Nemotron-3-Nano | 24 | PARTIAL_PASS | migration `0.820809591642925`; strict set-leaving `0.533713200379867` | 2026-05-09 | `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z` |
| Phase 3 | Granite-Tiny union protection | Granite-4.0-H-Tiny | 24 | KILL | union median recovery `0.0`, CI `[0.0, 0.7111432441994825]` | 2026-05-10 | `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z` |
| Phase 4 | Granite-Small intervention | Granite-4.0-H-Small | 24 | KILL | union median recovery `0.0`, CI `[0.0, 0.06954064195470389]`; no-gap `0.375` | 2026-05-12 | `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z` |
| Phase 5 prime | Pure-Transformer control | DeepSeek-R1-Distill-Qwen-1.5B | 24 | DYNAMIC | migration `0.8393787202380952`; strict set-leaving `0.6705729166666666` | 2026-05-12 | `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z` |
| Phase 5 double-prime | Qwen3.6 hook attempt | Qwen3.6-35B-A3B | 0 | FAIL_INFRA | SGLang exposes no public per-layer hook without source modification | 2026-05-12 | `experimental/outlier_migrate/phase5_double_prime/results/om_phase5dp_qwen36_20260512T070500Z` |
| Phase 6 | RSPR method | none | 0 | SKIPPED | Phase 4 no-gap `0.375` exceeded measurable-gap gate | 2026-05-12 | `swarm/blocked_phase6_testbed_selection.md` |
| Phase 7 | Falcon-H1 replication | Falcon-H1-0.5B-Instruct | 24 | WITHIN_LINEAGE_2_CONSISTENT | migration `0.7833543771043772`; strict set-leaving `0.6736111111111112` | 2026-05-12 | `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z` |
| Phase 8 | Kimi Linear | Kimi Linear | 0 | SKIPPED | not run after SGLang hook risk and queue reprioritization | 2026-05 | `swarm/goal.md` |
| Experiment D | Decomposition | Granite, Nemotron, DeepSeek | packets | COMPLETE | strict set-leaving vs within-set split computed | 2026-05-12 | `experimental/outlier_migrate/decomposition_analysis/component_decomposition.md` |
| Experiment E | Threshold sensitivity | four packets | packets | COMPLETE | top-1% stable through top-2%; top-5% changes split | 2026-05-12 | `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.md` |
| Step 9.0 | Premise replication | Granite, Nemotron, DeepSeek | packets | PASS | strict set-leaving: Granite `0.566234756098`, Nemotron `0.53371320038`, DeepSeek `0.670572916667` | 2026-05-12 | `experimental/outlier_migrate/phase9/step9_0_decomposition_replication.md` |
| M2 | Position-conditioned sets | Granite-Small | 12 | KILL_RANDOM_CONTROL_BEATS | M2 median `-0.8668373133910525`; random-bin `-0.19928902322343722` | 2026-05-14 | `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z` |
| M10 | Hard position-binned scales | Granite-Small | 12 | KILL_RANDOM_CONTROL_BEATS | M10 median `0.2344479278340464`; random-bin `0.9959353868798071` | 2026-05-16 | `experimental/outlier_migrate/phase9/results/om_phase9_m10_granite_small_vac12_20260515T085800Z` |
| M11 | EMA-smoothed protection | Granite-Small | 12 | KILL_AMBIGUOUS | best alpha median `0.048299284137681975`, CI `[-8.94115615101423, 0.7045992558730417]` | 2026-05-16 | `experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z` |
| M18 | Joint activation+K | Granite-Small | 12 | KILL_AMBIGUOUS | activation+K median `-0.34359084412632024`, CI `[-14.071191710978855, 0.7406021242797479]` | 2026-05-17 | `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z` |
| Post-M18 analyses | Five no-GPU diagnostics | four packets | packets | COMPLETE | K/V migration and recovery curves not identifiable; stable core exists | 2026-05-17 | `experimental/outlier_migrate/phase9/post_m18_analysis/` |
| DecDEC baseline | Algorithmic reactive selection | Granite-Small | 12 | PASS_BASELINE_REPORTED | DecDEC median `-0.07003478125820645`, CI `[-8.495693501232758, 0.6741763703501289]` | 2026-05-18 | `experimental/outlier_migrate/phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z` |
| FFT spectral | Dense trajectory spectrum | Granite-Small dense packet | packet | COMPLETE | low-frequency power `0.27456126536708364`; entropy `0.8530978298777753`; autocorr `100` tokens | 2026-05-18 | `experimental/outlier_migrate/phase9/spectral_analysis.md` |
| Per-component dissection | Quamba2 scope check | Granite, Nemotron, Falcon, DeepSeek | packets | COMPLETE | SSM/Mamba block outputs drift around `0.335-0.343`; true internal SSM tensors not isolated | 2026-05-18 | `experimental/outlier_migrate/phase9/per_component_dissection.md` |
| M11b | Budget scaling | Granite-Small | 12 | PASS | top5 median `0.4492840911245966`, CI `[-1.3009000187907436, 1.00079062654749]`; static-top10 `-0.05176717655558488` | 2026-05-18 | `experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z` |
| M12 | Hysteresis | none | 0 | NOT_STARTED | demoted after M11; may be revisited only if mechanism value rises | 2026-05 | `swarm/goal.md` |
| M17 | Afterglow | none | 0 | SKIPPED | skipped after M11 kill and M17 skip authorization | 2026-05 | `swarm/vacation_decisions/20260515T213500_m17_skip_authorization.md` |
| M26 | Stable core | none yet | 0 | NOT_STARTED | queued after M11b; requires stable-core size check first | 2026-05 | planned `experimental/outlier_migrate/phase9/preregister_om_phase9_m26_stable_core.md` |
| M27 | Layer-stratified | none yet | 0 | NOT_STARTED | queued after M26 | 2026-05 | planned `experimental/outlier_migrate/phase9/preregister_om_phase9_m27_layer_stratified.md` |
| ParoQuant | SOTA W4A16 baseline | none yet | 0 | NOT_STARTED | non-negotiable baseline, Granite then DeepSeek | 2026-05 | planned `experimental/outlier_migrate/phase9/preregister_om_paroquant_baseline.md` |
| KL accumulation | SLQ engagement | none yet | 0 | NOT_STARTED | load-bearing for compound-error and M31 decision | 2026-05 | planned `experimental/outlier_migrate/phase9/preregister_om_kl_accumulation.md` |

## 3. Exact Empirical Findings

- Four-model strict set-leaving: Granite-Small `0.566234756098`; Nemotron-3-Nano `0.533713200379867`; DeepSeek-R1-Distill-Qwen-1.5B `0.6705729166666666`; Falcon-H1 `0.6736111111111112`.
- Original migration fractions: Granite-Tiny `0.8178385416666667`; Granite-Small `0.843165650406504`; Nemotron `0.820809591642925`; DeepSeek `0.8393787202380952`; Falcon-H1 `0.7833543771043772`.
- M2: median recovery `-0.8668373133910525`, CI `[-3.4352892513350524, 0.5952386657662287]`; random-bin beat M2 by `0.6675482901676153`.
- M10: median `0.2344479278340464`, CI `[-2.072905653258708, 0.5035214944499808]`; random-bin beat M10 by `0.7614874590457608`.
- M11: best alpha `m11_alpha_0_5`; median `0.048299284137681975`, CI `[-8.94115615101423, 0.7045992558730417]`; random-walk median `-1.2061055483492975`.
- M18: activation+K median `-0.34359084412632024`, CI `[-14.071191710978855, 0.7406021242797479]`; KIVI key-only `-2.1741283113927237`; random coupled `-4.6643797310509125`.
- DecDEC: median `-0.07003478125820645`, CI `[-8.495693501232758, 0.6741763703501289]`; static-top10 `-0.22827336820997218`; random-reactive `-1.2626049261250778`.
- M11b: top1 median `-0.29418059401495783`; top5 `0.4492840911245966`; top10 `0.24075640427266048`; static-top10 `-0.05176717655558488`. Top5 beats static-top10 by `0.5010512676801815`, but top5 CI is wide and negative at the lower bound.
- FFT spectral analysis: only Granite dense M11 packet is identifiable. Median low-frequency power first 10% bins `0.27456126536708364`; normalized spectral entropy `0.8530978298777753`; autocorrelation length `100` tokens. Four-model FFT is not identifiable from sparse six-position packets.
- Per-component dissection: Granite Mamba block-output mean set-leaving `0.335365853659`; Granite attention `0.341463414634`; Nemotron Mamba `0.342995169082`; Nemotron attention `0.358024691358`; Nemotron MoE `0.338164251208`; Falcon post-sum residual `0.320707070707`; DeepSeek Transformer post-block output `0.361607142857`.
- Always-protected core mean fraction of layer top-1% count: Granite `0.516463414634`; Nemotron `0.616809116809`; DeepSeek `0.508928571429`; Falcon `0.570707070707`. This is a stable core, not a full solution.
- Post-M18 limitations: K/V set-leaving is not identifiable; per-position recovery curves are not identifiable because method packets score one endpoint window.

## 4. Three-Mechanism Framework Status

Mechanism 1, boundary discontinuities harmful: supported by M2 and M10. Both hard/discontinuous policies were beaten by random controls. This suggests selected boundary changes can actively harm quality rather than merely fail to help. A definitive test would compare hard-bin, smooth-bin, and random-bin policies on identical telemetry with the same protected budget.

Mechanism 2, smoothness necessary but insufficient: M11 supports this only weakly. EMA smoothing was not beaten by random-walk, unlike M2/M10, but recovery was far below a reliable positive bar. M11b complicates the story: with a top-5 budget, the same alpha family mechanically passes by median. This suggests smoothness alone is insufficient, but smoothness plus more budget may be live.

Mechanism 3a, signal staleness: M18 and DecDEC are evidence against simple current-signal fixes. M18 coupling was less bad than K-only/random controls but still negative; DecDEC was less bad than random reactive selection but negative. A decisive test would measure whether future top channels can be predicted above oracle-independent baselines from current state.

Mechanism 3b, budget insufficiency: M11b is the strongest support so far. Top5 beats top1, top10, and static-top10 by median, although uncertainty remains wide and the pattern is non-monotonic. Replication on Nemotron or DeepSeek would decide whether budget is a real mechanism or a Granite-slice artifact.

Mechanism 3c, compound error: still unresolved. Phase 3/4 no-gap behavior and broad failures are consistent with compounding, but not diagnostic. The planned KL accumulation experiment is the first direct test and also determines whether M31 pulsed precision is worth running.

## 5. Scoop Audit Results

Central empirical claim is partially scooped. DecDEC (arXiv 2412.20185, OSDI 2025) already reports low recall of static outlier analysis against per-step oracle top-K over short horizons. The defensible novelty is four-axis differentiation: 20K-token horizon rather than 100 steps; reasoning LLMs rather than chat-tuned Transformers; hybrid Mamba-2, parallel hybrid, and pure Transformer rather than Transformer-only; and AIME/MATH/GPQA-style reasoning traces rather than WikiText/BBH.

The headline hook is now the Quamba2 contrast. Quamba2 argues channel persistence/order preservation for SSM activations. Our defensible statement is scoped: at the block-output level, SSM/Mamba-class blocks in Granite and Nemotron drift at rates comparable to attention blocks in long-decode W4A16 packets. We do not yet isolate internal SSM state tensors.

Required terminology disambiguations: "outlier migration" in this project means decode-position channel-set drift; SmoothQuant uses migration for activation-to-weight difficulty transfer; MoBiQuant uses it for precision-dependent token sensitivity shifts. "Hot channels" in HCP/CHON refers to persistent pretraining-time NVFP4 channels, not our drifting inference-time W4A16 channels. "TTQ" must be disambiguated as test-time TTQ for arXiv 2603.19296 versus classical Trained Ternary Quantization.

Tested adjacent methods: M2/M10/M11/M18/DecDEC/M11b. Explicitly dropped or demoted because of scoop or low information value: M13, M14, M16, M19-M24; M15 and M25 only as deferred channel-axis variants; M12 demoted after M11; M17 skipped to prioritize M18. Required citations still pending in paper or already partially present include DecDEC, Quamba2, ParoQuant, SLQ, KIVI, PM-KVQ, PMPD, HCP/CHON, ChanMix, MixKVQ, LAQuant, QEP, and Activation Sensitivity as a Unifying Principle.

## 6. Current Paper Draft State

Paper path: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`. Current title: **Decode-Position Channel Drift in Long Reasoning Traces**.

Current abstract, verbatim from the draft:

> Quamba2 argues that SSM activations exhibit channel-order preservation and activation persistence, properties that support offline per-channel quantization for Mamba-family models. We test the corresponding assumption at reasoning-scale decode horizons by measuring decode-position channel-set drift: whether channels in the top 1% by magnitude early in decode remain in the same protected set later. Granite-4.0 hybrids show large drift under frozen gates: the original rank-drift metric is 0.8178385416666667 on Granite-4.0-H-Tiny and 0.843165650406504 on Granite-4.0-H-Small. The phenomenon also appears outside the Granite family: Nemotron-3-Nano gives 0.820809591642925, DeepSeek-R1-Distill-Qwen-1.5B gives 0.8393787202380952, and Falcon-H1 gives 0.7833543771043772. A stricter set-membership decomposition shows that the systems-relevant component is not only within-set rank shuffling: strict set-leaving is 0.566234756098 on Granite-Small, 0.533713200380 on Nemotron-3, 0.670572916667 on DeepSeek, and 0.673611111111 on Falcon-H1. We then test static, position-conditioned, scale-table, smoothed, and cross-tensor W4A16 protection interventions. All fail their preregistered gates, but with different failure modes: M2 and M10 are beaten by random-bin controls, M11 is not beaten by its random-walk control yet recovers only 0.048299284138 median of the BF16-vs-static gap, and M18 activation+K coupling beats K-only and random-coupled controls but still has negative median recovery (-0.343590844126). The DecDEC algorithmic reactive baseline is less damaging than a random reactive control but also has negative median recovery (-0.070034781258). The resulting paper is a measurement and mechanism paper, not a positive quantization method: boundary discontinuities are actively harmful, smoothing removes that harm but does not recover quality, and cross-tensor coupling has diagnostic signal without reliable recovery. The remaining mechanism question is whether future methods need larger protected budgets, stable-core targeting, or a way to address long-horizon compound error.

Section structure: Introduction motivates Quamba2 contradiction and cost; Related Work handles Mamba quantization, static protection, DecDEC, terminology; Methods define drift/decomposition/interventions; Results report cross-model measurements; Method Outcomes report failed/ambiguous methods; Discussion/Limitations scope W4A16 and small/hybrid models.

Integrated since scoop audit: Quamba2 headline, DecDEC differentiation, terminology footnote, industrial motivation, M2/M10/M11/M18/DecDEC, spectral/per-component diagnostics. Pending: M11b update, M26/M27/ParoQuant/KL, final committee rounds, updated abstract after M11b.

Committee trajectory: 2026-05-15 Draft 0 scores COLM 7, MLSys 6, adversarial 7. 2026-05-17 post-M18 scores COLM 7, MLSys 6, ICLR 5, adversarial 7. Main gap remains no replicated positive method.

## 7. Vacation-Mode Decisions Made Autonomously

Major decisions logged in `swarm/vacation_decisions/`:

| Decision | Context | Choice | Rationale |
|---|---|---|---|
| M2 OOM adaptation | Full M2 memory pressure | 12-trace vacation slice | Land interpretable result rather than stall |
| M2 cache cleanup | Memory cleanup needed | Reuse completed caches | Avoid rerunning expensive valid scores |
| M2 finalization | Runner incomplete but scores done | Finalize from cache | Preserved prereg metrics |
| M10 trace count | Runtime pressure | 12 traces | Vacation-mode V4 |
| M11 final snapshot | Need stable snapshot | Use existing dense packet | Avoid duplicating telemetry |
| M17 skip | M11 smoothness result landed | Skip M17, prioritize M18 | M18 tests different mechanism |
| Large artifact limit | GitHub rejected huge cache | Do not track large cache | Push audit files without 423 MB activation gz |
| M18 rule | Cross-tensor decision ambiguity | Use standard checker | Avoid post-hoc threshold changes |
| M11b alpha choice | Need budget test | alpha `0.3` | Best prior M11 sweep setting |
| DecDEC checker serialization | Checker write bug | Patch `exist_ok=True` style issue | No metric change |
| M11b reuse activation | Avoid duplicate 400 MB capture | Reuse M11 activation/traces | Same model/trace telemetry |

## 8. What's Currently In Flight or Queued

Nothing is currently running on GPU. The last landed run is M11b. Current state entry is `external_collaboration_state_export`; this document is the active task.

Next queue, if the human wants to continue the current plan:

1. M26 stable-core size measurement, then M26 prereg/run if core size is suitable.
2. M27 layer-stratified protection after M26.
3. ParoQuant baseline on Granite-Small and DeepSeek-R1-Distill-Qwen-1.5B, optionally 8B if VRAM permits.
4. KL accumulation experiment for SLQ comparison and M31 gating.
5. Conditional M31 if KL decay supports pulsed precision; conditional M33 if M26 is partial.
6. Paper integration and committee rounds.
7. Final comprehensive synthesis document.

Infrastructure risks: GitHub cannot accept single files over 100 MB without LFS. The M11b local packet includes `activation_magnitudes.jsonl.gz` at 423 MB, omitted from the pushed commit. Qwen3.6 per-layer hooks remain blocked in SGLang without source modification.

## 9. Open Questions and Branching Points

- Should the paper pivot immediately to M11b replication, since M11b is the first mechanical PASS, or finish M26/M27/ParoQuant/KL first?
- If M11b does not replicate, is the strongest story the negative mechanism paper: boundary discontinuities harmful, smoothing insufficient, reactive/cross-tensor selection insufficient?
- Should ThoughtFlow-FP8 be integrated as format-contrast evidence, or kept separate to avoid confounding model/task/format differences?
- Should LatentWire be discussed only in synthesis, or does latent-transfer work suggest a non-precision intervention point?
- Is the Quamba2 contradiction strong enough at block-output level, or should a future packet hook internal SSM tensors before making that a headline claim?
- Should training-process correlates be mentioned in the workshop paper? The argument for inclusion is honesty about N=4 variation; the argument against is high confounding.
- If the human wants ICLR rather than COLM workshop, the next high-value work is likely replication plus ParoQuant/SLQ baselines, not another speculative method.

## 10. Methods Considered But Not Run

Dropped or not authorized in the current sprint: M13, M14, M16, M19, M20, M21, M22, M23, M24 due to scoop or low expected value in the audit. M15 probe-then-protect and M25 attention-weighted channel importance are medium-scoop/deferred. M12 hysteresis is demoted because M11 already tested a smoothness variant, though it remains possible future triangulation. M17 afterglow was authorized then skipped after M11 to prioritize M18. M30 spaced rehearsal is deprioritized because synthetic checks suggested it may protect recently-departed rather than currently important channels. M31 pulsed full-precision protection is conditional on KL decay. M33 stable-core plus periodic recalibration is conditional on M26 partial signal.

ICLR follow-up candidates to list but not run now: streaming-PCA online subspace tracking, spectral predictors, robust-PCA online decomposition, Kalman per-channel scale predictors, wavelet decode-time multi-resolution quantization, Krylov-adaptive online rotation, and information-theoretic bit reallocation. Also deferred: M25 attention-weighted channel importance and M28 long-decode calibration redesign. Non-authorized methods include method cocktails of three or more failed ideas, sigma-delta/error-feedback quantization, dithered quantization, sparse autoencoder protection, robust H-infinity protection, and random projection/reservoir methods.

ThoughtFlow status for collaborators: the original sparse-cache positive-method branch is stopped. It has useful diagnostic/falsification artifacts, including an initial RDU first-gate pass that later failed broader reproduction. It should not be re-proposed as a live positive method without a fresh preregistration and a stronger evaluation surface.

LatentWire status for collaborators: the repo contains prior COLM artifacts and review packets under `paper/`, `colm/`, and `colm_final/`. This export did not rerun LatentWire. The connection to OutlierMigrate is conceptual: latent-transfer or compression-side interventions may offer alternatives to per-channel precision protection, but this remains an ideation path rather than landed evidence.
