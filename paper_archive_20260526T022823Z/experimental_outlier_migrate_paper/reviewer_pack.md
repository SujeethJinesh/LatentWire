# Decode-Position Channel Drift Reviewer Pack

- status: measurement/mechanism paper draft; Granite Phase 0/1 passed, partial
  Nemotron-3 passed, DeepSeek Transformer control passed, Falcon-H1
  within-lineage replication passed; Phase 3, Phase 4, and Phase 9 M2
  intervention attempts killed; not camera-ready final
- current decision: `KILL_M2_RANDOM_CONTROL_BEATS`
- current paper readiness: viable COLM workshop characterization/mechanism
  draft after Draft 0 integration; not a positive-method or systems paper
  because no tested protection method has survived matched controls

## Paper Link

- Draft TeX: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- Draft PDF: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

## Current Claim

The draft now supports this characterization claim: in long reasoning traces,
the identity of top-magnitude channels changes substantially with decode
position across multiple measured architectures. The systems-relevant
component is strict set-leaving: channels that are in the early top-1% set
often leave that set later, rather than merely shuffling rank inside it.

The method claim is negative. Three W4A16 protection attempts have failed:
static union protection on Granite-Tiny, static union protection on
Granite-Small with a longer scoring window, and Phase 9 M2
position-conditioned set switching. M2 is especially important because the
random-bin negative control beats the intended position-conditioned assignment
by the preregistered kill margin.

This is not full cross-validation and not evidence that a channel-drift-aware
intervention improves quality, latency, memory, or robustness. Qwen3.6 and
Kimi Linear remain deferred. The paper should be framed as a measurement and
mechanism paper with honest failed-method evidence, not as a positive method.

Important novelty boundary: do not claim this is the first dynamic-outlier
finding in Mamba. QMamba and OuroMamba already document dynamic hidden-state
or activation-outlier behavior in vision Mamba settings. The new contribution
is the long-decode language-model measurement, cross-architecture
decomposition, and negative evidence about simple static/position-binned
protected-channel methods.

The strict set-leaving decomposition is post-hoc interpretability for static
channel-protection relevance. It is not the original preregistered gate metric;
checker decisions remain based on their frozen packet-specific decision rules.

## Claimed / Not Claimed

Claimed:

- decode-position channel drift under preregistered Phase 0/1 metrics on two
  Granite hybrid model sizes;
- strict set-leaving is a large component of the measured drift signal;
- partial Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1 results reduce
  Granite-family-specific risk;
- static union protection and simple position-conditioned protected-set
  switching fail under their frozen gates.

Not claimed:

- no positive systems method;
- no deployable quantization recipe;
- no latency, throughput, memory, or quality improvement;
- no completed Qwen3.6/Kimi validation;
- no architecture-general claim beyond the measured packets.

## Strongest Evidence

| Gate item | Phase 0 exact value | Phase 1 exact value | Partial Phase 2 Nemotron-3 exact value |
|---|---:|---:|---:|
| Checker decision | `PASS_OM_PHASE0_DECODE_TIME_MIGRATION` | `PASS_OM_PHASE1_REPLICATED_AT_SCALE` | `PARTIAL_PASS_OM_PHASE2_NEMOTRON3_ONLY_QWEN36_KIMI_DEFERRED` |
| Strict set-leaving fraction | 0.634244791667 | 0.566234756098 | 0.533713200380 |
| Strict set-leaving 95% CI | [0.605208333333, 0.664192708333] | [0.550076219512, 0.581707317073] | [0.467414529915, 0.599982193732] |
| Within-set rank-shuffling fraction | 0.175260416667 | 0.270934959350 | 0.269082383666 |
| Within-set rank-shuffling 95% CI | [0.160156250000, 0.191015625000] | [0.260797764228, 0.280614837398] | [0.238841405508, 0.299056267806] |
| Original preregistered migration fraction | 0.8178385416666667 | 0.843165650406504 | 0.820809591642925 |
| Original migration bootstrap 95% CI | [0.797265625, 0.8368489583333334] | [0.8334349593495936, 0.8511432926829268] | [0.7865325261158594, 0.8544931149097815] |
| Preregistered dynamic threshold | point >= 0.05 and CI lower > 0.05 | point >= 0.05 and CI lower > 0.05 | point >= 0.05 and CI lower > 0.05 |
| Artifact complete | true | true | true |
| Model | `ibm-granite/granite-4.0-h-tiny` | `ibm-granite/granite-4.0-h-small` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Model commit | `791e0d3d28c86e106c9b6e0b4cecdee0375b6124` | `b8c0982bab7fde4eb48110f5a069527c008fab39` | `cbd3fa9f933d55ef16a84236559f4ee2a0526848` |
| Prompt SHA | `sha256:2f27c54baa8448e033d6e82f53f775dc6abe38188e4f1e5c0b97e3c74fe7c1dd` | `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e` | `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e` |
| Activation artifact SHA | `sha256:2783aa329d82e8f43ea4f342e52f0a25fc283b0b8971880f2da093739474d687` | `sha256:326a04351e0dda28cf919fe4745d7d9341011db13d940a06cf9c8470633003e1` | `sha256:bc841b938a125f05649f84577c0ddb7ab287213a683892cb81ea6ca06296206d` |
| Trace count | 12 | 24 | 24 |
| Layer count | 40 | 40 | 52 |
| Decode positions | 100, 500, 1000, 5000, 10000 | 100, 500, 1000, 5000, 10000, 20000 | 100, 500, 1000, 5000, 10000, 20000 |
| Bootstrap seed | 20260508 | 20260508 | 20260508 |

## Phase 3 Intervention Result

| Item | Exact value |
|---|---:|
| Checker decision | `KILL_OM_PHASE3_INTERVENTION_FAILS` |
| Artifact complete | true |
| Primary union median recovery | 0.000000000000 |
| Primary union 95% CI | [0.000000000000, 0.711143244199] |
| Pass rule | median recovery >= 0.50 and CI lower > 0.30 |
| Kill rule | recovery < 0.20 OR CI upper < 0.30 |
| Kill reason | median recovery 0.000000000000 < 0.20 |
| Static-2% matched-budget median recovery | 0.000000000000 |
| Static-2% 95% CI | [0.000000000000, 0.356017001423] |
| Magnitude-average median recovery | 0.061276118929 |
| Magnitude-average 95% CI | [0.000000000000, 0.512245438462] |
| Best control minus union | 0.061276118929 |
| Control stop condition | false; margin below 0.10 |
| Sparse grid median recovery | 0.000000000000 |
| Dense grid median recovery | 0.000000000000 |
| No-recoverable-static-gap traces | 10 / 24 |
| No-recoverable-static-gap fraction | 0.416666666667 |
| No-gap kill rule | >25% no-recoverable-static-gap traces |
| Union traces with positive recovery | 11 / 24 |
| Union traces with zero recovery | 10 / 24 |
| Union traces with negative recovery | 3 / 24 |
| Union traces with recovery >1 | 4 / 24 |

Recovery can be negative when a regime is worse than static-1%, greater than
one when it beats BF16 on the scored window, and zero when the static-1% regime
has no recoverable gap versus BF16. The paper now states this explicitly.
Median recovery was preregistered because the ratio can become very large when
the static gap is small; mean recovery is diagnostic only, not the gate
statistic.

## Phase 4 Intervention Result

| Item | Exact value |
|---|---:|
| Checker decision | `KILL_OM_PHASE4_INTERVENTION_FAILS` |
| Artifact complete | true |
| Primary union median recovery | 0.000000000000 |
| Primary union 95% CI | [0.000000000000, 0.069540641955] |
| No-recoverable-static-gap traces | 9 / 24 |
| No-recoverable-static-gap fraction | 0.375000000000 |
| Static-2% matched-budget median recovery | 0.000000000000 |
| Magnitude-average median recovery | 0.000000000000 |
| Control stop condition | false |

Interpretation: Phase 4 rules out the simplest explanation that Phase 3 failed
only because Granite-Tiny was too robust or because the 64-token scoring
window was too short. The larger Granite-Small model and 512-token window
still did not produce a positive static-union method.

## Phase 5'/7 and Phase 9 Premise Evidence

| Packet | Decision | Original drift | Strict set-leaving | Within-set shuffling |
|---|---|---:|---:|---:|
| DeepSeek-R1-Distill Phase 5' | `DYNAMIC_REGIME_TRANSFORMER` | 0.839378720238 | 0.670572916667 | 0.165736607143 |
| Falcon-H1 Phase 7 | `WITHIN_LINEAGE_2_CONSISTENT` | 0.783354377104 | 0.673611111111 | 0.097537878788 |
| Phase 9 Step 9.0 Granite-Small | `PASS_PHASE9_PREMISE_CHECK` | 0.836763211382 | 0.566234756098 | 0.270934959350 |
| Phase 9 Step 9.0 Nemotron-3 | `PASS_PHASE9_PREMISE_CHECK` | 0.801667853751 | 0.533713200380 | 0.269082383666 |
| Phase 9 Step 9.0 DeepSeek | `PASS_PHASE9_PREMISE_CHECK` | 0.834914434524 | 0.670572916667 | 0.165736607143 |

Phase 9 Step 9.0 also reports overall adjacent-bin top-1% Jaccard overlap
0.745999647119. That justified testing position-binned M2, but it did not
make M2 pass.

## Phase 9 M2 Result

| Item | Exact value |
|---|---:|
| Checker decision | `KILL_M2_RANDOM_CONTROL_BEATS` |
| Artifact complete | true |
| Vacation trace count | 12 |
| Included positive-static-gap traces | 8 |
| No-recoverable-static-gap traces | 4 / 12 |
| M2 median recovery | -0.866837313391 |
| M2 95% CI | [-3.435289251335, 0.595238665766] |
| Static-3% matched-cost median recovery | -0.625927638959 |
| Random-bin median recovery | -0.199289023223 |
| M2 minus static-3% median | -0.240909674432 |
| M2 minus random-bin median | -0.667548290168 |
| Kill reason | random-bin control beat M2 by >0.10 |

Interpretation: M2 is not a positive method on Granite-Small. The random-bin
control is also bad in absolute terms, but it is less bad than the intended
position-conditioned assignment, so the position-bin switching rule fails its
negative-control test.

## Artifact Paths

- Phase 0 preregistration: `experimental/outlier_migrate/phase0/preregister_om_phase0.md`
- Phase 0 result packet: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`
- Phase 0 checker output: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/checker_result.json`
- Phase 0 metrics: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/metrics.json`
- Phase 0 artifact completeness: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/artifact_check.json`
- Phase 0 migration decomposition: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/migration_decomposition.md`
- Phase 0 model provenance: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/model_provenance.json`
- Phase 0 prompt manifest: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/prompt_manifest.json`
- Phase 0 activation manifest: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/activation_magnitude_manifest.json`
- Phase 0 activation artifact: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/activation_magnitudes.jsonl.gz`
- Phase 1 preregistration: `experimental/outlier_migrate/phase1/preregister_om_phase1.md`
- Phase 1 result packet: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`
- Phase 1 checker output: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/checker_result.json`
- Phase 1 metrics: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/metrics.json`
- Phase 1 artifact completeness: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/artifact_check.json`
- Phase 1 migration decomposition: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/migration_decomposition.md`
- Phase 1 model provenance: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/model_provenance.json`
- Phase 1 prompt manifest: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/prompt_manifest.json`
- Phase 1 activation manifest: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitude_manifest.json`
- Phase 1 activation artifact: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitudes.jsonl.gz`
- Partial Phase 2 result packet: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`
- Partial Phase 2 checker output: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/checker_result.json`
- Partial Phase 2 metrics: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/metrics.json`
  (artifact-checked OutlierMigrate metrics file; this runner does not emit a
  separate `profiler_metrics.json`)
- Partial Phase 2 artifact completeness: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/artifact_check.json`
- Partial Phase 2 migration decomposition: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/migration_decomposition.md`
- Partial Phase 2 model provenance: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/model_provenance.json`
- Partial Phase 2 activation manifest: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/activation_magnitude_manifest.json`
- Partial Phase 2 activation artifact: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/activation_magnitudes.jsonl.gz`
- Phase 3 preregistration: `experimental/outlier_migrate/phase3/preregister_om_phase3_intervention.md`
- Phase 3 result packet: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`
- Phase 3 checker output: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/checker_result.json`
- Phase 3 metrics: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/metrics.json`
- Phase 3 per-trace metrics: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/per_trace_metrics.json`
- Phase 3 controls: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/control_metrics.json`
- Phase 3 grid sensitivity: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/grid_sensitivity.md`
- Phase 3 artifact completeness: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z/artifact_check.json`
- Phase 3 diagnostic: `experimental/outlier_migrate/phase3/diagnostic.md`
- Layer-stratified analysis: `experimental/outlier_migrate/phase3/results/layer_stratified_migration.md`
- Layer-stratified figure source: `experimental/outlier_migrate/phase3/results/layer_stratified_migration.json`
- Phase 4 result packet: `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z`
- Phase 4 checker output: `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z/checker_result.json`
- Phase 5' result packet: `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z`
- Phase 5' checker output: `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z/checker_result.json`
- Experiment D/E analysis directory: `experimental/outlier_migrate/decomposition_analysis`
- Phase 7 result packet: `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z`
- Phase 7 checker output: `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z/checker_result.json`
- Phase 9 Step 9.0 decomposition: `experimental/outlier_migrate/phase9/step9_0_decomposition_replication.md`
- Phase 9 Step 9.0 bin overlap: `experimental/outlier_migrate/phase9/step9_0_bin_overlap_analysis.md`
- Phase 9 M2 result packet: `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z`
- Phase 9 M2 checker output: `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z/checker_result.json`
- Phase 9 M2 diagnostic: `experimental/outlier_migrate/phase9/results/om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z/diagnostic.md`

## Related-Work Sources Added

- QMamba primary source: `https://arxiv.org/abs/2501.13624`
- OuroMamba primary source: `https://arxiv.org/abs/2503.10959`
- Quamba primary source: `https://arxiv.org/abs/2410.13229`
- Quamba-SE primary source: `https://arxiv.org/abs/2601.09451`
- Mamba-PTQ primary source: `https://arxiv.org/abs/2407.12397`
- MambaQuant primary source: `https://arxiv.org/abs/2501.13484`
- SmoothQuant primary source: `https://arxiv.org/abs/2211.10438`
- AWQ primary source: `https://arxiv.org/abs/2306.00978`
- QuaRot primary source: `https://arxiv.org/abs/2404.00456`
- KVQuant primary source: `https://arxiv.org/abs/2401.18079`
- BlockDialect primary source: `https://arxiv.org/abs/2501.01144`
- Kimi Linear primary source: `https://arxiv.org/abs/2510.26692`
- Qwen3.6 official model card: `https://huggingface.co/Qwen/Qwen3.6-35B-A3B`
- Gated Delta Networks primary source: `https://arxiv.org/abs/2412.06464`
- Gated Linear Attention primary source: `https://arxiv.org/abs/2312.06635`

Use these sources only for theoretical motivation and architectural context.
Do not claim that Kimi Linear, Qwen3.6, GLA, RWKV-7, or any other unmeasured
language model empirically exhibits OutlierMigrate migration until that model
is measured.

## Citation Spot-Check

Spot-checked on 2026-05-10 against primary sources:

- Kimi Linear resolves to arXiv `2510.26692`, title "Kimi Linear: An
  Expressive, Efficient Attention Architecture".
- Qwen3.6 resolves to the Hugging Face model card
  `Qwen/Qwen3.6-35B-A3B`; the model card describes a Gated DeltaNet / gated
  attention / MoE hybrid layout.
- Quamba-SE resolves to arXiv `2601.09451`, title "Late Breaking Results:
  Quamba-SE: Soft-edge Quantizer for Activations in State Space Models".

This is a spot-check, not a complete citation audit. Human final review should
still verify every bibliography entry before submission.

## Reviewer Risks

- Phase 0 and Phase 1 are same-family Granite measurement evidence; Phase 2 is
  a partial Nemotron-3 check, not completed Qwen3.6/Kimi cross-validation.
- Do not describe the packet as camera-ready, full cross-validation, a
  positive systems method, or a validated positive-method branch.
- Do not describe union-set protection, Granite-Small union protection, or M2
  position-bin switching as successful. They killed with
  `KILL_OM_PHASE3_INTERVENTION_FAILS`,
  `KILL_OM_PHASE4_INTERVENTION_FAILS`, and
  `KILL_M2_RANDOM_CONTROL_BEATS`.
- No systems-method claim remains: no latency, throughput, memory, quality,
  or deployable quantization recipe is validated.
- The paper does not yet include a contamination audit or independent seed
  repeat beyond the recorded bootstrap procedure.
- The paper does not yet include a complete exploratory-history audit for the
  top-1% fraction, rank-delta threshold, decode positions, prompt slice, or
  model choices.
- The result says decode-position top-channel rank dynamics are real on the
  measured packets; the attempted interventions did not improve the
  preregistered recovery metrics robustly.
- Dynamic outliers are not novel to Mamba broadly; QMamba and OuroMamba are
  prior evidence in vision Mamba. The paper should use the broader
  decode-position channel drift frame rather than a Mamba-specific title.
- Static-protection systems (BlockDialect, AWQ, SmoothQuant, QuaRot, KVQuant)
  are deployment motivation, not defeated baselines. The paper can say they
  motivate validation of static maps on hybrid decode, not that they fail.
- The mechanism discussion is theoretical. Channel-wise gating, KDA,
  Qwen3.6-style Gated DeltaNet, and GLA-style mechanisms plausibly explain why
  top-magnitude channels can change over decode, but no unmeasured model should
  be described as empirically positive.
- Do not tune positions, top-channel fraction, rank-delta threshold, or model
  selection on these packets; the Phase 0 and Phase 1 surfaces were
  preregistered.

## Saturated / Alive / Next Branch

- saturated: Phase 0, Phase 1, Phase 2 partial Nemotron-3, Phase 5',
  Experiment D/E, Phase 7, and Phase 9 Step 9.0 measurement surfaces are
  closed for the current draft.
- killed as methods: Phase 3 static union, Phase 4 Granite-Small union, and
  Phase 9 M2 position-bin switching.
- alive: COLM workshop characterization/mechanism paper on decode-position
  channel drift and strict set-leaving across measured architectures.
- promoted: the broad decode-position channel drift hypothesis beyond
  Granite-family-only or Mamba-specific framing.
- weakened: a fixed position-100 protected-map interpretation and the idea
  that simple static or position-binned protected sets are enough.
- not established: completed cross-model transfer, delta-rule linear-attention
  validation, RWKV-7/GLA generalization, or a positive intervention method.
- next exact gate: committee review of Draft 0, then M10 on Granite-Small if
  vacation-mode execution continues after the paper pass.
