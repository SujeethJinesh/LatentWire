# New 200 Positive-Method Triage

Date: 2026-06-04

Scope: ideation, filter, scoop, and queue emission only. No experiments, no confirm access, no GPU, and no registry cards for the full seed bank.

## Current State Ingested

- LatentWire one-way WZ / score packets, `L_SCORECOMP`, `L_B1`, `L_Q1`, and deterministic C2C/KVComm packet smokes are killed or bounded-negative as method evidence.
- EXP1 v3 corrected cache ceiling is `INCONCLUSIVE_UNDERPOWERED`; only ARC tested a sane hidden receiver baseline and showed `+0.029` with CI crossing zero.
- EXP4 v3 L-A2 is `PARKED_NEEDS_STRONGER_GENERATOR`: `100` prompts, `1600` candidates, verifier scores present, but only `36` prompts with any correct candidate.
- Channel-Set C_A1 is the first GPU backfill priority, but promotion is disabled until paired native ParoQuant-vs-tight-clip rows pass on Granite, DeepSeek, and Falcon.
- C-F-style survival core cannot queue without a fresh identical-row denominator because the prior branch was control-contaminated.

## Mandatory Gates

Communication survivor gate:

```text
I(source_signal; Y | receiver_state) >> 0.28 bits
```

The idea must state why the source holds information the receiver cannot recover from its own forward pass. Same-family, free, one-way score/cache packets fail this gate by lineage.

Quantization survivor gate:

```text
paired identical-row improvement on >=2 models including a sentinel,
with no median/tail regression and no no-gap denominator artifact
```

Rotation re-skins fail unless they introduce a drift-specific, cross-model-non-regressing mechanism.

## Funnel Counts

| stage | count | rule |
| --- | ---: | --- |
| seed ideas | 208 | 100 Channel-Set + 100 LatentWire/CacheWire + 8 elevated complementary/private-source supplements |
| survivors after lineage/headroom filter | 60 | plausible headroom gate and not directly killed/dominated |
| registry-candidate candidates | 20 | math/adversary/scoop survivors worth a card after human sign-off |
| immediate MPS-first queue | 8 | cheapest, most-discriminating probes; no experiments launched in this pass |

## 60 Survivors

| id | paper | mechanism_family | nearest killed/dominated relative | mandatory gate | hard baseline | scoop verdict | EVI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L-PC1 | latentwire | cross-family specialist source | same-family score/cache packets | conditional MI with specialist signal beyond receiver state | target-only + source-index/confidence + source logits | OPEN | 0.94 |
| L-PC2 | latentwire | tool-private computation | one-way score packet | tool-result/verifier signal must add beyond receiver state | text tool-result control + target-only | OPEN | 0.93 |
| L-PC3 | latentwire | private retrieval source | source-score packet | retrieved-evidence signal must add beyond receiver state | visible retrieved text at equal bytes + source-index | OPEN | 0.82 |
| L-PC4 | latentwire | extra-compute source | full source-only score fusion null | self-consistency/cluster signal must add beyond receiver state | same-budget target sampling + source confidence | OPEN | 0.91 |
| L-PC5 | latentwire | private verifier over receiver candidates | L-A2 generator-limited pool | verifier score must rank candidates where receiver has entropy | target verifier + candidate derangement | OPEN | 0.90 |
| L-PC6 | latentwire | trained residual fuser | free fusion under-recovers | trained fuser must beat free fusion under wrong-row controls | C2C/LCF anchor + zero-source | PARTIAL_HIGH_RISK | 0.76 |
| L-PC7 | latentwire | private code-test packet | source-copy packets | hidden test outcome must add beyond receiver state | visible test-count control | OPEN | 0.78 |
| L-PC8 | latentwire | private retrieval contradiction | source answer leak | contradiction bit must not identify answer directly | equal-byte visible contradiction tag | OPEN | 0.74 |
| L-G1 | latentwire | verifier margin packet | L-A2 weak generator | >=80 correct-candidate prompts; verifier CMI/gain beyond target | target_score/source_score/verifier baselines | OPEN_PARTIAL | 0.87 |
| L-G2 | latentwire | pairwise verifier tournament | candidate-delta controls | pairwise preferences must survive candidate derangement | Bradley-Terry target-only | OPEN_PARTIAL | 0.81 |
| L-G3 | latentwire | negative-evidence packet | answer-index packets | eliminates bad candidates without leaking correct ID | candidate-index and source-top1 controls | OPEN_PARTIAL | 0.72 |
| L-G5 | latentwire | solution-cluster packet | answer packet leak | cluster ID must not equal candidate ID | cluster-size and random-cluster controls | OPEN_PARTIAL | 0.70 |
| L-E1 | latentwire | 64-candidate math rerank | 4-way MCQ redundancy | high-entropy candidate pool CMI beyond target | target self-consistency rerank | OPEN_PARTIAL | 0.77 |
| L-E2 | latentwire | self-consistency cluster rerank | source confidence packet | private source cluster choice must add beyond receiver samples | same-budget receiver samples | OPEN_PARTIAL | 0.79 |
| L-X4 | latentwire | code-test outcome packet | answer leak packets | hidden tests add beyond receiver state | visible pass-count control | OPEN | 0.73 |
| L-X5 | latentwire | invariant-check packet | source-copy packet | invariant signal adds beyond receiver state | visible invariant text tag | OPEN | 0.66 |
| L-R1 | latentwire | receiver-query cache delta | L_Q1 failed query packet | query-conditioned cache delta must beat source-index/confidence and query-only/reply-only | C2C/LCF/KVComm anchor | PARTIAL_HIGH_RISK | 0.70 |
| L-R2 | latentwire | ambiguous-pair query | source-index packet | source margin over receiver top-2 must add conditional info | target top-2 margin | PARTIAL_HIGH_RISK | 0.64 |
| L-R4 | latentwire | entropy-gated packet | L_B1 trust packet | packet only on high target entropy; positive risk-coverage delta | UCCI-like router + target confidence | PARTIAL_HIGH_RISK | 0.57 |
| L-C2 | latentwire | control-trained LCF-lite | C2C/KVComm smokes | wrong-row/zero-source controls must collapse; byte accounting literal | C2C/LCF + zero-source | PARTIAL_HIGH_RISK | 0.69 |
| L-C5 | latentwire | cache-delta verifier rerank | cache trace control collapse | cache delta only improves candidate rerank, not generation leak | target verifier + LCF anchor | PARTIAL_HIGH_RISK | 0.63 |
| L-T2 | latentwire | LCF-teacher byte code | dense teacher not deployable | byte code distills only target-needed teacher effect | LCF/C2C teacher + target-only | PARTIAL_HIGH_RISK | 0.62 |
| L-T5 | latentwire | control-contrastive distill | teacher-delta ties controls | learned packet must fail on wrong-row/zero-source | wrong-row and zero-source planted controls | PARTIAL_HIGH_RISK | 0.61 |
| L-I2 | latentwire | interactive Slepian-Wolf | one-way WZ killed | receiver query changes source code and adds conditional info | one-way WZ + source-index | PARTIAL_HIGH_RISK | 0.58 |
| L-I5 | latentwire | info-bottleneck packet | source-copy packet | positive `I(Z;Y|R)-alpha I(Z;A_S)` | source answer/confidence controls | OPEN_PARTIAL | 0.55 |
| L-M2 | latentwire | diversity-weighted source | same-family redundancy | source decorrelation predicts positive conditional MI | source-index/confidence | OPEN_PARTIAL | 0.54 |
| L-M4 | latentwire | anti-correlated expert packet | weak source damage | source helps only target blind spots with harm gate | target-only + source-only | OPEN_PARTIAL | 0.53 |
| L-P1 | latentwire | proper-scoring packet | accuracy-only score packet | log-score/Brier gain beyond calibrated target | UCCI/calibrated router | PARTIAL_HIGH_RISK | 0.46 |
| L-P5 | latentwire | belief-shape codebook | label/source-index packet | belief shape adds beyond top1/margin | source score sketch | PARTIAL_HIGH_RISK | 0.44 |
| L-S1 | latentwire | wrong-row contrastive training | control-insensitive smokes | matched gain collapses under wrong-row | wrong-row/zero-source | OPEN_AS_DEFENSE | 0.51 |
| L-S2 | latentwire | candidate-derangement training | candidate-index leak | deranged candidates destroy gain | candidate permutation controls | OPEN_AS_DEFENSE | 0.49 |
| L-S5 | latentwire | query/reply orthogonality | L_Q1 ablation explained signal | query-only/reply-only both null | query-only/reply-only controls | OPEN_AS_DEFENSE | 0.48 |
| L-N1 | latentwire | Bradley-Terry delta code | source rank packet | pairwise utility delta adds beyond target pairwise | target BT model | OPEN_PARTIAL | 0.45 |
| L-N3 | latentwire | Kendall-syndrome code | candidate-index leak | ranking correction not candidate ID | target ranking + random syndrome | OPEN_PARTIAL | 0.43 |
| L-MI1 | latentwire | conditional-MI task router | all-source packet use | route only rows with measured conditional MI | target entropy/source confidence | OPEN_AS_ENABLER | 0.56 |
| L-MI3 | latentwire | complementarity score packet | source-score copy | complementarity feature orthogonal to target scores | source score sketch | OPEN_PARTIAL | 0.42 |
| L-Y4 | latentwire | byte-frontier profiler card | systems overclaim | literal byte/min-code/latency accounting | text and cache baselines | OPEN_AS_DEFENSE | 0.36 |
| C-T1 | channel_set | GPD-CVaR clip selector | tight-clip one-model retune | paired >=2 model tail win and no median regression | ParoQuant paired native | OPEN_PARTIAL | 0.88 |
| C-T2 | channel_set | model-local tail alpha | cross-model transfer trap | model-local alpha chosen on dev, gate non-regression | ParoQuant + static top-k | OPEN_PARTIAL | 0.67 |
| C-T3 | channel_set | no-gap-aware tail selector | no-gap denominator artifact | positive-gap-only tail win | CE21 no-gap filter | OPEN_AS_DEFENSE | 0.52 |
| C-T4 | channel_set | worst-trace repair budget | Granite-only tail rescue | sentinel worst-trace repair without median harm | paired ParoQuant | OPEN_PARTIAL | 0.63 |
| C-S1 | channel_set | survival StableCore clean | C-F random-control contamination | identical-row denominator beats random/static controls | static/random controls | OPEN_PARTIAL | 0.69 |
| C-S3 | channel_set | Cox warmup predictor | C-F contaminated | warmup hazard predicts held-out drift/uplift | static top-k + random hazard | OPEN_PARTIAL | 0.58 |
| C-C1 | channel_set | CUSUM drift trigger | adaptive overfit | trigger predicts KL/tail spike on held-out rows | static policy + random trigger | OPEN_PARTIAL | 0.50 |
| C-C4 | channel_set | Kalman drift-state selector | one-policy retune | latent drift state predicts policy choice across models | fixed-library warmup selector | OPEN_PARTIAL | 0.46 |
| C-D1 | channel_set | co-drift Givens pairing | rotation re-skin | co-drift pairing beats ParoQuant on >=2 models | ParoQuant/OSCAR | PARTIAL_HIGH_RISK | 0.49 |
| C-D2 | channel_set | channel graph communities | top-k drift overfit | graph communities predict stable protected set | static top-k | PARTIAL_HIGH_RISK | 0.44 |
| C-I1 | channel_set | rate-distortion protection | magnitude-only top-k | KL saved per bit predicts gate uplift | ParoQuant/ResQ | OPEN_PARTIAL | 0.48 |
| C-O1 | channel_set | submodular KL-gain greedy | unpowered selector | dev marginal KL gains transfer to gate | static top-k/ParoQuant | OPEN_PARTIAL | 0.43 |
| C-R2 | channel_set | fail-safe static fallback | one-model adaptive harm | reject adaptive policy when confidence low | static top-k | OPEN_AS_DEFENSE | 0.41 |
| C-P2 | channel_set | DMD drift modes | C-F static survival | drift modes predict future top-k membership | survival/static controls | OPEN_PARTIAL | 0.42 |
| C-B4 | channel_set | posterior no-regression gate | median regression | posterior `P(median harm)` below threshold | paired ParoQuant | OPEN_AS_DEFENSE | 0.39 |
| C-U1 | channel_set | treatment-effect trace router | DeepSeek/Falcon regression | predicted uplift positive on held-out rows and >=2 models | ParoQuant + target confidence | OPEN | 0.84 |
| C-U3 | channel_set | counterfactual horizon control | wrong-horizon artifact | wrong-horizon policy collapses | wrong-horizon control | OPEN_AS_DEFENSE | 0.37 |
| C-U5 | channel_set | mediation drift-to-loss audit | diagnostic-only drift | drift reduction mediates loss/tail improvement | CE21/OSC controls | OPEN_AS_DEFENSE | 0.36 |
| C-H2 | channel_set | HBM-cost-aware CVaR | quality-only overclaim | tail gain per HBM byte positive | profiler baseline | OPEN_AS_SYSTEMS | 0.34 |
| C-W1 | channel_set | fixed-library warmup selector | brittle single retune | selects ParoQuant/C_A1/reject on held-out gate | static ParoQuant + reject | OPEN | 0.76 |
| C-W3 | channel_set | warmup tail-risk classifier | post-hoc tail selection | warmup predicts catastrophic ParoQuant tail | random classifier | OPEN | 0.65 |
| C-K2 | channel_set | answer-length-conditioned policy | prompt overfit | length/horizon predicts policy benefit | fixed policy | OPEN_PARTIAL | 0.40 |
| C-K4 | channel_set | reasoning-phase selector | phase-unaware clip | scratchpad/final phases differ in drift/uplift | fixed clip | OPEN_PARTIAL | 0.38 |
| C-E1 | channel_set | KL-growth router | drift diagnostic only | KL-growth class routes static/adaptive | static policy | OPEN | 0.61 |
| C-E4 | channel_set | quantization-gap oracle router | no-gap artifact | only intervene where BF16/static gap recoverable | CE21 no-gap filter | OPEN_PARTIAL | 0.55 |
| C-M1 | channel_set | robust policy portfolio | one-policy fragility | minimax regret across models | best single policy | OPEN_PARTIAL | 0.41 |
| C-L3 | channel_set | trace-policy pairwise model | one-model selection | predicts which policy wins across models | fixed-library baseline | OPEN_PARTIAL | 0.40 |
| C-N2 | channel_set | leverage-score channel protection | magnitude-only top-k | leverage predicts tail protection value | static top-k/ResQ | PARTIAL_HIGH_RISK | 0.35 |
| C-Y1 | channel_set | measure-then-choose protocol | systems overclaim | calibration checklist prevents tail/median harm | current runbook | OPEN_AS_DEFENSE | 0.33 |
| C-Y3 | channel_set | tail-safe ParoQuant wrapper | C_A1 regression risk | fallback unless CVaR sentinels pass | ParoQuant native | OPEN_PARTIAL | 0.62 |
| C-Y5 | channel_set | reviewer-defense bundle | diagnostics-only | OSC/DecDEC/no-gap/parity automated pass/fail | named baselines | OPEN_AS_DEFENSE | 0.32 |

## Top 20 Registry-Candidate Candidates

These do not become registry cards in this pass. They are card candidates after human sign-off, math/adversary review, and pre-launch review.

| rank | id | contribution_role | host | why it survives | kill condition |
| ---: | --- | --- | --- | --- | --- |
| 1 | L-PC1 | positive_method | MPS_FIRST | Directly tests the untried complementary-source regime. | Conditional MI <= 0.28 bits or source signal reducible to source top1/confidence. |
| 2 | L-PC2 | positive_method | MPS_FIRST | Private tool computation is genuinely unavailable to receiver. | Tool/verifier signal does not beat visible/equal-byte controls. |
| 3 | L-PC4 | positive_method | MPS_FIRST | Extra compute creates a plausible private signal without GPU training. | Same-budget receiver samples explain the effect. |
| 4 | L-PC5 | positive_method | MPS_FIRST | Fixes L-A2 by verifying receiver candidates rather than relying on weak generator. | Fewer than 80 correct-candidate prompts or verifier CMI is null. |
| 5 | C-U1 | positive_method | MPS_FIRST | Fresh Channel-Set positive: drift predicts when to route/intervene. | Drift features fail to predict uplift beyond baseline confidence on held-out rows. |
| 6 | C-W1 | positive_enabler | MPS_FIRST | Converts brittle C_A1/ParoQuant choice into a reject-capable policy. | Selector does not beat best fixed policy or harms a sentinel. |
| 7 | C-T1 | positive_method | GPU_AFTER_MPS | Existing Channel-Set live bet; needs native paired replay. | Any model lacks paired rows, or DeepSeek/Falcon median/tail regresses. |
| 8 | L-G1 | positive_method | GPU_AFTER_MPS | Best LatentWire positive if stronger candidate cache exists. | Correct-candidate subset below 80 prompts or target verifier dominates. |
| 9 | L-C2 | positive_method | MPS_FIRST | Control-trained LCF-lite is the cheapest C2C/LCF proxy. | Wrong-row/zero-source controls do not collapse. |
| 10 | L-PC3 | positive_method | MPS_FIRST | Private retrieval creates source-only evidence not in receiver state. | Equal-byte visible retrieval tag explains the gain. |
| 11 | C-S1 | positive_method | MPS_FIRST | Survival core remains plausible only with a clean denominator. | Identical-row random/static controls match or beat it. |
| 12 | C-W3 | positive_enabler | MPS_FIRST | Cheap cached tail-risk classifier for C_A1 gating. | Warmup features do not predict catastrophic tail rows. |
| 13 | L-E2 | positive_method | MPS_FIRST | Self-consistency cluster transmission tests private compute entropy. | Receiver same-budget self-consistency matches the source cluster. |
| 14 | L-X4 | positive_method | MPS_FIRST | Hidden code-test syndrome has genuine private signal. | Visible pass-count or candidate ID controls explain effect. |
| 15 | C-E1 | positive_enabler | MPS_FIRST | KL-growth router is a drift-as-signal variant. | KL growth does not predict policy choice or tail risk. |
| 16 | C-Y3 | positive_enabler | GPU_AFTER_MPS | Tail-safe ParoQuant wrapper may be claimable as a safety method. | Fallback/reject policy does not improve CVaR without median harm. |
| 17 | L-R1 | positive_method | MPS_FIRST | Receiver-conditioned cache delta attacks the oracle gap. | Loses to source-index/confidence or is scooped by generic C2C/LCF framing. |
| 18 | L-T5 | positive_enabler | MPS_FIRST | Turns destructive controls into training objective. | Control-contrastive packet still works on wrong-row/zero-source. |
| 19 | C-T4 | positive_method | GPU_AFTER_MPS | Worst-trace repair is directly aligned with paper tail story. | Tail improvement is Granite-only or causes sentinel median harm. |
| 20 | C-Y5 | diagnostic_defense | MPS_FIRST | Required reviewer defense bundle for Channel-Set claims. | Missing any named baseline/control: OSC, DecDEC, CE21 no-gap, ParoQuant parity. |

## Killed Or Parked Blocks From the Original 200

- `L-O1` to `L-O5`: killed/parked. Same-family specialization and affine score bridges are too close to killed score/source-confidence mechanisms.
- `L-D1` to `L-D5`: mostly parked. Damage/trust routing repeats `L_B1` unless attached to private evidence and a new conditional-MI gate.
- `L-A1` to `L-A5`: parked unless private retrieval evidence is actually unavailable to the receiver; otherwise equal-byte visible retrieval/text baselines dominate.
- `L-H1` to `L-H5`: parked or killed. Text/control hybrids are high leak risk unless answer identity and source-top1 are provably unrecoverable.
- `L-F1`, `L-F2`, `L-F4`, `L-F5`: baseline/defense only; C2C/LCF adjacency is too direct for a new claim without byte-limited receiver conditioning.
- `C-G1` to `C-G5`: parked as rotation-space high-risk. Requires a passed orthogonality/full-precision-equivalence test and cross-model non-regression.
- `C-D3` to `C-D5`: parked; co-drift variants need a non-rotation-reskin mechanism and paired sentinels.
- `C-A1` to `C-A5` architecture-specific variants: parked unless exact architecture-local caches exist and no branch/MoE overclaim is made.
- `C-H1` to `C-H5` and most `C-Y*`: defense/systems cards unless quality improves at matched latency/byte cost.

## Immediate Queue Decision

Emit `queues/mps_first.yaml` with the eight cheapest high-discrimination probes:

1. `L_PC1_cross_family_specialist_ceiling`
2. `L_PC2_tool_augmented_source_ceiling`
3. `L_PC5_private_verifier_receiver_candidates`
4. `L_C2_control_trained_lcf_lite_proxy`
5. `C_U1_drift_as_signal_router`
6. `C_W1_fixed_library_warmup_selector`
7. `C_S1_clean_survival_stablecore_denominator`
8. `C_Y5_channel_set_defense_bundle`

Emit `queues/gpu_after_mps.yaml` for only the GPU-dependent branches that must wait for MPS evidence. `queues/gpu_foreground.yaml` remains empty, and C_A1 remains first in `queues/gpu_backfill.yaml`.
