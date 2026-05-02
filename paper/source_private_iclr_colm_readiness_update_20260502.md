# Source-Private ICLR/COLM Readiness Update

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets, public-benchmark
  ARC/OpenBookQA endpoints, and byte/exposure systems accounting form the
  defensible core. HellaSwag is no longer a current receiver-improvement
  headline; it is diagnostic/headroom and negative-ablation evidence.
- Exact gap: one positive source-family repair, learned receiver, or
  common-language connector must beat packet-only under paired uncertainty,
  destructive controls, and at least one strict cross-family or cross-benchmark
  falsification. The Mac-local Phi-3 cross-family source diagnostic and the
  TinyLlama hidden/query PCA/ridge, transport/common-basis, nonlinear
  sparse-query cache-bottleneck, and train-only MLP cache-to-packet connectors
  now fail this gate. The Llama-8B
  true non-Qwen scout now runs locally after the MPS workaround, but also fails
  the strict validation/paired-uncertainty gate. The systems boundary table is
  paper-ready as accounting and the native ingest gate now refuses premature
  claims, but native NVIDIA serving rows remain incomplete.

## Contributions To Put Forward

1. Fixed-byte source-private evidence packets with source-destroying controls.
   Current support: ARC-Challenge and OpenBookQA public-basis rows remain
   positive, and HellaSwag shows strong packet/headroom diagnostics.
2. Public-basis/common-coordinate packet methods for real benchmarks. Current
   support: ARC Fourier/anchor-syndrome packets and OpenBookQA 3B shared-basis
   packets survive seed repeats and same-byte text controls; strict
   source-family robustness is still negative.
3. Systems byte/exposure accounting for packet transfer versus KV/cache
   transfer. Current support: Mac-local decode/packet-ring traces and
   cross-benchmark byte-floor comparators; native vLLM/SGLang throughput rows
   are still pending.
4. A rigorous falsification ladder for latent/hidden-code communication.
   Current support: HellaSwag source-score, receiver-selector, discrete-query,
   anchor-relative common-basis, switch-observability, and PQ hidden-code gates
   are now explicitly recorded as negative or non-headline.

## What Changed

The systems boundary figure/table V3 is now materialized:

- artifact:
  `results/source_private_systems_boundary_figure_table_20260502/`
- files: `systems_boundary_figure_data.json`, `.csv`,
  `systems_boundary_table.md`, `systems_boundary_table.tex`,
  `systems_boundary_waterfall.svg`, and `manifest.json`;
- pass gate: `True`;
- packet rows: `4`;
- framed packet range: `4-15B`;
- minimum source-state floor: `768B`;
- minimum source-state floor versus largest packet: `51.2x`;
- native NVIDIA systems complete: `False`.

Lay explanation: this table compares what crosses the boundary. LatentWire
sends a tiny task hint; C2C/KVComm-style systems and KV quantizers move, store,
or compress internal model memory. The table is a systems accounting win, not a
GPU speed claim.

The 2026-05-02 evidence bundle now ingests the ARC source-family
packet-confidence router diagnostic:

- artifact:
  `results/source_private_arc_challenge_source_family_router_diagnostic_20260502/source_family_router_diagnostic.json`
- selected receiver-confidence metric: `best_score`;
- test router accuracy: `0.315` versus Qwen-substituted packet `0.317`;
- router minus Qwen-substituted mean/min: `-0.002/-0.008`;
- minimum paired CI95 low versus Qwen-substituted: `-0.023`;
- packet oracle on the same disagreement rows: `0.586`;
- pass gate: `False`.

Lay explanation: TinyLlama and Qwen sometimes send different tiny hints. The
diagnostic asked whether Qwen could tell which hint to trust by looking at
simple confidence signals. It could not, even though an oracle could choose the
better hint much more often.

The ARC source-side score router gate now tests the next obvious repair:

- artifact:
  `results/source_private_arc_challenge_source_score_router_gate_20260502/source_score_router_gate.json`
- selected validation rule: `source_index_pair_lookup`;
- validation router/Qwen: `0.451/0.389`, delta `+0.063`, CI95 low `+0.010`;
- frozen test router/Qwen/oracle: `0.315/0.317/0.586`;
- frozen test router-minus-Qwen mean/min: `-0.002/-0.002`;
- frozen test CI95 low versus Qwen: `-0.031`;
- best scalar source-confidence row ties or hurts Qwen-substituted packets;
- pass gate: `False`.

Lay explanation: this time the source models themselves attached a tiny
confidence signal before the router chose which packet to trust. The validation
rule looked promising, but it did not transfer to frozen test rows. That means
the failure is not just missing a simple source-confidence byte.

The ARC Qwen-1.5B stronger-source diagnostic adds a promising but bounded
positive result:

- artifact:
  `results/source_private_arc_challenge_source_family_cache_falsification_20260502_qwen15_cpu/source_family_cache_falsification.json`
- alternate source family: `qwen2.5_1.5b`;
- overall pass gate: `False` because validation Qwen-disagreement pass is
  `0/5`;
- frozen test full-slice pass: `5/5`, matched/target/text `0.442/0.265/0.401`;
- frozen test Qwen-disagreement pass: `5/5` on `388` rows;
- frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.482/0.184/0.456/0.296`;
- frozen test matched-minus-Qwen-substituted min: `+0.294`;
- frozen test CI95 low versus Qwen-substituted: `+0.216`.

Lay explanation: using a stronger Qwen-1.5B source instead of TinyLlama makes
the tiny ARC packet very useful on frozen test disagreements with Qwen-0.5B.
This is encouraging, but it is same-family and validation-gate-incomplete, so
it should be framed as source-strength evidence rather than the final
cross-family result.

The ARC Phi-3 cross-family source diagnostic tests the strict non-Qwen branch:

- artifact:
  `results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/source_family_cache_falsification.json`
- alternate source family: `phi3_mini_4k`;
- overall pass gate: `False`;
- validation full/Qwen-disagreement pass: `0/5` and `0/5`;
- frozen test full/Qwen-disagreement pass: `0/5` and `0/5`;
- frozen test full-slice matched/target/text: `0.244/0.265/0.241`;
- frozen test Qwen-disagreement rows: `833`;
- frozen test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.200/0.340/0.209/0.273`;
- frozen test matched-minus-Qwen-substituted min: `-0.143`;
- frozen test CI95 low versus Qwen-substituted: `-0.193`;
- Phi source-choice accuracy before packet: validation `0.274`, test `0.246`.

Lay explanation: this run asked a different model family, Phi-3, to send the
same tiny ARC hint. The hint did not help the Qwen receiver; on the rows where
Phi and Qwen disagreed, the Qwen-substituted packet was much stronger. This
rules out the available Mac-local Phi-3 source as the cross-family repair.

The ARC candidate-syndrome connector gate now tests the next learned cached
repair:

- artifact:
  `results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502/candidate_syndrome_connector_gate.json`
- selected primary view: `tiny_score_shape_connector`;
- frozen test selected-primary/Qwen/oracle: `0.288/0.317/0.586`;
- frozen test connector-minus-Qwen mean/min: `-0.029/-0.040`;
- frozen test CI95 low versus Qwen: `-0.091`;
- paired-family diagnostic test accuracy: `0.316` versus Qwen `0.317`;
- pass gate: `False`.

Lay explanation: instead of only choosing between the whole TinyLlama hint and
the whole Qwen hint, this experiment trained a small scorer that could pick any
answer option using the cached packet and source-score patterns. It still lost
to Qwen on frozen test rows, so the remaining headroom requires richer hidden
or query-level information rather than more cached score geometry.

The ARC hidden/query common-basis gate tests the richer Mac-local connector:

- artifact:
  `results/source_private_arc_challenge_hidden_query_common_basis_gate_20260502_tinyllama_disagreement/arc_challenge_hidden_query_common_basis_gate.json`
- source family: `TinyLlama-1.1B-Chat-v1.0`;
- train/select surface: `144` validation TinyLlama-vs-Qwen disagreement rows;
- frozen test surface: `473` TinyLlama-vs-Qwen disagreement rows;
- selected view/PCA/ridge: `hidden_residual / 32 / 100.0`;
- test matched/Qwen-substituted/cached-Tiny mean:
  `0.230/0.317/0.269`;
- test matched-minus-Qwen-substituted mean/min: `-0.088/-0.106`;
- test CI95 lower bound versus Qwen-substituted: `-0.160`;
- candidate-roll and receiver spectral-permutation controls remain below
  Qwen-substituted accuracy;
- pass gate: `False`.

Lay explanation: this run asked whether TinyLlama's internal hidden/query
vectors could be translated into the successful ARC public packet coordinate
system. On the hard rows where TinyLlama and Qwen disagree, that translated
TinyLlama hint was worse than the ordinary TinyLlama packet and much worse
than the Qwen-substituted packet.

The ARC transport/common-basis gate tests the next geometric repair:

- artifact:
  `results/source_private_arc_challenge_transport_common_basis_gate_20260502_tinyllama_disagreement/arc_challenge_transport_common_basis_gate.json`
- train/select surface: `144` validation TinyLlama-vs-Qwen disagreement rows;
- frozen test surface: `473` TinyLlama-vs-Qwen disagreement rows;
- candidates: nearest-neighbor barycentric transport, QJL-style sign-projected
  nearest-neighbor transport, and whitened orthogonal Procrustes;
- selected method/view/parameter: `whitened_procrustes / query_residual /
  dim=32`;
- frozen test matched/Qwen-substituted/cached-Tiny mean:
  `0.228753/0.317125/0.269345`;
- matched-minus-Qwen-substituted mean: `-0.088372`;
- matched-minus-cached-Tiny mean: `-0.040592`;
- CI95 lower bound versus Qwen-substituted: `-0.160677`;
- candidate-roll and spectral-permutation controls remain below Qwen;
- pass gate: `False`.

Lay explanation: this run tried three simple ways to translate TinyLlama's
internal hints into Qwen's public packet coordinate system: copy from nearby
examples, copy from random sign sketches, or rotate the spaces into alignment.
None helped on held-out rows.

The ARC sparse-query cache-bottleneck gate tests the nonlinear Mac-local
connector after the PCA/ridge and transport failures:

- artifact:
  `results/source_private_arc_challenge_sparse_query_cache_bottleneck_gate_20260502_tinyllama_disagreement/arc_challenge_sparse_query_cache_bottleneck_gate.json`
- train/select surface: `144` validation TinyLlama-vs-Qwen disagreement rows;
- frozen test surface: `473` TinyLlama-vs-Qwen disagreement rows;
- candidates: hidden/query residual views through train-only PCA, random
  Fourier features, top-k sparse query activations, and ridge decoding into the
  public ARC Fourier/anchor receiver basis;
- selected view/PCA/RFF/active/gamma/ridge:
  `hidden_query_residual / 16 / 32 / 16 / 1.0 / 1000.0`;
- frozen test matched/Qwen-substituted/cached-Tiny mean:
  `0.248203/0.317125/0.269345`;
- matched-minus-Qwen-substituted mean: `-0.068922`;
- matched-minus-cached-Tiny mean: `-0.021142`;
- CI95 lower bound versus Qwen-substituted: `-0.138531`;
- candidate-roll/content-rotation/spectral-permutation controls remain below
  Qwen-substituted accuracy;
- pass gate: `False`.

Lay explanation: this run gave TinyLlama a small nonlinear translator before it
sent its usual 12-byte ARC hint. The translator asked sparse questions of
TinyLlama's internal hidden/query state and converted those answers into the
public packet basis. On new hard rows, the translated hint was still worse than
Qwen's own packet.

The native systems result ingest gate turns the runbook into an enforceable
reviewer guardrail:

- artifact:
  `results/source_private_native_systems_result_ingest_gate_20260502/native_systems_result_ingest_gate.json`
- validator pass: `True`;
- native systems complete: `False`;
- paper native win allowed: `False`;
- measurement rows ingested: `0`;
- required baseline rows: `11`;
- missing required rows: `11`;
- invalid measurement rows: `0`;
- missing rows: LatentWire cached/end-to-end packet rows, target-only
  vLLM/SGLang, same-byte visible text, source-label-copy, C2C, KVComm, KVCOMM,
  QJL, and TurboQuant.

Lay explanation: this checker will not let the paper claim a real GPU systems
win until every required method has the same accuracy, latency, memory,
traffic, byte, and source-exposure fields.

The ARC Llama-8B frozen-disagreement source scout now clears the Mac hardware
blocker but fails the strict source-family gate:

- script:
  `scripts/build_source_private_arc_challenge_llama8b_disagreement_source_scout.py`
- artifact:
  `results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502/llama8b_disagreement_source_scout.json`
- surface: `144` validation and `473` frozen test TinyLlama-vs-Qwen ARC
  disagreement rows;
- source model: locally cached `Meta-Llama-3.1-8B-Instruct`;
- MPS workaround: `attn_implementation=eager`, `choice_batch_size=1`;
- pass gate: `False`;
- validation matched/Qwen-substituted/cached-Tiny mean:
  `0.356/0.389/0.250`;
- validation CI95 lower bound versus Qwen-substituted: `-0.174`;
- frozen test matched/Qwen-substituted/cached-Tiny mean:
  `0.368/0.317/0.269`;
- frozen test matched-minus-Qwen-substituted mean/min: `+0.051/+0.017`;
- frozen test CI95 lower bound versus Qwen-substituted: `-0.035`;
- frozen test same-byte visible text: `0.495`.

Lay explanation: this run asked whether a stronger non-Qwen model can send the
same 12-byte ARC hint on the hard rows. It produced a promising frozen-test
average, but it lost on validation and the uncertainty interval still overlaps
zero, so it is not a reviewer-safe positive.

The ARC Llama-8B failure probe explains why the source-choice branch should not
be revived as-is:

- artifact:
  `results/source_private_arc_llama8b_failure_probe_20260502/arc_llama8b_failure_probe.json`
- pass gate: `False`;
- best overall router: `source_matches_llama_prediction`;
- best overall router deployable without source index: `False`;
- best deployable router: `packet_margin_ge:0.131799`;
- validation/test best-deployable-router accuracy: `0.408/0.362`;
- test source/Qwen oracle accuracy: `0.613`;
- test Llama/Qwen packet oracle accuracy: `0.532`;
- test same-byte visible text minus Llama packet: `+0.126`;
- test source-to-Llama-packet loss: `0.186`.

Lay explanation: Llama often knows useful answers, but the current tiny packet
does not reliably carry that answer to the receiver. A simple diagnostic rule
can exploit the audit-only source index, but that would amount to transmitting
the source's answer choice. The next positive method must therefore learn a
better bottleneck/receiver interface rather than just route source choices.

The ARC hidden/query MLP cache connector tests the remaining Mac-local learned
connector supported by the current caches:

- artifact:
  `results/source_private_arc_challenge_hidden_query_mlp_cache_connector_gate_20260502_tinyllama_disagreement/arc_challenge_hidden_query_mlp_cache_connector_gate.json`
- pass gate: `False`;
- selected view/PCA/hidden/weight decay: `query_residual / 16 / 16 / 0.001`;
- frontier candidates: `36`;
- validation/test disagreement rows: `144/473`;
- frozen test matched/Qwen-substituted/cached-Tiny/target/same-byte-text:
  `0.232/0.317/0.269/0.268/0.258`;
- matched-minus-Qwen-substituted mean: `-0.085`;
- matched-minus-cached-Tiny mean: `-0.038`;
- CI95 lower bound versus Qwen-substituted: `-0.154`;
- candidate-roll/content-rotation/spectral-permutation controls:
  `0.261/0.255/0.236`.

Lay explanation: this run tried a small learned translator instead of another
hand-built geometry map. TinyLlama's cached hidden/query signals were decoded
into the public ARC packet language, but on new hard rows the learned 12-byte
hint was worse than Qwen's own packet and worse than the cached Tiny packet.

The 2026-05-02 evidence bundle now ingests the HellaSwag PQ hidden innovation
codec gate:

- artifact:
  `results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/hellaswag_pq_hidden_innovation_codec_gate.json`
- default `pq_pca16_m4_k2_identity`: `0.497070` accuracy versus packet-only
  `0.501953`, delta `-0.004883`, CI95 low `-0.017578`;
- best diagnostic scout `pq_pca8_m2_k8_orthogonal`: `0.508789`, delta
  `+0.006836`, CI95 low `0.000000`;
- pass gate: `False`.

Lay explanation: the experiment tried to let TinyLlama send a tiny compressed
fingerprint of its hidden-state reasoning, not just its answer choice. Qwen got
that one-byte fingerprint and its own scores, then tried to choose better. The
fingerprint was sensitive to corruption controls, but it did not reliably help
Qwen beyond the compact packet-only baseline.

## Reviewer-Safe Decision

Cut the claim that HellaSwag currently demonstrates receiver improvement over
packet-only. Keep HellaSwag as:

- a hard non-science benchmark and complementarity/headroom diagnostic;
- a systems byte/exposure row under the fixed packet contract;
- a negative-ablation suite showing which hidden/source-code variants fail.

Do not describe the current HellaSwag branch as robust cross-model latent
reasoning. The stronger claim requires a learned connector that survives the
same controls.

## ICLR Needs

- Positive learned receiver/common-basis method on at least two public
  benchmarks or one benchmark plus a strict cross-family pair.
- The highest-value next method is a true tokenwise 16-64 query bottleneck or
  soft-prefix connector trained against target loss. The current mean-cache
  MLP proxy is negative and should not be widened.
- Seed repeats, larger frozen slices, paired CIs, and source-destroy controls.
- Direct competitor comparisons against C2C/KVComm-style KV/cache transfer and
  KV quantization byte floors.
- Native NVIDIA serving rows for TTFT, TPOT, goodput, GPU memory, HBM traffic,
  payload bytes, and source-exposure flags, ingested through the new validator.

## COLM Workshop Needs

- Keep the scope honest: source-private packet protocol, public-benchmark
  positives, systems/accounting, and negative latent-code ladder.
- Make the three contributions explicit and avoid claiming solved latent
  reasoning.
- Add one clean figure summarizing the gate tree: ARC/OpenBookQA positive,
  SciQ/CommonsenseQA diagnostic, HellaSwag branch killed for receiver
  improvement, and TinyLlama hidden/query PCA/ridge, transport, sparse-query
  bottleneck, and MLP cache-to-packet connector ruled out for ARC repair.

## Blockers For User Help

- NVIDIA access for vLLM/SGLang native systems baselines and any true
  continuous query/cache connector.
- A trainable query/cache connector or new stronger cross-family source branch;
  the current Mac-local Phi-3 and Llama-8B source-choice senders are not enough.
- Confirmation on whether COLM should be framed as a workshop paper with
  negative HellaSwag ablations or held until a positive learned connector
  exists.

## Next Exact Gate

Run a stronger true cross-family source or trainable query/cache connector on
NVIDIA, then fill the native systems schema with matched vLLM/SGLang/C2C/
KVComm/QJL/TurboQuant rows. Receiver-only scalar routing, source-side scalar
confidence routing, cached candidate-level packet/score connectors, Mac-local
Phi-3 source packets, and shallow TinyLlama hidden/query PCA/ridge connectors
or static transport/Procrustes connectors, random Fourier sparse-query cache
bottlenecks, and train-only MLP mean-cache connectors are now ruled out for
this ARC disagreement surface. The
same-family Qwen-1.5B stronger-source diagnostic should be repeated with a true
stronger cross-family source before making the ICLR claim. The current
Llama-8B source-choice scout is now scientifically resolved as a strict gate
failure, although a separately selected Llama prompt/scoring/calibration branch
could be revived if it first clears validation. The Llama failure probe and the
negative MLP cache connector jointly show that useful source signal is being
lost at the current packet/receiver interface; the next exact method branch
needs tokenwise target-forward connector infrastructure, while Mac-local work
should focus on consolidation and the NVIDIA systems/connector runbooks.

## 2026-05-02 Gate-Tree Consolidation

Added the gate-tree artifact:
`results/source_private_iclr_gate_tree_and_connector_plan_20260502/`.

This locks the current contribution boundary:

- keep the fixed-byte source-private packet protocol;
- keep public-basis ARC/OpenBookQA benchmark gates plus the negative ladder;
- keep systems byte/exposure accounting, but not native speed/HBM claims;
- cut TinyLlama mean hidden/query connector variants on ARC;
- make the next live method gate a target-loss tokenwise soft-prefix/query
  connector.

The next exact Mac-local step is an ARC n32 target-loss soft-prefix preflight
using Qwen source features and a frozen target. If it passes destructive
controls, the paper-facing gate becomes the OpenBookQA 3B train-only receiver
over packet-only with positive paired CI95 lower bound.

## 2026-05-02 Soft-Prefix Preflight Scaffold

Added the first isolated target-loss ARC/OpenBookQA soft-prefix scaffold and
CPU smoke artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_hidden_n8_cpu_label_choice/`.

Readout:

- implementation path exists and emits reviewer-critical controls;
- MPS target `inputs_embeds` is blocked by an Apple attention-shape failure;
- CPU Qwen-source-hidden n8 smoke is negative: matched soft-prefix `0/4`,
  target-only `1/4`, slots-only/static prefix `2/4`, same-norm noise `2/4`,
  and source-label-copy audit upper bound `3/4`.

Decision: the current tiny selected-hidden soft-prefix is not a positive
method. Keep the branch alive only as a larger tokenwise/query connector with
better source pooling, continuation scoring selected on validation, seed
repeats, and ideally NVIDIA. This result strengthens the reviewer story by
showing we now test source necessity directly rather than relying on shallow
mean-cache proxies.

## 2026-05-02 Residual Soft-Prefix Follow-Up

Added row-centered residual feature modes to the soft-prefix preflight and a
CPU smoke artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_hidden_residual_n8_cpu_label_choice/`.

Readout:

- residual selected hidden improves over absolute selected hidden on the same
  n8 smoke surface;
- matched accuracy moves from `0/4` to `1/4`;
- matched-minus-best-control margin improves from about `-0.751` to `-0.150`;
- pass gate remains `False` because slots-only/static prefix reaches `2/4` and
  label-shuffled margin remains stronger than matched.

Decision: row-centered residuals are a better feature basis than absolute
selected hidden, but the selected-vector soft-prefix connector is not yet a
positive method. The next branch should be tokenwise/query pooling over all
source candidate states, not another single-vector variant. This is also a
stronger story for reviewers because it shows the project is actively testing
whether the source information is necessary rather than assuming hidden-state
transfer works.

## 2026-05-02 Token-Pool Query Follow-Up

Added token-pool source feature modes and a learned query-pooling soft-prefix
connector to the ARC/OpenBookQA preflight:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_token_pool_residual_n8_cpu_label_choice/`.

Readout:

- implementation path works for rank-3 source token pools and preserves the
  existing target-only/static/zero/shuffle/noise/train-mean/label-shuffled/
  same-byte/audit controls;
- corrected token extractor avoids silently pooling the whole prompt when
  candidate suffix tokens are truncated;
- matched query soft-prefix accuracy is `0/4` on the n8 CPU smoke surface;
- target-only, zero-source, train-mean source, and same-byte visible text each
  reach `1/4`;
- label-shuffled reaches `2/4`;
- matched-minus-best-control margin is about `-0.566`;
- pass gate remains `False`.

Decision: shallow all-candidate token pooling is now ruled out on this Mac
surface. The negative result weakens the hypothesis that more hidden-state
exposure alone solves cross-model communication. The next live connector
branch should be candidate-level residual pooling with explicit
source-score-only, hidden-residual-only, and score+hidden factor ablations,
then only widen if the n8 smoke beats label-shuffled and zero-source controls.

## 2026-05-02 Candidate-Pool Factor Follow-Up

Added candidate-level score/hidden factor modes to the ARC/OpenBookQA
soft-prefix preflight and ran three n8 CPU ablations:

- cached source-choice score residual:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_cached_score_pool_residual_n8_cpu_label_choice/`
- hidden candidate residual:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_candidate_pool_residual_n8_cpu_label_choice/`
- hidden+score candidate residual:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_candidate_pool_residual_n8_cpu_label_choice/`

Readout:

- score-only matched reaches `1/4`, but zero-source reaches `2/4`;
- hidden-only matched reaches `1/4`, tying target-only and zero-source, but
  still loses on margin;
- hidden+score matched falls to `0/4`, while target-cache-only and same-norm
  noise reach `2/4`;
- all three pass gates are `False`;
- source-label-copy audit remains `3/4`, so the cached source selection has
  answer-side signal, but the learned connector does not use it safely.

Decision: this rules out shallow candidate-level query soft-prefix pooling on
the current Mac-local surface. The next live branch should be a conditional
candidate innovation/syndrome packet: encode source evidence relative to the
target's own candidate state, then compare hidden-only, score-only, and
score+hidden factors at matched rate with explicit target-derived and
shuffled-source packet controls.

## 2026-05-02 Public-Innovation Soft-Prefix Follow-Up

Added train-only public-side-information innovation modes to the ARC/OpenBookQA
soft-prefix preflight:

- `hf_choice_hidden_public_innovation_candidate_pool(_residual)`
- `hf_choice_hidden_score_public_innovation_candidate_pool(_residual)`

Readout:

- hidden public innovation residual is prediction-free but reaches only `1/4`,
  while shuffled-source reaches `2/4`;
- hidden+score public innovation residual with ridge `10` reaches `2/4` and
  has a small positive margin delta (`+0.051`), but only ties the
  target-cache-only learned prefix at `2/4` and also ties same-norm noise;
- target-conditioned query worsens to `1/4`;
- ridge `1000` lowers the fit explained-variance diagnostic from `0.9985` to
  `0.6295`, but also worsens matched accuracy to `1/4`;
- all pass gates remain `False`.

Decision: rule out train-only public-ridge innovation as a standalone
soft-prefix positive method on this Mac-local gate. The conditional-innovation
story remains scientifically cleaner than raw hidden transfer, but the next
exact branch should be a source-control-contrastive conditional innovation
receiver with packet-level candidate-roll controls, not another answer-CE-only
soft-prefix tweak.

## 2026-05-02 Source-Control Contrastive Follow-Up

Added a matched-only source-control contrastive objective to the ARC/OpenBookQA
soft-prefix preflight. The receiver now optionally penalizes a matched packet
when its gold-vs-distractor margin does not beat zero-source, shuffled-source,
same-norm-noise, and candidate-roll-source packets.

Readout:

- one epoch, weight `0.2`: matched reaches `3/4`, but candidate-roll source
  also reaches `3/4`; matched-minus-best-control margin is about `-0.043`;
- three epochs, weight `0.2`: matched collapses to `1/4`, ties target-only by
  accuracy, and loses the margin gate by about `-0.234`;
- both pass gates remain `False`;
- the new `candidate_roll_source` control is the key destructive diagnostic:
  it preserves same-row packet statistics while changing candidate slot
  alignment.

Decision: weaken the contrastive soft-prefix branch. Do not widen this exact
method. For ICLR, the next learned-receiver branch must be
candidate-alignment-sensitive and must clear candidate-roll/candidate-
derangement controls before any larger ARC/OpenBookQA slice. For the systems
contribution, the highest-value Mac-local follow-up is a byte-amplification
ablation that compares fixed packets with 64B cache-line padding and synthetic
QJL/TurboQuant/KVQuant source-state byte floors, without claiming NVIDIA
throughput until native serving rows exist.

## 2026-05-02 Byte-Amplification Systems Follow-Up

Added a Mac-local byte-amplification ablation:
`results/source_private_byte_amplification_ablation_20260502/`.

Readout:

- pass gate: `True`;
- benchmark rows: `4`;
- interface rows: `40`;
- packet framed-byte range: `4-15B`;
- worst-case single-request cache-line amplification: `16.0x`;
- worst-case single-request DMA amplification: `32.0x`;
- minimum one-token KV/source-state floor: QJL 1-bit at `768B`;
- QJL floor versus largest framed packet: `51.2x`;
- QJL floor versus `64B` cache-line padded packet: `12.0x`;
- TurboQuant 3.5-bit one-token KV floor: `2688B`;
- C2C/fp16 one-token KV floor: `12288B`;
- fp16 source-score stress row: `8B`, but explicitly marked non-private
  because it exposes raw source scores.

Decision: promote this artifact into the systems evidence bundle. It answers a
likely reviewer objection cleanly: byte-small alternatives exist, but bytes are
not the only axis; source exposure and object scaling matter. LatentWire packet
rows are fixed task evidence, while QJL/TurboQuant/KIVI/KVQuant/C2C/KVComm
rows are compressed or transported state objects. This does not close native
throughput claims; NVIDIA vLLM/SGLang rows remain a blocker for ICLR systems
strength.

## 2026-05-02 Candidate-Alignment Receiver Follow-Up

Added an external ARC candidate-alignment receiver:
`scripts/run_source_private_arc_candidate_alignment_receiver_preflight.py`.
The gate scores choices outside the target LM using public candidate features,
source candidate sketches, source/public compatibility terms, and destructive
controls.

Readout:

- sign-16 hidden-only sketch (`8B`): matched `1/4`, best control `3/4`;
- unquantized hidden-only pointwise sketch (`256B`): matched `3/4`, but mean
  margin still loses target-public-only by about `0.012`;
- sign-64 hidden-only sketch (`32B`): matched `1/4`, best control `2/4`;
- int8-16 pairwise sketch (`66B`): matched `3/4`, target-public-only also
  `3/4`;
- unquantized pointwise n16 probe: matched `1/8`, target-public-only `3/8`;
- all pass gates are `False`.

Decision: do not widen this linear candidate-alignment receiver. The only
positive-looking n8 row is uncompressed, margin-negative, and does not survive
n16. The next method branch should freeze a target-public scorer and train a
source residual correction only on target errors, with wrong-row,
candidate-roll, zero-source, and target-only controls. If that fails, the
right architectural move is a permutation-equivariant DeepSets/Set Transformer
receiver with source-control contrastive training.
