# SA2 Positive-Method Ideation, 2026-05-24

## Status Gate

Current paper readiness: **not ICLR-ready as a positive-method paper**. The
LatentWire story is still "cross-model latent transfer has narrow, reproducible
same-family clues, but no method has survived the stronger decision surface."
The exact blocking gap is a positive method that survives (1) a larger frozen
slice, (2) seed repeats with paired uncertainty, and (3) at least one strict
cross-family falsification pair.

Current story of the paper: the strongest LatentWire clue is
`dynalign_module_replace_residrank16`, which reaches `0.1250` on the frozen
GSM8K32 contract, tying `C2C` and beating `target_alone = 0.0625` and
`text_to_text = 0.0312`, with full numeric coverage (`32/32`) and `2/32` wins.
Reviewer diagnostics make this more interesting but still provisional:
`source_alone = 0.0312`, the source is wrong on both latent-win examples, and
the oracle bound on GSM8K32 is already `4/32`, so the slice is saturated.

What is saturated: fixed gauge wrappers (`0/32`), first SAE adapter (`0/32`),
byte-only alignment on same-family GSM8K32 (`0.0312`), heavier dynalign teacher
variants (`0.0312-0.0625`), saliency/eigenspace/preserve-core residuals
(`0.0312-0.0625`), one-gate routed residuals (`0.0625`), and simple routed
banks (`0.0938`). What remains alive: dynalign residual rank16 and its
value-routed preservation (`0.1250`), quotient/GPA/sparse-dictionary low-shot
toys, sequence-aligned byte sidecars under interface stress (`0.0360` MSE at
1-shot), anchor-preserving codecs, and learned connector/query-bottleneck
interfaces.

No GPU experiment, preregistration, or code path is proposed here. This is a
method portfolio for M37+.

## Evidence Anchors Used

| Artifact | Concrete read | Implication for ideas |
|---|---:|---|
| GSM8K32 control | target `0.0625`, text relay `0.0312`, C2C `0.1250` | Any method must beat text, target, and C2C on a bigger slice; text can harm. |
| Dynalign rank16 | `0.1250`, `2/32` wins, `0` losses, coverage `32/32` | The live clue is output-aware and narrow, not generic residual capacity. |
| Tokenbasis rank16 | `0.0625`, `0/32` wins | Rank16 alone is not enough; the teacher/interface matters. |
| Source diagnostic | source `0.0312`; source wrong on both latent wins | Avoid "answer-copying" narratives; seek conditional reasoning signals. |
| GSM8K32 oracle | `max(target, dynalign)=0.1250` | GSM8K32 cannot rank more verifier-side variants. |
| Sequence sidecar toy | `0.0360` MSE at 1-shot under strong interface stress | Interface-side sidecars are the best low-shot toy clue. |
| Quotient+GPA+sparse dictionary | `0.0568/0.0576` MSE at 1/2 shots | Low-shot canonical shared bases are alive, but only as ingredients. |
| Sticky routing toy | route acc `0.9875`, stability `1.0000` | Routing helps only when feature-grounded and stable. |
| Confidence-only routing | route acc `0.2812-0.3688`, harm `0.5813` | Confidence must be a stop/uncertainty feature, not the router. |
| M11b/outlier-migrate analogy | Granite top5 median recovery `0.4493`, CI crosses zero; Nemotron invalid packet | Decode-time budget adaptation is suggestive, not settled; use only as inspiration. |

## Summary Table

| ID | Idea | Primary mechanism | Main empirical motivation | Est. GPU cost | PASS probability | Suitability |
|---|---|---|---|---:|---:|---|
| M37 | Conditional innovation resampler | Encode source-minus-target residual through learned queries | Source wrong on both wins; target needs deltas, not answers | Medium | 15-28% | ICLR if cross-family passes |
| M38 | C2C-compatible dynalign fuser | Per-layer/per-head fuser with dynalign residual channel | C2C and dynalign both `0.1250`; simple banks fall to `0.0938` | Medium | 12-24% | ICLR baseline-challenger |
| M39 | Sequence-aligned byte sidecar | Byte/span + sequence profile side channel | Best toy MSE `0.0360` under interface stress | Low-Med | 12-22% | Workshop to ICLR if cross-family |
| M40 | Relative-anchor sparse packet | Anchor-coordinate sparse features, not raw hidden maps | Fixed wrappers collapse; quotient matching gets MSE `0.0796` | Medium | 10-20% | Workshop first |
| M41 | Rate-distortion query bottleneck | Q-Former/Perceiver-style learned soft queries | Interface redesign avoids raw latent overfit | Medium-High | 14-26% | ICLR if beats C2C |
| M42 | Event-triggered innovation gate | Transmit only when target uncertainty and source delta agree | Text harms; oracle-saturated small slice | Low | 8-16% | Workshop |
| M43 | Anchor-preserving tail codec | Preserve dominant anchors, learned tail packet | Uniform tail toy `0.9896` acc, MSE `0.0284`; naive codebook stalls | Low-Med | 8-18% | Workshop |
| M44 | Sticky feature-routed projector bank | Stable feature router + small projector experts | Sticky route acc `0.9875`; simple banks fail | Medium | 8-17% | Workshop |
| M45 | Error-correcting syndrome packet | Send correction syndrome over target trace errors | Source wrong on wins implies hidden partial info | Medium | 7-16% | Workshop |
| M46 | Local kNN geometry bridge | Preserve neighborhoods instead of global CKA/rotation | Fixed global maps collapse to `0/32` | Medium | 7-15% | Workshop |
| M47 | Soft-prefix hidden gist | Source hidden to 4-16 target soft tokens | Gist/prefix prior; target text relay harms | Medium | 9-18% | Workshop/ICLR if robust |
| M48 | Receiver-side Kalman controller | Innovation gain controller over residual updates | Value-routed preserves `0.1250`; saliency/eigenspace fail | Medium | 8-16% | Workshop |
| M49 | Source-disagreement packet | Encode where source and target latent predictions disagree | Conditional value over target-alone is the real goal | Medium | 6-14% | Workshop |
| M50 | Byte-level distillation bridge | Train receiver through byte interface across tokenizers | Same-family tokenizer is easy; cross-family mismatch is real | Medium | 7-16% | Workshop |
| M51 | Attribution-path packet | Send sparse attribution graph features | Needs interpretability and non-copy proof | High | 5-12% | Workshop |
| M52 | Matched-rate KV packet baseline | Strong KV quant/compression baseline plus source-conditioned residual | Reviewers will require matched bytes/rate | Low-Med | 6-12% | Baseline, not headline |
| M53 | Multi-source consistency receiver | Two weak source packets with disagreement-aware fuse | Text can poison; latent may avoid poison | High | 5-11% | Workshop only |
| M54 | Dynamic budget latent packet | Increase packet budget only on hard positions | M11b budget clue; CI weak | Medium | 5-13% | Workshop |
| M55 | Structured-text plus latent hybrid | Schema trace + tiny latent residual | Structured text is the real text baseline | Low-Med | 6-14% | Necessary control |
| M56 | Abstaining receiver selector | Learned accept/defer around candidate packet | Oracle bound on GSM8K70 was `10/70` vs live `8/70` | Low | 6-12% | Control/appendix |

## Method Ideas

### M37: Conditional Innovation Resampler

Mechanism: run both source and target on the prompt, align only enough to form a
conditional residual, then feed `A_state - f(B_state)` through 16-64 learned
queries consumed by the target. Train the bridge on target next-token loss plus
a small information bottleneck penalty. This is Wyner-Ziv style source coding:
the target already has side information, so the packet should encode what the
source knows that the target lacks.

Empirical motivation: the current latent wins are not source-answer copying:
`source_alone = 0.0312`, and the source is wrong on both `dynalign+rank16` win
examples. `text_to_text = 0.0312` also shows raw source output can poison the
target. A conditional packet directly targets the non-copy explanation.

Prior-art adjacency: Cache-to-Cache trains a receiver fuser for cache exchange
(arXiv:2510.03215). Q-Former/BLIP-2 and Perceiver Resampler/Flamingo show that
learned query bottlenecks can bridge heterogeneous encoders and LLMs
(https://arxiv.org/abs/2301.12597, https://arxiv.org/abs/2204.14198). This
idea differs by encoding a conditional innovation rather than the source state.

Implementation/GPU cost: medium. Reuse the campaign runner; add a small
query-resampler module and train on 2k-10k GSM8K-style items before frozen
500+ evaluation. No custom kernels.

Honest PASS probability: 15-28%. Risks: may learn a target-only soft prompt,
may collapse cross-family if the conditional residual is not stable, and can be
expensive enough to lose the byte/latency story. Composability: high with M39,
M42, M48. Suitability: ICLR-worthy if it beats C2C and structured text at
matched rate on same-family and one cross-family pair; otherwise workshop.

### M38: C2C-Compatible Dynalign Fuser

Mechanism: implement a strong C2C-like fuser as the baseline and add the
dynalign rank16 residual as a separate gated input, with per-layer gates,
per-head/value weighting, and a binary inference gate. Treat dynalign as one
source feature, not the whole bridge.

Empirical motivation: C2C and `dynalign_module_replace_residrank16` both reach
`0.1250` on GSM8K32. Simpler value-bank and query-bank variants fall to
`0.0938`, which says fuser structure matters. The review note explicitly warns
that the killed one-gate row is weaker than modern C2C-style per-layer/per-head
fusion.

Prior-art adjacency: C2C is the closest competitor (arXiv:2510.03215). The
fuser must be presented as a strong baseline unless the dynalign channel adds
positive matched-rate lift.

Implementation/GPU cost: medium. Needs model-cache capture and a small fuser
training loop; no architecture surgery beyond cache integration.

Honest PASS probability: 12-24%. Risks: may simply reproduce C2C, may be
scooped by C2C if no distinctive conditional residual contribution remains.
Composability: high with M37 and M42. Suitability: ICLR only if the dynalign
channel gives clear lift or lower bytes/latency over C2C; otherwise it is the
required competitor.

### M39: Sequence-Aligned Byte Sidecar

Mechanism: attach a tokenizer-agnostic byte/span sidecar plus sequence-alignment
profile to a sparse shared-basis packet. The sidecar should include byte-span
hashes, local alignment confidence, and compact source-state summaries tied to
span positions rather than token IDs.

Empirical motivation: under strong interface corruption, the sequence-aligned
sidecar is the best shared-basis toy: MSE `0.0360` at 1-shot and `0.0362` at
2-shot, beating plain byte sidecar (`0.0392/0.0394`) and remap-only
(`0.0566/0.0570`). Same-family Qwen tokenizers are effectively identical, so
this should be tested on cross-family pairs, not the easy same-family pair.

Prior-art adjacency: cross-tokenizer byte distillation (arXiv:2604.07466),
TokAlign (arXiv:2506.03523), and byte-level interfaces are adjacent. The
novelty must be source-state communication plus sequence-aligned sidecar, not
byte remapping alone.

Implementation/GPU cost: low-medium. Mostly data/interface work plus a bridge
module; run on Qwen->Llama/Phi/Mistral-style mismatch.

Honest PASS probability: 12-22%. Risks: may only improve reconstruction MSE
without task lift; may be irrelevant for same-tokenizer pairs. Composability:
very high with M37 and M40. Suitability: workshop if task lift is small; ICLR
if it creates the first strict cross-family positive row.

### M40: Relative-Anchor Sparse Packet

Mechanism: represent source states by similarities to task-relevant anchors,
then sparsify those anchor coordinates before target injection. Avoid a global
linear map; train target-side decoder on anchor coordinates and log anchor
usage/recovery.

Empirical motivation: fixed rotations and shared-basis wrappers collapse to
`0/32`; quotient-aware matching succeeds in the toy when made gauge-invariant,
with `0.0796` MSE at 1-shot and exact head matching. This suggests local
invariant coordinates are more promising than raw hidden-state geometry.

Prior-art adjacency: Relative Representations (OpenReview:
https://openreview.net/forum?id=SrC-nwieGJ) and the Platonic Representation
Hypothesis (https://arxiv.org/abs/2405.07987) motivate anchor/local structure,
while calibration critiques warn against overclaiming global convergence.

Implementation/GPU cost: medium. Requires anchor construction, sparse packet
training, and destructive anchor-shuffle controls.

Honest PASS probability: 10-20%. Risks: first SAE adapter already failed
(`0/32`), so sparse features must be task- and receiver-trained. Composability:
high with M39 and M45. Suitability: workshop first; ICLR only with
interpretable anchors and cross-family task lift.

### M41: Rate-Distortion Query Bottleneck

Mechanism: a standalone learned connector: source hidden states cross-attended
by 8/16/32/64 learned queries, output as soft tokens or layer injections for
the target. Sweep query count and bits to produce a rate-distortion curve.

Empirical motivation: the present evidence is a single operating point. The
reviewer's power critique says GSM8K32 cannot distinguish `2/32` from `4/32`;
a rate curve on 500+ examples would make the claim interpretable. Query
bottlenecks also directly limit interference, which text relay currently
violates.

Prior-art adjacency: Flamingo Perceiver Resampler and BLIP-2 Q-Former are the
obvious connector precedents; Vision Wormhole is adjacent for heterogeneous
model connectors (https://arxiv.org/abs/2602.15382).

Implementation/GPU cost: medium-high. Needs training data and larger frozen
evaluation. Strong controls: target-only, structured text, C2C, random-source,
zero-source, shuffled-source.

Honest PASS probability: 14-26%. Risks: can become a generic task adapter
rather than communication; must require source-conditioned destructive
controls. Composability: high with M37, M39, M55. Suitability: ICLR if the
source controls are clean and the rate curve beats C2C/structured text.

### M42: Event-Triggered Innovation Gate

Mechanism: transmit or apply the latent packet only when target uncertainty,
source-target disagreement, and packet confidence all cross a learned gate.
The default path is target-alone; the method is judged by help/harm and
coverage, not raw always-on accuracy.

Empirical motivation: `text_to_text` harms (`0.0312` vs target `0.0625`), and
many valid latent variants add one win plus one loss. The GSM8K70 diagnostic
had live row `8/70` and oracle `10/70`, so selective acceptance has headroom on
larger slices even though GSM8K32 is saturated.

Prior-art adjacency: selective classification and verifier-gated packet work;
not novel alone, but novel if the gate is source-conditioned over latent
innovation rather than output confidence. Confidence-only routing is explicitly
ruled out by local toy accuracy `0.2812-0.3688`.

Implementation/GPU cost: low. Requires no new source model; train/evaluate
gate on existing candidate rows after larger frozen campaigns.

Honest PASS probability: 8-16%. Risks: reviewer may see it as a selector, not
a communication method; can overfit small slices. Composability: high with all
packet methods. Suitability: workshop or appendix unless paired intervals are
strong.

### M43: Anchor-Preserving Tail Codec

Mechanism: preserve a small set of dominant/anchor coordinates losslessly or
high precision, and encode the tail through a learned residual codec. Unlike
the failed naive codebook, condition the tail on target sensitivity and anchor
identity.

Empirical motivation: preserve-topk uniform tail toy reaches `0.9896` accuracy
and MSE `0.0284`, while naive codebook tail stalls around `0.9844` and MSE
`0.2470`. The positive signal is anchor preservation, not the first codebook.

Prior-art adjacency: KV/cache compression and quantization methods such as
KVQuant (https://arxiv.org/abs/2401.18079), KIVI
(https://arxiv.org/abs/2402.02750), and SmoothQuant
(https://arxiv.org/abs/2211.10438) are matched-rate baselines. The novelty
would be receiver-task-conditioned cross-model packet coding.

Implementation/GPU cost: low-medium. Mostly offline codec training plus task
eval at matched bytes.

Honest PASS probability: 8-18%. Risks: may improve MSE without reasoning
accuracy; byte accounting can erase the gain. Composability: high with M40.
Suitability: workshop unless it beats C2C/KV baselines at matched bytes.

### M44: Sticky Feature-Routed Projector Bank

Mechanism: route examples/tokens to a small bank of projectors using stable
feature IDs and paraphrase-stable routing, with load-balance and route entropy
regularization. Use sticky route memory to prevent route flapping.

Empirical motivation: toy sticky feature routing reaches route accuracy
`0.9875` and stability `1.0000`, but confidence-only routing is harmful and
simple value/query banks on GSM8K32 fall to `0.0938`. The open question is
whether stable feature routing, not bank capacity alone, is the missing part.

Prior-art adjacency: mixture-of-experts routing and feature dictionaries. This
must be scoped as a communication router, not generic MoE.

Implementation/GPU cost: medium. Needs real route-pool feature IDs and random,
confidence, and oracle route controls.

Honest PASS probability: 8-17%. Risks: route labels may be unavailable or
unstable on real prompts; interaction with residuals may be non-additive.
Composability: medium with M37/M43. Suitability: workshop.

### M45: Error-Correcting Syndrome Packet

Mechanism: train a compact packet to predict target error syndromes rather
than final answers: missing operation, wrong quantity binding, sign/unit error,
or arithmetic-step mismatch. Inject the syndrome as a small continuous or
structured latent correction.

Empirical motivation: source wrong on both latent wins implies the signal may
be partial reasoning structure. A syndrome packet tests whether the source can
identify target failure modes even when it cannot solve the final problem.

Prior-art adjacency: coding-theory syndrome framing and process-supervision
style reasoning diagnostics. Keep citations to coding analogy minimal unless
implemented with real parity/control tests.

Implementation/GPU cost: medium. Requires labeling or deriving error classes
from traces; cheaper than full connector training if labels are synthetic.

Honest PASS probability: 7-16%. Risks: syndrome labels may be noisy; reviewers
may call it a verifier unless source-conditioned latent features are essential.
Composability: high with M42/M55. Suitability: workshop first.

### M46: Local kNN Geometry Bridge

Mechanism: train the bridge to preserve local neighborhoods and task-neighbor
relations across source and target states instead of optimizing global CKA,
rotation, or dominant eigenspaces. Use neighborhood recall, trustworthiness,
and local rank metrics as telemetry.

Empirical motivation: global fixed wrappers fail catastrophically (`0/32` and
coverage `0/32`), and eigenspace residual regresses to `0.0312`. Local
geometry is the remaining plausible geometry story.

Prior-art adjacency: Relative Representations and the Platonic Representation
Hypothesis support local/anchor views; avoid broad "universal latent space"
claims.

Implementation/GPU cost: medium. Needs cached hidden states and neighborhood
mining; task eval still required.

Honest PASS probability: 7-15%. Risks: can become another MSE win with no task
lift; local neighborhoods may not align for weak models. Composability: medium
with M40. Suitability: workshop.

### M47: Soft-Prefix Hidden Gist

Mechanism: source hidden states are compressed into 4-16 target-embedding soft
tokens prepended to the target. Train only the encoder/projection and optional
receiver adapter, with zero-source and shuffled-source controls.

Empirical motivation: text relay harms, but a small continuous prefix can carry
non-surface information with a controlled interference footprint. This is the
simplest learned connector variant.

Prior-art adjacency: Gist tokens for prompt compression
(https://arxiv.org/abs/2304.08467), prefix tuning, BLIP-2/Q-Former connectors.
Novelty requires source-hidden conditioning, not target-only prompt tuning.

Implementation/GPU cost: medium. Easy to prototype; needs rigorous controls.

Honest PASS probability: 9-18%. Risks: may be dismissed as soft-prompt tuning;
source ablations are mandatory. Composability: high with M37/M41. Suitability:
workshop; ICLR only if source-conditioned controls are decisive.

### M48: Receiver-Side Kalman Controller

Mechanism: model residual repair as an innovation-gain controller. The target
estimates its own state uncertainty, the source packet estimates innovation,
and the controller learns a bounded gain per layer/token.

Empirical motivation: value-routed residual preserves the live `0.1250`, while
saliency/eigenspace/preserve-core variants drop to `0.0312-0.0625`. The
working behavior may be controlled innovation magnitude rather than better
basis selection.

Prior-art adjacency: observer/Kalman filtering is only an analogy; do not
overclaim unless the implementation logs innovation covariance/gain.

Implementation/GPU cost: medium. Add gain head and telemetry to existing
dynalign residual harness.

Honest PASS probability: 8-16%. Risks: controller may simply learn to turn off;
needs nonzero help/harm telemetry. Composability: high with M37/M42. Suitability:
workshop.

### M49: Source-Disagreement Packet

Mechanism: encode only positions where the source and target disagree in
intermediate logits, latent classifiers, or answer-candidate distributions.
The packet is sparse over disagreement events.

Empirical motivation: target-alone is weak (`2/32`), text source can poison
(`1/32`), and source-alone is also weak (`1/32`). Useful signal likely lives in
where computations diverge, not in either full answer.

Prior-art adjacency: multi-agent debate and disagreement-based active learning
are adjacent, but the method should be framed as latent conditional
communication.

Implementation/GPU cost: medium. Requires paired traces and a disagreement
feature extractor.

Honest PASS probability: 6-14%. Risks: disagreement may correlate with both
models being wrong; can add noise. Composability: medium with M42/M45.
Suitability: workshop.

### M50: Byte-Level Distillation Bridge

Mechanism: train a receiver adapter that consumes source packets expressed over
byte spans instead of token positions. This is stronger than remapping: the
receiver is explicitly trained to interpret byte-indexed latent summaries.

Empirical motivation: real tokenizer sweep says Qwen2.5->Qwen3 is effectively
tokenizer-identical, but Qwen->Mistral/Phi3 has decoded overlap around `0.80`
and boundary F1 `0.93-0.95`. The cross-family gate needs an interface that is
not accidentally same-tokenizer.

Prior-art adjacency: Cross-Tokenizer LLM Distillation through a Byte-Level
Interface (arXiv:2604.07466). The novelty is cross-model latent packet
consumption, not distillation alone.

Implementation/GPU cost: medium. Needs cross-family pair setup and byte-span
alignment.

Honest PASS probability: 7-16%. Risks: may solve tokenizer mismatch while
leaving latent mismatch untouched. Composability: high with M39/M41.
Suitability: workshop unless it creates cross-family lift.

### M51: Attribution-Path Packet

Mechanism: extract sparse attribution-path features from the source trace and
send only path IDs/weights that target-side probes map to reasoning-relevant
updates.

Empirical motivation: reviewers want interpretability, and many dense repairs
are non-additive. A path packet can make help/harm and failure modes auditable.

Prior-art adjacency: Transformer Circuits attribution graph methods
(https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and
sparse feature/crosscoder work. This is high-risk because it depends on
interpretability tooling quality.

Implementation/GPU cost: high. Needs attribution extraction, probes, and
bridge training.

Honest PASS probability: 5-12%. Risks: tooling overhead, fragile at small
model scale, may not improve accuracy. Composability: medium with M40/M45.
Suitability: workshop/interp venue unless accuracy lift is strong.

### M52: Matched-Rate KV Packet Baseline

Mechanism: create a strong matched-rate KV/cache compression baseline and add a
source-conditioned residual only if it beats the pure compression baseline.

Empirical motivation: reviewers will not accept latent transfer if the same
bytes spent on target KV/cache compression or prompt compression wins. The
rotalign KV row uses about `171661` bytes and only ties target with coverage
failure (`28/32`), while dynalign module replace uses about `680219` bytes for
`0.0938`; byte accounting is central.

Prior-art adjacency: KVQuant, KIVI, SmoothQuant, and related KV-cache methods.
This is mostly a baseline, but it can become a method if the residual is
clearly source-conditioned.

Implementation/GPU cost: low-medium. Strong byte accounting and existing
baselines are enough for first pass.

Honest PASS probability: 6-12% as a headline method; high value as a required
control. Risks: baseline may beat the method and kill the story. Composability:
required for all serious variants. Suitability: baseline/appendix.

### M53: Multi-Source Consistency Receiver

Mechanism: use two cheap heterogeneous sources and let the target consume only
agreement/consistency features, not either source's full answer. The receiver
learns when independent latent evidence agrees.

Empirical motivation: a single weak source is often wrong, but latent wins are
not answer copying. Agreement may filter source poison while preserving hidden
reasoning clues.

Prior-art adjacency: multi-agent LLM and C2C-style communication. Novelty is
source-private latent consistency at fixed byte budget.

Implementation/GPU cost: high because it doubles source inference and controls.

Honest PASS probability: 5-11%. Risks: too expensive, hard to beat structured
text/debate baselines, weaker ICLR novelty. Composability: medium with M42.
Suitability: workshop only unless unexpectedly strong.

### M54: Dynamic Budget Latent Packet

Mechanism: allocate larger latent packet budgets to hard positions/examples
using target uncertainty and packet reconstruction difficulty. Keep average
bytes fixed through a global budget.

Empirical motivation: OutlierMigrate M11b suggests budget scaling can matter:
Granite-Small top5 recovery `0.4493` beat static-top10 by `0.5011` median, but
with wide CI and no accepted Nemotron replication. Use this only as an analogy:
fixed tiny budgets may be too small for hard target states.

Prior-art adjacency: adaptive quantization and prompt/KV compression. Needs
matched average-byte controls to avoid being "more bytes wins."

Implementation/GPU cost: medium. Requires byte scheduler and budget sweeps.

Honest PASS probability: 5-13%. Risks: budget effects may not transfer from
quantization to communication; reviewers will punish unmatched rate.
Composability: high with M37/M41/M52. Suitability: workshop.

### M55: Structured-Text Plus Latent Hybrid

Mechanism: use a strict structured text trace (answer candidates, quantities,
operations) plus a tiny latent residual packet. The latent component must beat
structured text at matched or accounted extra bytes.

Empirical motivation: freeform text relay is too weak and harmful; reviewers
will compare against structured text/tool traces, not just raw source output.
If latent only beats freeform text, the claim is weak.

Prior-art adjacency: chain-of-thought/tool traces and prompt compression. This
is more of a threat-model hardening method than a pure latent method.

Implementation/GPU cost: low-medium. Build structured baselines first; then
add latent residual.

Honest PASS probability: 6-14%. Risks: structured text may dominate latent,
turning this into a negative result. Composability: necessary baseline for
M37/M41. Suitability: control for ICLR; possible workshop if hybrid wins.

### M56: Abstaining Receiver Selector

Mechanism: train a receiver to accept, reject, or defer the latent packet based
on source-conditioned features, not just target confidence. Report coverage,
risk, help/harm, and paired utility.

Empirical motivation: GSM8K70 showed live row `8/70` and oracle `10/70`, so
there is some selector headroom outside GSM8K32. Many methods have zero or one
loss; abstention could preserve gains while avoiding harms.

Prior-art adjacency: selective prediction and verifier-controlled generation.
Not novel enough alone unless the source-conditioned features are essential.

Implementation/GPU cost: low. Can be trained after candidate rows exist.

Honest PASS probability: 6-12%. Risks: selector-only contribution; small
headroom; can overfit. Composability: high with all methods. Suitability:
appendix/control, not headline.

## Top-5 Recommendation

1. **M37 Conditional innovation resampler**. Highest expected value because it
directly explains the non-copy wins, controls interference, and creates a
clear source-conditioned claim.
2. **M41 Rate-distortion query bottleneck**. Best clean interface pivot and the
most reviewer-legible way to compare against C2C and structured text.
3. **M39 Sequence-aligned byte sidecar**. Best cross-family/interface bet
because it is already the strongest low-shot toy under tokenizer-like stress.
4. **M38 C2C-compatible dynalign fuser**. Required strong baseline; becomes a
method only if dynalign residual adds lift or reduces bytes over C2C.
5. **M40 Relative-anchor sparse packet**. Best interpretability/geometry branch
that is not already killed by fixed rotations, but should be gated behind the
connector/interface bets.

Next exact gate before any method promotion: freeze a larger same-family slice
with seed repeats and paired intervals, include structured text and C2C, then
run one strict cross-family pair. GSM8K32 should remain a reproducibility and
falsification smoke only.

## Citation Pointers

- Cache-to-Cache, arXiv:2510.03215.
- Flamingo / Perceiver Resampler: https://arxiv.org/abs/2204.14198.
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597.
- Gist Tokens: https://arxiv.org/abs/2304.08467.
- Relative Representations: https://openreview.net/forum?id=SrC-nwieGJ.
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987.
- Cross-Tokenizer LLM Distillation through a Byte-Level Interface:
  https://arxiv.org/abs/2604.07466.
- TokAlign: https://arxiv.org/abs/2506.03523.
- Vision Wormhole: https://arxiv.org/abs/2602.15382.
- KVQuant: https://arxiv.org/abs/2401.18079.
- KIVI: https://arxiv.org/abs/2402.02750.
- SmoothQuant: https://arxiv.org/abs/2211.10438.
- Transformer attribution graph methods:
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html.
