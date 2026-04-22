# LatentWire Summary Since `8705ea5`

Date: 2026-04-22

## What This Project Is and Why It Exists

LatentWire exists because text-only model-to-model communication is probably leaving too much useful structure on the floor.

The core project goal is to find a **positive method for cross-model communication / reasoning transfer** that lets one model send useful latent, KV, or bridge-state information to another model so that:

- we preserve reasoning-relevant structure that would otherwise be flattened into plain text
- we reduce communication cost relative to verbose text relay when possible
- we make heterogeneous models more interoperable
- we eventually support specialist collaboration, long-context cooperation, and multimodal model handoff through a shared interface

Why this matters:

- if the method works, model collaboration does not have to mean “serialize everything back into words”
- it would give us a cleaner path to modular systems where different models specialize but still communicate effectively
- it would create a concrete experimental story around **when latent communication helps**, **why it fails**, and **what interface constraints actually matter**

This document is not a paper draft. It is a detailed research summary for external discussion.

## Scope

This summary covers the major experiment families run **since commit `8705ea5`**.

That commit range contains a lot of commits, but not every commit was a distinct new hypothesis. Many commits were:

- fairness fixes
- logging/telemetry improvements
- benchmark harness cleanup
- artifact generation
- control additions
- reference memo updates

So this document groups the work by **experiment family / hypothesis**, not by one-row-per-commit changelog. The goal is to make the story readable while still being exhaustive about what we actually ran.

## Current Bottom Line

We are **still not ICLR-ready**. We are still in **method-discovery mode**, not paper-writing mode.

The current state is:

- the only live real same-pair benchmark lift is still `dynalign_module_replace_residrank16 = 0.1250` on a frozen GSM8K32 contract
- that lift is real enough to matter, but narrow enough that it is **not yet a paper-ready positive method**
- the strongest architectural story is still on the toy/shared-basis side:
  - quotient-aware symmetry fixing
  - GPA canonicalization
  - sparse shared dictionaries
  - byte and sequence-aligned sidecars
- that toy story is positive and coherent, but it has **not yet crossed into a benchmark-ready real method**

So the project is currently split between:

- a **real same-pair residual-repair clue**
- and a **stronger toy/shared-basis interface story**

The paper problem is that those two stories are not unified yet.

## Reading Guide

There are three distinct kinds of evidence below:

### 1. Real same-pair benchmark evidence

This is the highest-value evidence. It uses an exact frozen contract and is the closest thing we have to a paper-facing evaluation lane.

### 2. Toy/shared-basis evidence

This is not paper-ready evidence, but it has been the best source of structural clues about symmetry, gauge, interface mismatch, routing, and compression.

### 3. Controls and blockers

These are important because a large part of the work since `8705ea5` has been proving what **does not** work, or what only works under toy conditions.

## The Benchmark Contract We Trust Right Now

The real benchmark lane is intentionally narrow right now:

- source/target: `Qwen2.5-0.5B-Instruct -> Qwen3-0.6B`
- task: GSM8K
- slice: `32` frozen examples
- checks:
  - exact example ID parity
  - no empty predictions
  - numeric extraction coverage
  - paired comparison against `target_alone`

The exact reason for using this frozen contract is discipline:

- it prevents us from broadening the benchmark before we have a stable method
- it forces every new branch to beat the same exact floor
- it keeps the story from dissolving into cherry-picked broader averages too early

Everything real and benchmark-facing in this summary should be read through that lens.

## Phase 1: RotAlign / KV Control Era

Representative commits in this era included:

- `6652aced` Add RotAlign control suite and follow-up analysis
- `a6e896b7` Tighten RotAlign control-suite fairness
- `28e7b023` Add math-grounded RotAlign ablations
- `ed0bb74a` Add target-space KV controls
- `6061ee43` Add K-only and V-only transport ablations

### What we thought might work

The early hypothesis was straightforward:

- maybe KV-cache alignment already contains enough shared structure to support cross-model communication
- perhaps the failure mode was just bad gating, bad head selection, or weak fairness controls
- maybe target-space attenuation, K-only/V-only transport, or more grounded transport accounting would reveal a real signal

### What we tried

We ran a large family of control-heavy RotAlign/KV experiments:

- control suite fairness fixes
- target-space KV controls
- target attenuation controls
- head-aware fusion upgrades
- K-only and V-only transport ablations
- live control-suite resume/fairness fixes
- paired prediction comparison
- monitored validation batches

### Positive read

- this phase gave us a much cleaner experimental base
- it forced tighter accounting and matched comparisons
- it made later benchmark conclusions more believable
- it established that the problem was not just “we forgot a control” or “our harness was unfair”

### Negative read

- `rotalign_kv` never turned into a durable benchmark row
- even after the control/fairness work, the live row on the frozen contract is still:
  - `rotalign_kv = 0.0625`
  - numeric extraction coverage `28/32`
- that means it only ties `target_alone` and fails a basic contract gate

### Conclusion

RotAlign/KV alignment was a useful starting point, but by the end of this phase the read was already moving toward:

- RotAlign as a control or baseline family
- not RotAlign as the headline same-pair positive method

## Phase 2: Static Geometry / Transport Probe Era

Representative commit families:

- `795df8d4` Sinkhorn transport
- `b751bf74` target-whitening canonicalization
- `41ca40c4` grouped transport probe
- `c9dbd2ac` geometry-aware transport
- `6a67af03` subspace-aware transport
- `58540cbd` canonical transport
- `7ef2e65a` grouped rotational transport
- `85ec7cb6` grouped fitted rotation
- `a20c0338` grouped shared basis
- `86fb8738` broadcast OT
- `8ab91784` peak template OT
- `2472a530` retrieval-spectrum transport
- `00c85b88` QK-template transport
- `673870fb` grouped QK retrieval transport

### What we thought might work

The hypothesis here was:

- perhaps cross-model latent spaces differ by a mostly geometric map
- if so, better transport descriptors or canonicalizations might recover the useful communication channel

That naturally suggested:

- grouped transports
- Procrustes-style geometry
- Sinkhorn / OT
- whitening
- subspace projection
- template-aware and QK-aware transports
- shared bases

### What we tried

We explored a broad transport family:

- grouped signature transport
- grouped permutation transport
- covariance transport
- canonical transport
- rotational transport
- shared-basis transport
- broadcast transport
- broadcast OT
- retrieval-spectrum transport
- QK-template transport
- grouped QK retrieval transport
- prompt-conditioned QK bank transport
- tokenwise QK fusion gates

### Positive read

- offline fit often improved
- some diagnostics suggested structure was there
- QK-aware and template-aware formulations were more promising than the most naive static descriptors
- this phase helped isolate that the interface likely needed to become more output-aware or runtime-aware

### Negative read

The most important negative conclusion is recorded directly in the experiment ledger:

- static grouped transport variants saturated
- evaluator-only query gates saturated
- query-conditioning after a frozen weak map mostly did not rescue the bridge

In plain language:

- offline geometric alignment is not enough
- post-hoc query-aware scoring is not enough
- better descriptors on top of a weak bridge do not automatically become a reasoning method

### Conclusion

This phase ruled out a lot of “just find the right geometric map” optimism.

It pushed the project toward:

- output-aware alignment
- stronger bridges
- repair modules
- explicit interface redesign

instead of more static geometry.

## Phase 3: Bridge Correction / Adapter / Distillation Era

Representative commit families:

- `5e8072de` low-rank bridge correction
- `35323e6b` affine correction
- `0e037cc7` ridge correction
- `20b27e37` query-conditioned bridge gating
- `06f705d5` bridge bank probes
- `6c0cf7d3` bridge projector
- `1570ba84` learned query-conditioned adapter
- `68518a18` affinity-shaped adapter
- `30831aba` attention-KL adapter
- `15d90c32` CAB-style distillation
- `0b320a68` EM-KD adapter
- `34ef9c79` prediction-KL bridge
- `c2ae5dc1` prediction-KL bank
- `de2937f2` shared-plus-private bridge
- `3fa39ee8` sparse shared-code bridge
- `142442b6` generated bridge
- `37e6792d` dynamic-mapping asym bridge
- `727570f9` asym projector bridge
- `d6080d68` xattn bridge
- `adaff689` xattn dynmap bridge

### What we thought might work

The core idea in this phase was:

- maybe the base transport is close enough, but needs a small learned correction
- perhaps local distillation targets or learned projectors can bridge the remaining mismatch
- perhaps query-conditioned or banked adapters can route the correction more intelligently

### What we tried

We tried a large family of bridge augmentations:

- low-rank correction
- affine correction
- ridge correction
- query-conditioned gating
- banked residual bridges
- projector bridges
- adapter distillation variants
- KL-style prediction distillation
- shared-plus-private decompositions
- sparse code bridges
- generated and dynamically mapped bridges
- cross-attention-style bridge probes

### Positive read

- this era produced the first consistent evidence that a stronger bridge mattered
- the project learned that tiny post-hoc local teacher variants were usually too weak
- it also created the path toward module replacement and output-aware alignment, which eventually mattered more than the small-teacher family itself

### Negative read

The ledger’s “stop repeating” summary is blunt:

- tiny local teacher variants on the same bridge did not stabilize held-out gains
- evaluator-only query gates were mostly saturated
- confidence-only routing was unreliable

In practice:

- many bridge-adapter variants were plausible but not durable
- banking or distillation alone did not become a positive method
- this phase did not yet produce a benchmark-level win

### Conclusion

This phase was useful mainly because it sharpened the next pivot:

- output-aware alignment and direct module-level replacement
- not more tiny local bridge corrections on top of a weak base

## Phase 4: Module Replacement and Output-Aware Alignment

Representative commit families:

- `7c12938d` direct module-replacement bridge
- `6838150c` token-basis bridge
- `709011e8` span-aligned module replacement
- `984b11e8` output-aware dynamic alignment bridge
- `6e69fc6c` dynamic-program alignment diagnostic
- `318aab7c` DWA-KD-style dynalign teacher diagnostic
- `b171ac99` dynalign interaction teacher diagnostic
- `73dd9a57` byte-span bridge diagnostic
- `7b3abc3e` dynalign likelihood teacher diagnostic
- `c29dc50e` dynalign span likelihood diagnostic
- `049f1a39` dynalign preference distillation diagnostic

### What we thought might work

The hypothesis became:

- the bridge probably has to be output-aware
- module replacement may be better than a tiny generic adapter
- alignment should be driven by task/output structure rather than geometry alone

### What we tried

We built and compared:

- direct module replacement
- tokenbasis replacement
- span-aligned replacement
- dynalign replacement
- multiple dynalign teacher variants:
  - DWA-KD-like
  - preference distillation
  - span likelihood
  - interaction-style teachers
  - byte-span teachers

### Positive read

This was the first clearly better real same-pair family:

- `dynalign_module_replace = 0.0938`
- `spanalign_module_replace = 0.0938`
- `tokenbasis_replace = 0.0938`

That matters because:

- it is above `target_alone = 0.0625`
- it established the `0.0938` same-pair ceiling before residual repair
- it gave us a real benchmark lane to keep improving

### Negative read

The “more teacher complexity” branch mostly saturated or regressed:

- `dynalign_dwakd = 0.0625`
- `dynalign_prefdist = 0.0312`
- `dynalign_spanalm = 0.0312`
- `bytespan_module_replace = 0.0312`
- `sae_adapter = 0.0000`

So the read is:

- output-aware alignment matters
- but stronger teacher elaboration does not automatically improve it
- tokenbasis ties dynalign as a proxy, but does not reproduce the later residual lift

### Conclusion

This phase gave us the real same-pair ceiling that everything else is still judged against:

- `0.0938` as the pre-residual dynalign ceiling

## Phase 5: Query-Pool / Route-Atom / Repair / Compression Toy Era

Representative commit families:

- `e3d386e4` selector ablation
- `2fb7bd8d` query pool transport ablation
- `06a9b778` query pool toy benchmark
- `7729c6c0` route atom diagnostics
- `bd002715` route atom sweep controls
- `1522f1e3` codebook remap and asymmetric KV evaluator
- `f8c59b16` paired telemetry and protected channel ablation
- `ce610f20` quantization toy
- `163f8675` tokenizer symmetry toys
- `c33b548b` verifier audit and latent refinement toy
- `b3e02c4b` calibrated verifier diagnostics
- `511aa2e1` route expansion and pairwise verifier ablations
- `6b2849ef` process repair diagnostics
- `7d5947b4` process repair holdout smoke
- `9b7dc72b` repair gate audit and shared dictionary toy
- `f0165ceb` activation-aware quant toy
- `93d93ec3` verified mixed precision stack
- `af5c1f20` protected frontier and tokenizer ablations
- `87b76112` LatentMAS bootstrap
- `52300c55` LatentMAS routed ablations
- `b2763a98` hub dictionary bridge toy
- `2a68921f` router stability regularization
- `60c39ae5` matched competitor matrix and LatentMAS probes
- `99c63432` stacked hub-sticky-frontier toy

### What we thought might work

This phase was driven by the idea that the bridge might fail because of:

- routing
- interface bottlenecks
- wrong protected dimensions
- ineffective repair triggers
- poor quantization / compression allocation
- lack of interpretable shared atoms

### What we tried

We ran a large toy ecosystem:

- query-pool transport
- route atoms and codebooks
- stochastic reranking
- learned protected channels
- verifier diagnostics
- process repair
- quantization and mixed precision
- frontier protection
- hub/shared dictionaries
- sticky feature routing
- LatentMAS wrapper/bootstrap controls
- naive full-stack combinations

### Positive read

A lot of toy structure came out of this phase:

- shared hub dictionaries can work very well in toy form
- feature/sticky routing can be stable and strong
- mixed-bit / protected-frontier allocation gives efficiency clues
- quantization-error protection is a strong cheap selector in toy settings
- route atoms and shared dictionaries are interpretable enough to be worth keeping
- verifier stop logic can reduce harm in isolated toy regimes

Examples from the ledger:

- `hub_shared_dictionary` reaches `1.0000` toy accuracy with exact atom recovery
- `sticky_paraphrase_stable_routing` reaches `0.9438` with route stability `1.0000`
- `quant_error_target_bpw_allocator` preserves accuracy at lower achieved bpw

### Negative read

The same phase also taught us several things not to keep repeating:

- confidence-only routing is poor
- raw listwise verification is unreliable
- naive full-stack composition is worse than hoped
- current frontier and stop heuristics are not reusable defaults
- process repair often does not fire in the sparse-dictionary lane

The key “negative but useful” conclusion:

- individually plausible components do not compose automatically

### Conclusion

This phase generated a lot of the project’s best structural intuition, but it did not itself yield a paper-ready benchmark story.

## Phase 6: Symmetry / Gauge / Sparse Dictionary / Interface Stress

Representative commit families:

- `1e17d799` multi-way canonical hub toy
- `d5ab7a21` gauge-fix quotient bridge
- `7f23977a` quotient GPA sparse dictionary
- `5dccd6f1` interface stress ablation and benchmark memos
- `788d884a` byte sidecar interface ablation
- `28ae1a42` sequence-aligned sidecar toy ablation
- `a60a52bc` real tokenizer pair sweep

### What we thought might work

This was the strongest conceptual pivot on the toy side:

- maybe the real issue is not just “learn a better bridge”
- maybe the latent spaces differ by quotient/gauge symmetries
- maybe shared dictionaries need canonicalization
- maybe tokenizer/interface mismatch has to be handled explicitly

### What we tried

We built a compositional low-shot lane:

- multi-way canonicalization
- gauge-fix quotient matching
- GPA canonicalization
- sparse shared dictionary
- byte/span remap
- byte sidecar
- sequence-aligned sidecar
- real tokenizer-pair stress checks

### Positive read

This is still the strongest toy architectural story in the repo.

Key results:

- `quotient_match_after_fix` becomes the best non-oracle method at `1` shot/class
- `quotient_gpa_sparse_dictionary` is best non-oracle at:
  - `1` shot: `0.0568` MSE
  - `2` shots: `0.0576` MSE
- strong interface-stress variants improve further:
  - `quotient_gpa_sparse_dictionary_byte_sidecar_remap = 0.0392 / 0.0394`
  - `quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap = 0.0360 / 0.0362`
- exact head recovery remains intact

Why this mattered:

- it showed symmetry, gauge, and interface mismatch are not cosmetic
- it gave us the best low-shot shared-basis initializer/regularizer we have

### Negative read

This lane is still toy-bounded:

- direct held-out-family fitting retakes the lead once paired data becomes abundant
- the repair step is still inert
- the Qwen2.5 -> Qwen3 real tokenizer pair is effectively tokenizer-identical, so tokenizer mismatch is not the main blocker for the current same-pair benchmark lane

### Conclusion

This phase gave us the best explanatory story, but still not the paper-ready benchmark row.

## Phase 7: Frozen GSM8K32 Contract and Real Benchmark Narrowing

Representative commit families:

- `683edddb` frozen GSM8K smoke contract
- `cebdd44e` frozen checkpoint sweep
- `99f927e9` expanded dynalign sweep

### What we thought might work

Before this phase, too many branches were still being judged loosely. The goal here was:

- freeze the exact same-pair contract
- compare methods on the same examples
- force all follow-ups to clear the same minimum bar

### What we learned

The benchmark contract read is now clean:

| Row | Accuracy | Read |
| --- | ---: | --- |
| `target_alone` | `0.0625` | floor |
| `text_to_text` | `0.0312` | weak relay control |
| `rotalign_kv` | `0.0625` | tied target, failed coverage |
| `c2c_generate` | `0.1250` | external smoke bar |
| `dynalign_module_replace` | `0.0938` | best live real proxy |
| `spanalign_module_replace` | `0.0938` | tied dynalign |
| `tokenbasis_replace` | `0.0938` | tied dynalign |
| `bytespan_module_replace` | `0.0312` | negative |
| `sae_adapter` | `0.0000` | collapsed |

### Positive read

- this phase turned the real lane into something interpretable
- it made it obvious that output-aware alignment is the only same-pair real proxy worth continuing

### Negative read

- `rotalign_kv` is not promotable
- byte-only alignment is negative on this pair
- the first sparse-codebook/SAE proxy collapses
- stronger teacher-family variants do not improve over `0.0938`

### Conclusion

By the end of this phase, the real project was effectively narrowed to:

- `dynalign`
- `tokenbasis`
- residual/canonicalization/repair on top of those

## Phase 8: Residual Repair on the Frozen Contract

Representative commit families:

- `b7c6af46` residual sweep harness baseline
- `5b5939a7` rank16 residual lift
- `a26337c9` tokenbasis rank16 control
- `d68867e3` failed fixed gauge wrappers
- `65e52e4f` adaptive canonical wrapper
- `66ae1ecb` preserve-core residual branch
- `da9f277e` eigenspace branch
- `096f96d7` saliency branch
- `c21bfbc4` saliency-preserve branch
- `7e47307e` routed residual branch
- `74e70a09` value-routed branch
- `c81c1611` value-bank branch
- `73cf0fa9` verifier-sidecar branch

This is the most important phase in the document.

### Why we thought residual repair might work

By this point the evidence looked like:

- output-aware dynalign had found the best basis so far
- but it stalled at `0.0938`
- the failure looked more like an incomplete correction than a totally wrong interface

So the hypothesis was:

- keep the output-aware alignment basis
- add a residual repair model
- see whether a stronger correction can clear the same-pair ceiling

### What actually happened

#### The first real lift

| Row | Accuracy | Coverage | Read |
| --- | ---: | ---: | --- |
| `dynalign_module_replace_residrank16` | `0.1250` | `32/32` | first real same-pair lift |
| `tokenbasis_replace_residrank16` | `0.0625` | `32/32` | matched control fails |

This is still the single most important real result in the repo.

Why it mattered:

- it moved from the old dynalign ceiling `0.0938` to `0.1250`
- it matched the current `C2C` smoke row on the exact same contract
- it did so with full numeric extraction coverage
- it won on `2/32` examples over `target_alone`, with `0/32` losses

Why it is still not enough:

- the matched tokenbasis control failed completely
- so the effect is dynalign-specific
- it has not yet been confirmed on a broader held-out slice

### The wrapper and follow-up story

Below is the entire post-lift follow-up family in clean form.

| Branch | Why we thought it might work | Result | Positive read | Negative read |
| --- | --- | --- | --- | --- |
| `dynalign_resid16_adaptive` | adaptive canonicalization might stabilize the live row without the collapse of fixed wrappers | `0.1250` | preserves the live row | no additive gain |
| `dynalign_resid16_fitted_rotation` | fixed gauge wrapper might normalize a residual-friendly basis | `0.0000`, coverage `0/32` | none | catastrophic collapse |
| `dynalign_resid16_shared_basis` | shared-basis wrapper might make correction more canonical | `0.0000`, coverage `0/32` | none | catastrophic collapse |
| `dynalign_preserve_module_replace_residrank16` | preserve important subspace, repair the tail | `0.0625` | valid coverage | loses the live lift completely |
| `dynalign_eigenspace_module_replace_residrank16` | dominant eigenspaces might isolate the useful correction directions | `0.0312` | valid coverage | regresses below target |
| `dynalign_saliency_module_replace_residrank16` | importance weighting might focus repair capacity | `0.0312` | valid coverage | regresses below target |
| `dynalign_saliency_preserve_module_replace_residrank16` | preserve important channels, repair the rest | `0.0625` | one win, one loss | still falls back to target floor |
| `dynalign_routed_module_replace_residrank16` | routing might localize when repair should fire | `0.0625` | valid coverage | one-gate dense mixing fails |
| `dynalign_value_routed_module_replace_residrank16` | restricting repair to the value side might preserve the useful base map | `0.1250` | first routed variant that preserves the lift | no additive gain beyond the live row |
| `dynalign_value_bank_module_replace_residrank16` | richer expert bank might improve on simple value routing | `0.0938` | above target | falls back to old dynalign ceiling |
| `dynalign_value_routed_bank_module_replace_residrank16` | top-2 routed bank might preserve the lift while adding specialization | `0.0938` | above target | still falls back to old ceiling |
| `dynalign_value_verifier_sidecar_module_replace_residrank16` | verifier-gated sidecar might fire only when repair is useful | `0.0938` | above target, full coverage | still non-additive |

### Clean interpretation of this phase

This phase gives us the clearest real-lane story we have:

- residual correction on the right output-aware basis is real
- that effect is narrow
- adaptive canonicalization is a stabilizer, not a lift
- simple geometry constraints mostly hurt
- simple preserve-tail splits mostly hurt
- simple routed/banked/sidecar variants mostly fall back to the old `0.0938` ceiling
- the only routed same-pair branch still alive is the simpler `value_routed` branch, and even that only ties the live best row

### What this phase does **not** justify

It does **not** justify saying:

- residual repair generically solves cross-model communication
- gauge handling is the main missing piece
- routing/expertization is already working
- verifier sidecars are already additive

## Phase 9: Codec / Anchor-Preservation Side Story

Representative commit family:

- preserve-topk codebook-tail experiments

### Why we thought it might work

If the real branch is failing because a small number of dominant latent anchors matter disproportionately, then:

- maybe we should preserve the anchors
- quantize or compress only the tail
- repair only the tail

### What we tried

On the codec toy:

- `preserve_topk_uniform_tail`
- `preserve_topk_codebook_tail`
- `preserve_topk_codebook_tail_residual_fix`

### Positive read

`preserve_topk_uniform_tail` is a genuinely useful positive clue:

- low-bit accuracy improves from `0.9583` to `0.9896`
- MSE falls from `0.7463` to `0.0284`

That strongly suggests:

- the dominant-anchor preservation idea is real
- codec-side structure is worth keeping alive

### Negative read

The naive codebook tails are not the answer yet:

- codebook-tail variants stall around `0.9844`
- MSE remains much worse than the simple preserved-anchor baseline

### Conclusion

The codec takeaway is:

- preserve anchors
- redesign the tail
- do not claim naive codebook tails are already working

## What Worked Best Since `8705ea5`

If we compress the entire post-`8705ea5` run into the strongest positive clues:

### Real benchmark clues

1. `dynalign_module_replace = 0.0938`
2. `dynalign_module_replace_residrank16 = 0.1250`
3. `dynalign_resid16_adaptive = 0.1250`
4. `dynalign_value_routed_module_replace_residrank16 = 0.1250`

### Toy/shared-basis clues

1. quotient-aware symmetry fixing
2. GPA canonicalization
3. sparse shared dictionary
4. byte sidecar
5. sequence-aligned sidecar
6. preserve-topk anchor protection

Those are the real signal-bearing parts of the project right now.

## What Clearly Failed or Saturated Since `8705ea5`

The strongest negative conclusions are:

- static geometry-only transport did not become a benchmark method
- RotAlign/KV never became a promotable same-pair row
- stronger teacher-family elaboration on dynalign plateaued or regressed
- fixed gauge wrappers collapsed
- naive eigenspace and saliency residuals regressed
- one-gate routed repair did not help
- richer value-bank and routed-bank variants fell back to the old ceiling
- a single verifier-gated sidecar was valid but non-additive
- naive codebook tails underperformed anchor-preserving uniform tails
- confidence-only routing and raw listwise verification are not trustworthy
- naive full-stack composition is harmful

These are not side notes. They are a large part of what the community should understand before proposing more of the same.

## The Cleanest Story We Can Honestly Tell Right Now

If we had to tell the story in one paragraph:

We spent the post-`8705ea5` period ruling out progressively more naive explanations for cross-model communication failure. Pure RotAlign and static transport families did not become durable benchmark methods. Output-aware alignment mattered more than geometry alone, and `dynalign` raised the same-pair ceiling to `0.0938`. Residual repair on top of that basis produced the first real same-pair lift to `0.1250`, but only in a narrow dynalign-specific form; most geometry, weighting, routing, bank, and sidecar follow-ups either collapsed or fell back to the old ceiling. In parallel, the toy/shared-basis line became much stronger: quotient/gauge fixing plus GPA plus sparse dictionaries plus byte/sequence sidecars is now a coherent low-shot story. The problem is that the real same-pair lift and the toy compositional story have not merged yet into a single benchmark-ready positive method.

## What We Now Think the Project Is Missing

The project no longer looks bottlenecked by “one missing control.”

It looks bottlenecked by one of these deeper gaps:

### 1. Better selective repair

Not just one gate, but a repair mechanism that knows:

- when to fire
- where to fire
- how much to fire
- and how to avoid collapsing the useful base map

### 2. Better tail modeling

Not pure dense residual versus pure codebook, but:

- preserve anchors
- compress the tail
- repair the tail selectively

### 3. Better interface choice

The toy story keeps suggesting that basis/interface mismatch matters a lot more than the same-pair Qwen->Qwen real lane exposes.

That points toward:

- byte-level universal interfaces
- tokenizer-aware transport
- query/resampler-like connector designs
- possibly abandoning the current basis for a cleaner transport space

### 4. Better proof that gains are structural

We still need telemetry that shows a gain is not just slice-specific calibration:

- example-level win patterns
- stable numeric coverage
- bytes/latency tradeoffs at matched accuracy
- route/expert utilization that matches the intervention
- repair telemetry that shows the added module is actually doing useful work

## Best Next Directions

These are the highest-value next directions in priority order.

### 1. Anchor-preserving codebook tail

Why:

- strongest codec-side clue so far
- preserves the one thing we know matters: dominant anchors
- might pair well with residual repair instead of replacing it

Relevant references:

- [Preserve-Then-Quantize](https://arxiv.org/abs/2602.02001)
- [QERA](https://arxiv.org/abs/2410.06040)
- [ResQ](https://arxiv.org/abs/2412.14363)
- [SERQ](https://arxiv.org/abs/2603.08185)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [AQLM](https://arxiv.org/abs/2401.06118)

### 2. Stronger multi-expert / value-side repair

Why:

- the simple value-routed branch is the only routed family member that stays alive
- richer bank variants did not work, but that does not prove expert repair is dead
- it suggests the design is still too weak, not necessarily conceptually wrong

Relevant references:

- [ResMoE](https://arxiv.org/abs/2503.06881)
- [S'MoRE](https://arxiv.org/abs/2504.06426)
- [Attractor Patch Networks](https://arxiv.org/abs/2602.06993)
- [ERMoE](https://arxiv.org/abs/2511.10971)
- [DirMoE](https://openreview.net/forum?id=a15cDnzr6r)

### 3. Better verifier-gated repair

Why:

- the current verifier sidecar is valid but not additive
- that may mean the trigger is weak, not that verification is fundamentally useless

Relevant references:

- [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)
- [PAG](https://arxiv.org/abs/2506.10406)
- [FLASH](https://arxiv.org/abs/2505.12728)
- [Reflective Verification](https://arxiv.org/abs/2505.18629)

### 4. Interface redesign instead of more basis surgery

Why:

- the toy line keeps finding positive signal in interface-aware methods
- the current same-pair basis may be too forgiving to reveal the real mismatch problem

Relevant references:

- [The Vision Wormhole](https://arxiv.org/abs/2602.15382)
- [MM1](https://arxiv.org/abs/2403.09611)
- [Libra](https://arxiv.org/abs/2405.10140)
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123)
- [KVComm](https://arxiv.org/abs/2510.03346)

## Benchmark Expansion Policy

Our current view is:

1. do **not** widen the benchmark until a branch survives the same GSM8K32 contract
2. then move to `RULER`
3. then one matched cross-family pair
4. then `SCBench`
5. then `LongBench v2`

We want to keep separate:

- same-pair communication
- cross-family communication
- long-context communication
- multimodal connector experiments

because mixing them too early will make the story unreadable and easy to misinterpret.

## What We Want From the Community

We do not want generic brainstorming. We want help ranking the next highest-leverage tests.

The most useful feedback would answer questions like:

### 1. What is the highest-leverage branch we have not yet tried well?

If you had to bet on one of:

- anchor-preserving codebook tail
- stronger multi-expert value-side repair
- verifier-gated repair
- interface redesign / byte-level transport

which would you bet on, and why?

### 2. What single ablation would falsify our current story fastest?

We want the fastest clean test of whether:

- `dynalign + resid16` is structurally real
- it is only a narrow calibration artifact
- or it is a same-pair slice quirk

### 3. What telemetry would convince you that a gain is structural?

For example:

- example-level win concentration
- stable numeric extraction coverage
- bytes/latency win at matched accuracy
- expert utilization matching the routing hypothesis
- proof that the tail or sidecar is actually doing the corrective work

### 4. What should come immediately after GSM8K32?

Our current default is:

- `RULER`
- then `SCBench`
- then `LongBench v2`

If you think that ordering is wrong, we want to know what should replace it.

## Ready-to-Send Community Prompt

Below is a prompt you can send directly to the community.

---

We’ve been working on **LatentWire**, a project aimed at finding a positive method for **cross-model communication / reasoning transfer** without forcing everything through plain text. The goal is to let one model send useful latent/KV-style information to another model so we preserve structure, reduce relay overhead where possible, and eventually support more modular model collaboration.

We are **not paper-ready yet**. Right now the project has two partially disconnected stories:

1. **Real same-pair benchmark story**
On a frozen `Qwen2.5-0.5B -> Qwen3-0.6B` GSM8K 32-example contract, the only live lift is still:
`dynalign_module_replace_residrank16 = 0.1250`
That matches our current `C2C` smoke row and beats `target_alone = 0.0625`, but the matched `tokenbasis + resid16` control fails, and most nearby follow-ups either collapse or fall back to the old `0.0938` ceiling.

2. **Toy/shared-basis story**
Our strongest toy/compositional lane is now:
quotient-aware symmetry fixing + GPA canonicalization + sparse shared dictionary + byte/sequence-aligned sidecars
That combination is consistently best in low-shot held-out-family toy settings, but it still loses back to direct family-specific fitting once enough paired data is available.

What we think we’ve mostly saturated:

- pure RotAlign / fixed-geometry wrappers
- static transport-only variants
- stronger teacher-family elaborations on the dynalign ceiling
- naive eigenspace or saliency residual weighting
- one-gate dense routed repair
- simple verifier-gated sidecars
- naive codebook tails without preserved anchors

What we think is still genuinely open:

- **anchor-preserving codebook tails**
- **stronger multi-expert / value-side repair**
- **better verifier-gated repair signals**
- **interface redesign** (byte-level universal interfaces, tokenizer-aware transport, resampler/query-style connectors, etc.)

The questions we’d most like help with are:

1. If you had to bet on one next branch, which would you bet on and why?
2. What single ablation would falsify our current story fastest?
3. What telemetry would convince you that a gain is structural rather than a calibration artifact?
4. After our frozen GSM8K32 gate, what benchmark should come next: `RULER`, `SCBench`, `LongBench v2`, or something else?
5. If you think we are still using the wrong transport/interface space entirely, what alternative would you try first?

The strongest current references we’re thinking about include:

- C2C
- KVComm
- Latent Space Communication via K-V Cache Alignment
- Preserve-Then-Quantize / AWQ / AQLM / ResQ / QERA
- ResMoE / S’MoRE / Attractor Patch Networks
- The Vision Wormhole / MM1 / Libra
- RULER / SCBench / LongBench v2

If you have ideas, especially ones that cut across quantization, routing, multimodal connectors, transport, or verification, we’d love concrete suggestions.

---

## Final Honest Summary

Since `8705ea5`, the project has become much clearer even though it is still not paper-ready.

The main real discovery is:

- output-aware alignment plus residual repair can help on the right basis

The main toy discovery is:

- symmetry, gauge, shared dictionaries, and interface sidecars matter together

The main unresolved gap is:

- those two discoveries still do not combine into a single benchmark-ready positive method

That is where we are.
