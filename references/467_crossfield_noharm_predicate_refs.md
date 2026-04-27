# Cross-Field No-Harm Predicate References

Date: 2026-04-27

## Problem Helped

The live blocker is target-self harm. Recent source-derived probes sometimes
recover clean source-necessary IDs, but they overwrite too many examples that
the target already solved. The next method must act more like an erasure-aware
side-information decoder: it should abstain unless source information supplies
evidence that is both source-specific and target-preserving.

## Sources And Mechanisms

1. Neural Distributed Source Coding
   - Source: `https://arxiv.org/abs/2106.02797`
   - Helps with: replacing brittle numeric residues with learned predicate or
     codebook bits.
   - Mechanism: train a compact source encoder whose message is decoded only
     with target-side context.
   - Experiment change: revive syndrome only as learned semantic predicates,
     not as hash residues.
   - Role: inspiration and baseline.

2. Distributed Deep JSCC with Decoder-Only Side Information
   - Source: `https://arxiv.org/abs/2310.04311`
   - Helps with: injecting source information without overwriting the target.
   - Mechanism: decoder-only side information enters at multiple receiver
     stages, suggesting a zero-init gated target receiver rather than a prefix
     replacement.
   - Experiment change: learned MPS branch should use zero-gated target-side
     receiver stages with source-destroying controls from the first gate.
   - Role: inspiration.

3. Side Information Vending Machine
   - Source: `https://arxiv.org/abs/1109.6665`
   - Helps with: deciding when to spend communication or repair calls.
   - Mechanism: the message selects a decoder-side action, not necessarily an
     answer. Actions include target-alone, self-repair, candidate expansion, or
     source-predicate decode.
   - Experiment change: add an action-gated branch only after a source-derived
     signal exists.
   - Role: theory support and method framing.

4. Semantic Entropy Probes
   - Source: `https://arxiv.org/abs/2406.15927`
   - Helps with: source-fault detection before applying source corrections.
   - Mechanism: cheap hidden-state probes predict semantic uncertainty and
     correctness risk.
   - Experiment change: once model execution is safe, collect source-side
     uncertainty features and reject high-risk source messages.
   - Role: source-fault detector inspiration.

5. BLIP-2 Q-Former
   - Source: `https://arxiv.org/abs/2301.12597`
   - Helps with: target-preserving learned connectors.
   - Mechanism: learned query tokens extract relevant source information from a
     frozen encoder.
   - Experiment change: zero-init gated query bottlenecks should compare MLP,
     fixed query, and Q-Former-style extraction under the same controls.
   - Role: baseline and connector inspiration.

6. Cache-to-Cache
   - Source: `https://arxiv.org/abs/2510.03215`
   - Project page: `https://fuvty.github.io/C2C_Project_Page/`
   - Helps with: fair cross-LLM communication baseline and decision surface.
   - Mechanism: direct KV-cache projection/fusion between LLMs.
   - Experiment change: any positive branch must either compete with C2C on
     accuracy or offer a clear byte/latency/TTFT systems tradeoff.
   - Role: baseline.

7. Multi-Way Representation Alignment
   - Source: `https://arxiv.org/abs/2602.06205`
   - Helps with: avoiding another global RotAlign tweak.
   - Mechanism: shared orthogonal universes and geometry-corrected Procrustes
     alignment suggest local/anchor-relative gauge fixing.
   - Experiment change: revive geometry only as anchor-relative sparse
     difference atoms with source-difference zeroing controls.
   - Role: inspiration and ablation design.

## Next Experiment Impact

The CPU no-harm gates failed on current artifacts, so this literature does not
justify another threshold sweep. It changes the next branch selection:

- Prune shallow numeric/hash syndrome and source-text feature routers.
- Revive source-predicate decoding only with learned semantic predicates and
  erasure-aware abstention.
- Keep zero-init target-preserving query bottlenecks as the next MPS branch
  after PID `31103` clears.
- Require source-fault detection before applying any non-abstaining source
  correction.

## Classification

Baseline: C2C.

Inspiration: neural distributed source coding, DeepJSCC-WZ, Q-Former,
multi-way alignment.

Theory support: side-information vending machine.

Ablation design: source-fault detectors, zero-source/shuffled-source/random
predicate/slots-only controls, target-preserving erasure abstention.
