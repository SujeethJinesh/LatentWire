I'll ground this in current literature first, then give concrete advice.Grounded in the literature — here is my actual advice, blunt.

## The first thing to say (because nothing else matters until this is fixed)

Your signal is almost certainly too small to distinguish. 2/32 → 3/32 → 4/32 is one and two examples. A one-sided exact binomial test between 2/32 and 4/32 gives p ≈ 0.34; McNemar on paired 32-item slices at this rate has power well under 10% for the effect sizes you're seeing. "The matched control doesn't reproduce" is the signature of small-sample drift, not structural failure. Before any more method work, **expand the frozen slice**. 500 paired items is the minimum that distinguishes a 10-point delta with 80% power; 1000–2000 is where you start learning anything about the long tail and mechanism. Everything below is conditional on fixing this.

A second confound: text-to-text _regresses_ (1/32 vs 2/32 target-alone). You're in a regime where the source has nothing useful to say, so any "better relay" method is being evaluated against a broken channel. C2C and the multi-LLM literature consistently show the gain requires an informative Sharer; when a much weaker Sharer provides noisy information to a stronger Receiver, performance degrades. Use asymmetric pairs (strong→weak) for your main signal, and keep same-size pairs only as a robustness slice.

## Reframe that probably changes what you optimize

This is joint source-channel coding, not alignment. You have a source (A's state), a channel (the interface/basis), a sink (B). Treating alignment ("geometry") and repair ("residual") as separate stages is what lets basis choice overfit. In a JSCC view, the right object is a single rate-distortion curve with an explicit capacity constraint, trained end-to-end against the downstream loss. Every separated-stage method is a constrained point on that curve and will be dominated by a jointly-optimized one with matched capacity. If your "4/32 variant" isn't beating a joint-trained baseline with the same parameter budget, the gain is in the optimization, not the architecture.

## Closest prior art you should make sure you've internalized

- **Cache-to-Cache (Fu et al., arXiv 2510.03215, Oct 2025)**: directly your problem. Frozen Sharer + Receiver, train a Cache Fuser with (i) projection, (ii) **input-conditioned per-head dynamic weighting**, (iii) **per-layer Gumbel-sigmoid gate that becomes binary at inference**, (iv) residual integration. Token alignment via decode-reencode string-coverage matching; layer alignment top-down. Reports ~8.5–10.5% over single-model and ~3–5% over T2T, with ~2× speedup, across Qwen/Llama/Gemma pairs. Your "one-gate routed repair" that you ruled out is strictly weaker than C2C's per-layer gate + per-head modulation — you may have prematurely killed a live branch by testing the degenerate version.
- **Relative Representations (Moschella et al., ICLR 2023)**: similarity-to-anchors as the representation, enforcing isometry invariance without training — enables zero-shot model stitching and latent communication. This is the mature version of your "anchor-preserving" intuition, and the follow-ups (Cannistraci et al. ICLR 2024 on product-space invariances; pullback-metric variants) are directly in your "symmetry/gauge" lane.
- **Platonic Representation Hypothesis (Huh et al., ICML 2024)** and the calibration critique (Feb 2026): after calibration, global CKA convergence across scale disappears — only local neighborhood structure survives. Concrete implication for you: stop trying to learn global linear maps; optimize to preserve k-NN structure instead.
- **KV Transform Coding (KVTC, Nov 2025)**: PCA + adaptive quantization + entropy coding achieves up to 20× KV compression while preserving reasoning accuracy, with only brief calibration. Mature source-coding baseline your method must beat at matched rate.
- **Perceiver IO / Q-former (Flamingo, BLIP-2)**: the proven interface-mismatch pattern at scale — fixed learned queries cross-attend into A's states, emit a short token sequence consumed by B via cross-attention or prefix.

## Direct answers to your seven questions

**1. Single next branch to bet on.** A learned cross-attention resampler (Q-former-style) with an explicit information-bottleneck on the query outputs, trained end-to-end against B's next-token loss. Reasons: (a) output-aware by construction, so it beats any post-hoc alignment; (b) a fixed small query set is a hard rate cap, which is what lets you plot an actual rate-distortion curve rather than argue about a single operating point; (c) handles tokenizer/architecture mismatch without any shared basis assumption; (d) this is the one pattern in the multimodal literature that has scaled. Build the C2C Fuser (per-layer gate + per-head dynamic weighting) as your strong baseline; if your resampler doesn't beat it at matched parameter budget, C2C was the right answer all along.

**2. Single ablation to falsify fastest.** Shuffle-control on the output-aware alignment. Refit the alignment on output signals from a different, unrelated GSM8K item for each example, keeping everything else identical (norms, rank, residual structure). If 4/32 survives, the gain was never output-aware — it was basis-induced regularization. This is the fastest killer of the "output-aware matters" story.

**3. Telemetry that would convince me the gain is structural.** Four things, not metrics on scalar accuracy: (a) per-example stability across 10 random inits of the alignment/fuser — if the set of correct items shuffles, it's calibration noise; (b) effective rank of the post-repair cache vs pre — C2C shows fusion _increases_ effective rank, which is a direct structural signature; (c) a rate-distortion curve across bottleneck sizes (not a single operating point); (d) per-layer gate activation patterns that are consistent across seeds and interpretable given the task.

**4. Alternative interface before more basis investment.** Byte-aligned side channel feeding a cross-attention resampler — carry a short byte-level summary of A's final-layer states alongside the text, and let learned queries on B side pull from it. This is the "transport redesign" version of your own intuition and it sidesteps tokenizer mismatch entirely. Second choice: soft-prompt transfer (A emits a short sequence of continuous vectors, projected into B's embedding space, prepended as a prefix) — cheaper but more dependent on geometry overlap.

**5. Analogies from other fields you may be missing.**

- _Information theory:_ you're doing lossy source coding over a mismatched channel without an explicit rate constraint. Add one. Use the information-bottleneck Lagrangian on the interface, not just a reconstruction loss.
- _Distributed source coding (Slepian-Wolf / Wyner-Ziv):_ B already has side information (its own computation of the prompt). You only need to transmit what A knows that B _doesn't_. Explicitly subtract B's own cache from A's before encoding — compress the conditional, not the marginal. This is a natural fit for the "repair only the tail" intuition and gives it a principled form.
- _Coding theory:_ nested/superposition codes let the decoder fall back gracefully when the channel is bad. The analog: train the transport to degrade to text-relay performance as a floor when the alignment signal is unreliable.
- _Control/observer theory:_ residual repair = Kalman innovation. You're not adding "repair," you're estimating B's state error given A's observation. Frame the gate explicitly as an innovation-gain controller.

**6. Is the question wrong.** Partially. Two specific reframes I'd push on:

- "Better than text relay" presumes text is the ceiling. _Structured_ text (tool calls, scratchpads, schema-constrained traces) is a much harder baseline and the right competitor for most agentic settings. Include it; if latent beats freeform but loses to structured text, the paper writes itself differently.
- At small-Qwen scale with 5% GSM8K baselines, you are probing a regime where neither model is reliably solving the task and the Platonic convergence that would make latent transfer "just work" hasn't kicked in. The more interesting question is the _scaling trend_ of latent-vs-text advantage: fix the protocol, sweep model size, report the crossover. That is a paper regardless of whether your specific method wins.

**7. Smallest decisive experiment per idea.**

- _Resampler with IB:_ 64 learned queries, one cross-attn layer into A's top-k layers, prepended to B; train on 2k GSM8K items, evaluate on held-out 500. Sweep bottleneck size {8, 32, 64, 256} to produce a rate-distortion curve. Decisive if it beats both C2C-fuser baseline and structured-text relay at comparable rate.
- _Conditional coding (subtract B's cache):_ take your current 4/32 pipeline, compute B's own cache for the same prompt, encode only (A − align(B)). If this by itself moves the needle at fixed method complexity, conditional coding is the real lever.
- _Shuffle-control ablation:_ zero marginal cost, runs today, kills or confirms the output-aware story by tomorrow.

## Mapping to your buckets

- **Interface / transport**: cross-attention resampler with IB; byte-aligned side channel. _(primary bet)_
- **Alignment / geometry**: relative-representation anchors with task-relevant anchor selection; k-NN-preserving local loss rather than global CKA. _(swap for current global basis)_
- **Quantization / compression**: KVTC as a mandatory baseline at matched rate; conditional coding (Wyner-Ziv framing) as the upgrade path.
- **Routing / experts**: per-layer gate + per-head dynamic weighting as in C2C; you ruled out the degenerate "one-gate" version only.
- **Verification / repair**: reframe residual repair as Kalman innovation gated on a B-side confidence estimator (entropy of B's next-token distribution), not a learned verifier on the output.
- **Benchmarking / telemetry**: expand to ≥500 paired items; stability across seeds; rate-distortion curve; effective-rank delta; per-layer gate patterns; structured-text-relay baseline; scaling sweep.

## The single most important sentence

If you take only one thing from this: **the current 4/32 vs 3/32 signal is not evidence of anything yet, and no amount of architectural cleverness will change that until the slice is large enough to carry a real effect size.** Fix evaluation first; the methods list above is only worth running once you can tell them apart.

This is a fascinating and highly constrained problem. The detailed breakdown of the LatentWire project shows a team doing rigorous, disciplined work by anchoring to a frozen smoke contract.

However, the fact that you have a 4/32 benchmark ceiling on a narrowly matched Qwen $\rightarrow$ Qwen pair, while your compositional toy models collapse when exposed to abundant paired data, suggests that trying to find a direct mathematical mapping between two distinct, highly non-linear latent spaces is fighting against the fundamental geometry of the models.

Here is a blunt, domain-driven assessment of your current state, drawing heavily from distributed systems and computer systems architecture.

---

### 1. The Highest-Leverage Branch: Interface Redesign

I would bet heavily on **Interface Redesign**, specifically moving away from direct latent mapping and toward a **query/resampler-style connector**.

Attempting to directly align two distinct latent spaces is mathematically equivalent to trying to force two completely different Instruction Set Architectures (ISAs) to share a raw L1 cache without a coherence protocol or translation layer. It rarely scales, and it is highly brittle to minor version changes. Instead of forcing Model B to accept Model A's raw or rotationally-aligned matrix, you need an "interconnect bus."

A Perceiver-style cross-attention bottleneck forces Model A's dense state into a fixed, low-dimensional set of latent queries. This creates a standardized transit schema, protecting Model B from the high-frequency noise and structural idiosyncrasies of Model A.

### 2. The Fastest Falsification Ablation

To determine if `dynalign + resid16` is structurally real or just a calibration artifact, perform a **Cross-Family Swap**.

Maintain the exact same GSM8K 32-example contract, but swap the target model from Qwen3-0.6B to a similarly sized model from a different family (e.g., Llama-3-8B-Instruct or a smaller equivalent).

- **If the score holds:** The repair method is learning true cross-model reasoning transfer.
- **If the score collapses to 0/32:** Your current 4/32 lift is merely overfitting to the shared architectural geometry and parameter initialization biases inherent to the Qwen lineage.

### 3. Telemetry for Structural Gain

To prove the residual repair is doing useful work rather than acting as a glorified regularization term, track the **Intervention Routing Metric**:

- Calculate the $L_2$ norm of the residual update at each token step: $$\| \Delta h_t \|_2$$
- Correlate this magnitude with the "reasoning density" of the token. In GSM8K, the repair module should fire with significantly higher magnitude on critical numeric extractions and arithmetic operators, and remain near zero on filler text. If the repair norm is uniformly distributed across the sequence, it is just adding generic noise/calibration.

### 4. Alternative Representation: Universal Transport

Before investing more in the current basis, try routing the communication through a **Sparse Autoencoder (SAE) Information Bottleneck**.

Latent spaces are dense and entangled. Instead of learning a mapping from $Latent_A \rightarrow Latent_B$, train an SAE to project $Latent_A$ into a highly sparse, interpretable universal dictionary. Then, train a lightweight adapter to map that sparse dictionary into Model B. If you can preserve the dominant anchors in a sparse transit state, you decouple the communication from the specific dense manifolds of either model.

### 5. Analogy from Another Field: Distributed Systems

When scaling massive distributed execution graphs, individual nodes do not share raw memory states. They pass structured messages via a well-defined RPC schema.

Currently, LatentWire is attempting a raw memory dump between two entirely different execution environments. You are missing a **schema**. The "toy story" you noted (sparse shared dictionaries, byte-aligned sidecars) works in low-shot settings because it is artificially enforcing a schema. To make this work at scale, the transport layer must force the models to communicate in a shared, intermediate protocol rather than trying to surgically align their internal logic gates.

### 6. Are you asking the wrong question?

You are asking: _"How do we map Model A's latent space to Model B?"_ The better question is: _"What is the minimum viable state delta Model B needs to bypass re-computing Model A's reasoning?"_ You do not need to send the whole state. You only need to send the exact structural "clues" that allow Model B's attention heads to snap directly to the correct output distribution without walking the full reasoning path.

### 7. Smallest Decisive Experiment

To test the resampler interface redesign:

1.  Freeze both models.
2.  Introduce a tiny cross-attention bottleneck (e.g., 16 to 32 learned queries) between the final reasoning layers of Model A and the input embedding space of Model B.
3.  Train _only_ the bottleneck queries on the 32-example GSM8K slice.
4.  If this bottleneck can achieve $\ge 4/32$, you have proven that an intermediary transit space is as effective as heavy basis surgery, but inherently more scalable.

---

If we assume direct latent mapping is a dead end and pivot entirely to an intermediary bottleneck connector, how would you structure the loss function to ensure the bottleneck prioritizes preserving the specific reasoning pathways rather than just falling back to generic token prediction?

# Cross-Model Latent Communication: What You're Likely Missing

I've read the full experimental history carefully. Here is blunt, domain-grounded advice.

---

## 1. The Bet: Joint Encoder-Decoder Communication Heads

**If I had to bet one branch, it would be: stop trying to align/repair existing source representations, and instead train a _learned communication protocol_ between the two models.**

Concretely: add a small **bridge encoder** (linear or MLP projection) on the source side that maps source hidden states into a fixed number of soft-token embeddings (4–8 vectors). Add a **bridge decoder** (cross-attention or additive injection) on the target side that integrates these soft tokens into the target's residual stream at a specific layer. Train encoder + decoder jointly with target-side loss, keeping both base models frozen.

**Why this beats the current direction:**

Your entire experimental history tells one consistent story: **target-side supervision is necessary; analytic/geometry-based importance is not.** The only thing that works (dynalign + resid16) is trained end-to-end with a target-side loss. Every approach that computes importance analytically—eigenspace, saliency, preserve-core, shared basis, gauge wrappers—fails. This is not a coincidence. The target model's sensitivity landscape is too complex to capture analytically; you must learn it.

The current approach is:

1. Take source's existing latents (designed for source's own computation, not for communication)
2. Try to map them into target's space
3. Add repair

This is like trying to build a modem by post-processing an analog signal that was never designed for transmission. In communications engineering, you don't do this—you design the **encoder and decoder jointly for the channel**. The source representation should be _produced_ with communication in mind.

This also unifies your two disconnected stories: the communication head naturally handles interface mismatch (it learns to encode in a way that survives the channel), and the sparse dictionary / canonicalization ideas become architectural constraints or regularizers on the communication space.

**Domain analogy from multimodal learning:** Your project is exactly where vision-language models were circa 2022. Early VLMs tried to align existing image encoders to text embedding spaces through geometric maps (like CLIP). The breakthrough came when people stopped aligning existing representations and built **learned connectors**—the projection in LLaVA, the Perceiver Resampler in Flamingo, the Q-Former in BLIP-2. These are trained to take one model's output and produce something the other model can consume. You are one year behind this transition, still in the "alignment" era.

**Relevant priors:** The Vision Wormhole (arXiv 2602.15382) is directly relevant—it shows that learned connectors between heterogeneous models can discover shared structure that geometric alignment misses. Also see Q-Former / BLIP-2 architecture for the general pattern.

---

## 2. The Fastest Falsification: Two Experiments, Zero New Training

### Experiment A: Source Correctness Diagnostic

**This is the single most important diagnostic you haven't reported.**

On the 2 examples where dynalign+resid16 wins over target_alone:

- **Does the source model get those examples right on its own?**

If yes → the communication might just be answer-copying via latent channel. The source knows the answer, the latent carries that signal, and the residual happens to route it. This is fragile and won't generalize.

If no → the communication is genuinely transferring useful _reasoning structure_ that the source computed but couldn't convert to a correct final answer. This is the strong version of your thesis and would be very exciting.

You should also check: on the examples where text*to_text relay \_hurts* the target (text*to_text = 1/32, worse than target_alone = 2/32), does the source model get the wrong answer? If so, text relay is poisoning the target with wrong answers, and the latent channel's advantage might be that it's \_too weak* to effectively poison—i.e., the gain is from doing less harm, not from transferring more signal.

**This diagnostic reshapes the entire project.** If the source model is usually wrong on GSM8K (a 0.5B model likely is), then you're trying to communicate from an unreliable source, and the entire premise needs qualification.

### Experiment B: Cross-Example Residual Swap

Take the 2 winning examples. For each:

1. Run with the correct example's residual → should win (confirms residual is responsible)
2. Run with the _other_ winning example's residual swapped in → if it still wins, the residual is doing something _structural_ (correcting a systematic bias); if it doesn't, the residual is _example-specific memorization_
3. Run with same-norm random noise as residual → establishes whether it's the specific residual values or just the energy

If (2) fails, the rank-16 residual has memorized 2 examples' corrections in 16 directions. That's not a communication method; it's a lookup table. And it explains why every elaboration breaks it—you're operating at the edge of memorization, not discovering transferable structure.

---

## 3. Telemetry for Structural vs. Calibration Noise

The single most convincing telemetry: **does the correction correlate with the source model's reasoning quality, or with some latent structural property?**

Specifically, instrument and report:

| Metric                                                                         | What It Tests                                    |
| ------------------------------------------------------------------------------ | ------------------------------------------------ |
| Source accuracy on win examples                                                | Is the source just right when it helps?          |
| Target-side output gradient w.r.t. communicated representation (per-dimension) | Which dimensions actually matter to the target?  |
| Residual activation overlap across winning examples                            | Structural correction or memorization?           |
| Oracle verification bound: max(target_alone, target+comm) per example          | What's the ceiling if verification were perfect? |

**The oracle verification bound is critical.** Run target_alone and target+communication on all 32 examples. For each example, take whichever output is correct. If the oracle bound is ≈4/32, your current method is already near-optimal for this source-target pair, and no amount of method improvement will help—you need a better source model or a different task. If the oracle bound is much higher (say 8+/32), there's real room for better communication methods.

I would not trust any gain as "structural" until you've reported source accuracy on the winning examples and the oracle verification bound.

---

## 4. Alternative Representation: Learned Soft-Token Prefix

Before investing more in the current latent alignment basis, try this:

1. Source model processes the input normally
2. At a chosen layer, extract the hidden state
3. Pass it through a **learned encoder** (MLP, maybe 2 layers) that produces 4–8 soft token vectors
4. These soft tokens are **prepended to the target model's input embeddings** (or injected at a specific layer via cross-attention)
5. The target model processes input + soft tokens normally
6. Train encoder + (if using cross-attention) decoder jointly with target-side next-token loss on the GSM8K training set

**Why this might work where current approaches fail:**

- It doesn't require full latent alignment—just learning to compress source reasoning into a few vectors
- It controls the interference footprint (number of soft tokens = communication budget)
- It's been proven to work across modality boundaries (image→soft tokens→LLM in VLMs)
- It doesn't require architectural compatibility—soft tokens are just embeddings the target can process
- It naturally handles the capacity constraint: fewer soft tokens = less interference with target's own computation

**Why text relay fails but this might not:** Text relay forces the target to process tokens from the source's _vocabulary_, which may be misleading or distracting. Soft tokens carry no surface-form semantics—they're pure learned signals that the target model has learned to interpret.

**Smallest decisive experiment:** Implement the above with 4 soft tokens, prepended as embeddings. Train on GSM8K train set for 1–2 epochs. Evaluate on your frozen GSM8K32 contract. This is a 1-day experiment. If it gets ≥4/32, the approach has legs. If it gets ≤2/32, the bottleneck is source information quality, not the communication method.

---

## 5. What You're Obviously Missing: The Interference Constraint

**The deepest signal in your data is not what works—it's what hurts.**

Text relay (1/32) is _worse_ than target alone (2/32). Every "add more stuff" approach (routing, banking, sidecars, codebook tails) either collapses or regresses to the old ceiling. The only thing that works is a narrow, constrained correction (rank-16 residual).

**Your real constraint is not information capacity—it's interference minimization.**

The target model (0.6B) is operating near its capacity limit on GSM8K. Any additional information, even correct information, can hurt if it's not perfectly integrated. This is why:

- Text relay hurts (adds distracting tokens)
- Most elaborations on dynalign+resid16 hurt (increase interference footprint)
- Rank-16 residual barely helps (it's constrained enough to not interfere much)

**From control theory:** This is exactly the problem of _minimal intervention control_. You don't want to replace the target model's computation; you want to apply the smallest possible corrective signal at the right point. The right framing is:

> **"What is the minimum intervention on the target model's computation that reliably improves its output, conditioned on the source model's computation?"**

This is a control problem, not a communication problem. You're not trying to replicate the source's state in the target; you're trying to find the smallest corrective signal that steers the target toward better outputs.

**From joint source-channel coding:** In communications, when you have a noisy channel and a complexity-constrained receiver, you don't send the raw source signal and hope the receiver can decode it. You design the encoder to produce a signal that's robust to the channel noise and easy for the decoder to process. The current approach is doing the equivalent of sending the raw source signal through increasingly sophisticated channel equalizers, when what's needed is a redesign of the source encoding.

---

## 6. If You're Asking the Wrong Question

I think the question is partially wrong. The current framing—"how do we transport latent representations between models?"—assumes that the source model's internal state contains transferable reasoning structure, and the bottleneck is transport.

**The better question:** _"Under what conditions does a source model's computation contain information that a target model cannot obtain on its own but can integrate if provided? And what is the minimal channel that extracts and delivers that information?"_

This question naturally leads to:

1. **Characterizing when the source has complementary information** (the source correctness diagnostic)
2. **Measuring the target's integration capacity** (the oracle verification bound)
3. **Designing minimal channels** (soft-token communication heads)
4. **Understanding interference** (why adding information hurts)

It also changes what you should benchmark. GSM8K with a 0.5B source and 0.6B target may be the hardest possible test: the source barely understands the task, the target barely has capacity to integrate external information. You might want a **sweeter-spot test** where the source has strong reasoning but the target has a complementary weakness—for example, a strong general model that's weak at math receiving hints from a weaker model that's specifically good at arithmetic.

---

## 7. Summary of Smallest Decisive Experiments

| #   | Experiment                                                                                   | Time    | What It Decides                                            |
| --- | -------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| 1   | Source model accuracy on GSM8K32, especially the 2 winning examples                          | 1 hour  | Is the gain answer-copying or reasoning transfer?          |
| 2   | Cross-example residual swap on the 2 winning examples                                        | 1 hour  | Is the residual structural correction or memorization?     |
| 3   | Oracle verification bound: max(target_alone, target+comm) per example                        | 1 hour  | What's the ceiling? Is the method near-optimal?            |
| 4   | Learned soft-token prefix (4 tokens, trained 1-2 epochs)                                     | 1 day   | Does a joint encoder-decoder beat repair-based approaches? |
| 5   | Target-side sensitivity: which dimensions of the communicated rep most affect target output? | 2 hours | Are you preserving the right anchors?                      |

**Do experiments 1–3 before anything else.** They require zero new training and will either validate the current direction or force a pivot. If the residual is memorization and the source is just right on the winning examples, you should stop investing in repair-based methods and pivot to the communication-head approach (experiment 4) or reconsider whether this source-target-task combination is the right test bed.

---

## Bucket Mapping

| Idea                                      | Bucket                       |
| ----------------------------------------- | ---------------------------- |
| Joint encoder-decoder communication heads | **interface / transport**    |
| Source correctness diagnostic             | **benchmarking / telemetry** |
| Cross-example residual swap               | **verification / repair**    |
| Oracle verification bound                 | **benchmarking / telemetry** |
| Target-side sensitivity analysis          | **alignment / geometry**     |
| Interference minimization framing         | **control / verification**   |
| Soft-token prefix                         | **interface / transport**    |

---

## One Final Warning

The fragility of your single positive result (dynalign+resid16 = 4/32, basis-specific, non-reproducing in matched controls) is a red flag. In my experience, results that narrow—where the method works on exactly one basis and every nearby variant fails—usually indicate that you've found a local optimum in a very flat landscape, not that you've found the right method. The method is likely overfitting to the specific interaction between this basis, this pair of models, and these 32 examples.

The fastest way to confirm or deny this: **run your existing best method on a second frozen slice of 32 GSM8K examples.** If the gain replicates, it's real. If it doesn't, you're chasing noise. This is more important than any new method experiment.

This is an exceptionally well-prepared research update. The discipline around the frozen contract is exactly right. I'm reading this as someone who has worked on representation alignment, multimodal connectors, and verification gating. I'll give you the bluntest read I can, organized by your questions.

## 1. What I Would Bet On Next

**Bet: Anchor-preserving codebook tail, but not as a standalone compression method.**

Why this, not the others:

- **Multi-expert repair** is premature. You don't yet have a repair signal that is reliably _additive_ on the real contract. Adding routing complexity to a signal that only works in one narrow branch (`dynalign + resid16`) and collapses in most nearby variants (`value_bank`, `routed_bank`, `preserve_core`) is more likely to obscure than illuminate.
- **Verifier-gated repair** suffers from the same problem. Your verifier sidecar was valid but non-additive. That suggests the verifier's gate is not firing _differentially_ on examples where repair would help. You need a repair mechanism that works _before_ gating can be properly evaluated.
- **Interface redesign** is the nuclear option. It's worth keeping as a parallel toy lane (byte sidecar, etc.), but abandoning the real lane entirely before you've exhausted the anchor-preservation hypothesis would be a mistake. The `preserve_topk_uniform_tail` toy result (accuracy 0.9583 → 0.9896, MSE 0.7463 → 0.0284) is the single strongest quantitative clue in the entire document outside the real `0.1250` row.

**Why anchor-preserving tail modeling specifically:**

The real `dynalign + resid16` lift is narrow. The toy `preserve_topk` clue suggests _where_ the narrowness might come from.

Hypothesis: The `dynalign` basis is already good for _most_ latent dimensions, but the repair module is being asked to correct errors across _all_ dimensions uniformly. The dimensions that actually need repair are likely a sparse, high-variance tail. By preserving the dominant anchors exactly (or with high precision) and only applying the low-rank residual to the tail, you may:

- reduce interference with the already-working parts of the `dynalign` map
- concentrate the repair capacity where the mismatch actually lives
- avoid the collapse observed in `dynalign_preserve_core`, which preserved the _whole_ core and repaired nothing, losing the lift

**The decisive experiment:**

Run `dynalign` with a _gated_ residual that only fires on dimensions below a certain variance or saliency threshold in the source representation.

- **Bucket 1 (anchors)**: top-k dimensions by source activation variance → pass through `dynalign` unmodified.
- **Bucket 2 (tail)**: remaining dimensions → apply the `residrank16` correction.

If this preserves the `0.1250` lift and _improves_ stability across the failed wrapper variants (e.g., `preserve_core`, `saliency_preserve`), you have a strong signal that selective repair is the right direction. If it collapses like `preserve_core`, then the repair module is likely doing something more subtle than tail correction, and you need to re-examine the `dynalign` basis itself.

---

## 2. The Single Ablation That Would Falsify the Current Story Fastest

**Ablation: Cross-family frozen contract.**

You are currently using `Qwen2.5-0.5B-Instruct → Qwen3-0.6B`. This is a same-family, nearly-same-architecture pair. The toy story strongly suggests that symmetry/gauge/interface mismatch matters most when models are _different_.

**The test:**
Run the exact same frozen GSM8K32 contract, same `dynalign_module_replace_residrank16` setup, but with a **different target family**.

- Source: `Qwen2.5-0.5B-Instruct`
- Target: `SmolLM2-1.7B-Instruct` or `Llama-3.2-1B-Instruct`

Why this falsifies:

- If `dynalign + resid16` still gives a lift over `target_alone` (even a small one), the story is **not** just same-family calibration. It's a real, albeit weak, cross-model communication method.
- If the lift **disappears entirely** and you fall back to `target_alone` or below, then your entire real-lane story is **specific to Qwen→Qwen**. The `0.1250` lift would be a family-specific calibration artifact, not a reasoning transfer method.

This is the fastest way to determine whether you should continue investing in the real same-pair lane or pivot entirely to the toy interface-mismatch lane (byte sidecars, etc.) as the main path forward.

---

## 3. Telemetry That Would Convince Me a Gain Is Structural

For a `0.1250` lift (4/32) on a 32-example slice, statistical significance is impossible. You need **mechanistic telemetry**.

**Convincing telemetry set:**

1.  **Error-type shift:** On the 2 examples where `target_alone` fails but `dynalign+resid16` succeeds, what was the failure mode of `target_alone`?
    - Calculation error (wrong arithmetic)?
    - Misinterpretation (wrong operation)?
    - CoT derailment (hallucinated numbers, premature stop)?
    - If the latent method _systematically_ fixes a specific class of errors (e.g., corrects an intermediate numeric value in the CoT), that's a structural clue. If it's just "different random seed gave a better guess," it's noise.

2.  **Intervention on the repair module:**
    - Take the `dynalign+resid16` model on a _successful_ example.
    - Ablate the `resid16` correction vector (set it to zero) and re-run generation.
    - Does the model _revert_ to the `target_alone` failure mode?
    - If yes, the repair module is causally responsible for the gain. If the gain persists without the repair vector, the lift came from somewhere else (e.g., the `dynalign` basis initialization, not the repair).

3.  **Example-level win concentration:**
    - Run the `dynalign+resid16` variant on the _full_ GSM8K validation set (or a larger held-out slice) and rank examples by the target model's standalone confidence/accuracy.
    - Does the latent method's lift concentrate on examples where `target_alone` was **low-confidence** or **high-confidence**?
    - A lift on low-confidence examples suggests the latent state is providing _missing_ reasoning. A lift on high-confidence examples suggests it's just reducing noise/variance, which is less interesting.

---

## 4. Alternative Representation to Try Before Investing More in the Current Basis

**Alternative: Textual-intermediate layer, not pure latent.**

You are trying to bypass text entirely. But your `text_to_text` baseline is a weak relay (1/32). What if the problem isn't text _per se_, but the **verbosity** of text?

**Experiment: "Structured CoT Relay"**

Instead of `source_model → latent → target_model`, try:
`source_model → structured_reasoning_trace → target_model`

Where the `structured_reasoning_trace` is a minimal, JSON-like representation of the reasoning steps:

```json
{
  "steps": [
    { "op": "extract", "values": [120, 60] },
    { "op": "multiply", "args": [120, 60], "result": 7200 },
    { "op": "divide", "args": [7200, 8], "result": 900 }
  ],
  "final_answer": 900
}
```

**Why this is worth testing:**

- It preserves the **reasoning structure** you care about (the sequence of operations).
- It compresses away the natural language fluff that might confuse the target model or waste context.
- It gives you a clean upper bound on what's possible with a _non-latent_ structured relay.
- If this structured relay gets, say, 6/32 or 8/32, then the problem with `text_to_text` was the _format_, not the medium. Your latent method would then need to beat _that_ bar, not the weak 1/32 bar.

This is a fast, high-signal experiment. It requires no new model training, only a prompt to the source model and a parser.

---

## 5. Mature Technique You're Obviously Missing (from Control/Communications)

**Analogy: Kalman filtering with a faulty sensor model.**

In control theory, when you have a sensor that is sometimes accurate and sometimes drifts, you don't just trust it blindly or ignore it. You use a Kalman filter: you have a _process model_ (how the system should evolve) and a _measurement model_ (how the sensor relates to the system). You update your belief by weighting the sensor's reading by its _estimated uncertainty_.

**Translation to your problem:**

- **Process model:** The target model's own next-token prediction distribution `P_target(x_t | x_<t)`.
- **Measurement model:** The latent state from the source model, which projects to a distribution over next tokens `P_source(x_t | latent)`.

Currently, you're doing hard replacement (`module_replace`) or additive correction (`resid`). You're not **fusing** the information based on uncertainty.

**The idea: Uncertainty-Weighted Latent Fusion**

At each generation step, the target model computes two things:

1. Its own logits `L_target`.
2. A logit correction `ΔL` derived from the source latent state (via a small learned projector).

The final logits are:
`L_final = L_target + g(uncertainty) * ΔL`

Where `g(uncertainty)` is small when the target model is _confident_ (low entropy in `L_target`) and large when the target model is _uncertain_ (high entropy). This is a soft, dynamic gate that doesn't require a separate verifier model.

**Smallest decisive experiment:**
Take your `dynalign` baseline. Instead of `module_replace`, use the source latent to produce a _logit bias vector_. Scale this bias by the target model's current token entropy. This is a 20-line change to the generation loop. If it lifts the `0.0938` ceiling, you've found a better fusion mechanism than hard replacement.

---

## 6. Are You Asking the Wrong Question?

**Yes, partially.**

Current question: "How do we build a better bridge for cross-model latent communication?"

**Better question: "What is the _minimal_ information one model needs from another to improve reasoning, and how do we _protect_ that information from being corrupted by the target model's own internal dynamics?"**

Your toy lane (symmetry, gauge, sidecars) is essentially about **protection**. Your real lane (dynalign, resid) is about **correction**. You're treating correction as a post-hoc fix to a faulty map.

What if the map isn't faulty, but the _target model's internal dynamics actively distort_ the incoming latent state? In neuroscience, this is the problem of **stimulus-specific adaptation** or **predictive coding**. The brain doesn't just passively receive signals; it predicts them and amplifies surprises.

**Implication for your work:**
The target model's residual stream already has a strong _prior_ about what should come next. When you inject a latent state via `module_replace`, the target model's subsequent layers may actively _suppress_ the injected information if it conflicts with the model's own internal predictions.

**The missing experiment:**
Measure the **KL divergence** between the target model's internal activations _with_ and _without_ the latent injection, layer by layer. You might find that the injected signal is strong in layer N (where you inject) but is rapidly attenuated by layer N+2. If so, the problem isn't the bridge; it's the target model's **resilience to external perturbation**. You would need a method to make the target model more _receptive_ (e.g., fine-tuning it to attend to a special "external context" token, rather than replacing its internal state).

---

## 7. Smallest Decisive Experiment for My Main Bet (#1)

**Experiment: Gated Residual Tail Repair**

**Setup:**

- Base: `dynalign_module_replace` (the `0.0938` ceiling).
- Modification: Add the `residrank16` module, but gate its _output_ based on source activation variance.

**Procedure:**

1. For a given input, run the source model and collect the activations at the layer you're aligning to. Compute the variance of each dimension across the sequence length. Let `v_i` be the variance of dimension `i`.
2. Choose a threshold `τ` such that the top `k` dimensions by variance are considered "anchors". (Start with k = 25% of `d_model`).
3. The residual correction vector `r` is computed as usual.
4. The final output is `h_dynalign + m ⊙ r`, where `m` is a mask vector. `m_i = 0` if dimension `i` is an anchor, `m_i = 1` otherwise.

**Comparison rows to run on the frozen GSM8K32 contract:**

- `dynalign` (baseline): `0.0938`
- `dynalign + resid16` (live): `0.1250`
- `dynalign + gated_resid16_tail` (new): ?
- `dynalign + gated_resid16_anchors` (inverse control): gate _only_ anchors, zero tail.

**Interpretation:**

- If `gated_resid16_tail` **matches or exceeds** `0.1250`, the hypothesis is supported. The gain comes from repairing the tail, and you can now ablate the anchor dimensions safely.
- If `gated_resid16_anchors` **matches** `0.1250`, the hypothesis is inverted. The repair is only needed on the most active dimensions.
- If **both gated versions regress** to `0.0938`, the dense, un-gated residual is essential, and the anchor/tail distinction is not the right axis. You would then pivot to the Kalman-style uncertainty fusion (#5) as the next most promising mechanistic hypothesis.

Here is a blunt, domain-grounded assessment mapped directly to your questions. I will prioritize the smallest decisive experiments and explicitly call out likely flawed assumptions.

---

### 1. If I had to bet on one next branch, what and why?

**Bucket:** `Interface/Transport` + `Quantization/Compression` + `Verification/Repair`
**Bet:** **Anchor-preserving bottleneck + sequence-level dynamic filtering (not static residuals).**

**Why:** Your `dynalign + residrank16` lift tells you two things: (a) the alignment basis is approximately correct, and (b) a low-rank correction helps but only when tightly constrained. Static residuals fail because representation drift is _non-stationary across tokens and layers_. The target model’s own forward pass already contains a strong prior over what the next latent _should_ look like. Treat the source latent as a noisy observation of that prior, and use a **predictor-corrector** scheme:

- Target computes its own next-step latent prediction.
- Compute divergence between prediction and transported source latent.
- Inject correction _only_ where divergence exceeds a threshold, and _only_ on preserved anchors (top-K directions).
  This merges your toy insight (gauge/symmetry matters) with your real insight (output-aware alignment + narrow residual helps), while avoiding the saturated one-gate/routing dead ends.

---

### 2. What single ablation would falsify your current story fastest?

**Ablation:** **Cross-task zero-shot transfer of the repair module.**
Train `dynalign + resid16` on the GSM8K32 contract. Freeze it. Evaluate on MATH50 or AQuA _without retraining_.

- **If accuracy drops to `target_alone` (0.0625):** The lift is task-specific calibration/overfitting to GSM8K numerical patterns, not a structural communication channel.
- **If it holds or transfers partially:** The lift captures a generalizable interface property.

**Alternative fast falsifier:** _Source-latent batch shuffling._ Permute source latents across examples within a batch before transport. If performance doesn't collapse, your method is learning dataset-level biases, not example-specific reasoning structure.

---

### 3. What telemetry would convince me a gain is structural?

You need four metrics tracked per example:

1. **Repair utilization sparsity:** % of tokens/layers where the repair module actually modifies the latent. If it's >40% dense, it's not selective; it's just overwriting.
2. **Information bottleneck signal:** KL divergence between target hidden states (with vs. without repair). Plot KL vs. task success. A structural gain shows a tight positive correlation: higher divergence on failed examples, lower on successful ones.
3. **Anchor stability:** Cosine similarity of top-K preserved directions across 3 random seeds. High stability + performance lift = structural basis alignment.
4. **Latency/Bytes vs. Accuracy Pareto:** Explicitly measure transmitted bytes (or equivalent token equivalents) vs. accuracy. If latent transfer uses <30% of text relay bandwidth at matched/exceeding accuracy, you have a concrete efficiency win. Without this, "latent communication" is just an engineering curiosity.

---

### 4. What alternative representation/interface would I try before investing more in the current basis?

**Assumption to challenge:** _Token-wise KV/residual alignment is the right transport space._ It almost never is across different models due to tokenizer splits, head permutations, layer-wise semantic drift, and attention pattern reordering.

**Alternative:** **Sequence-level trajectory waypoints + resampler interface.**

- Extract a fixed-size set of "reasoning waypoints" from the source via learned queries (Perceiver/Q-Former style) or temporal pooling over reasoning-critical layers.
- Transport these waypoints through a tokenizer-agnostic continuous space.
- Target model interpolates them via cross-attention or a lightweight prefix adapter.
  **Why:** This bypasses token-to-token mapping entirely. You're aligning _state transitions_, not instantaneous activations. The toy sidecar results (byte/sequence-aligned) already hint this direction; formalizing it with a resampler removes the need for perfect gauge fixing at the token level.

---

### 5. Missing analogy/technique from another field?

**Field:** Communications & Control Theory
**Analogy:** **Hybrid ARQ / Incremental Redundancy + Event-Triggered Control.**
In modern comms, you don't send a full frame and hope it decodes. You send a base sketch, let the receiver attempt decoding, and only transmit parity bits if the receiver signals high uncertainty. Your current pipeline sends a full latent "frame" + static residual. Instead:

- Send compressed anchors.
- Let target run forward.
- If next-token entropy over key reasoning steps exceeds a threshold, trigger a targeted "parity burst" from the source tail.
  This makes communication _adaptive and task-aware_, not static. It naturally solves your "when/where/how much to fire" repair problem without brittle routing gates.

---

### 6. If I think you're asking the wrong question, what's the better one?

**Current question:** _"How do we align Model A's latents to Model B's latents so B can continue reasoning?"_
**Better question:** _"What is the minimal sufficient statistic of A's reasoning trajectory that reduces B's task uncertainty below a decision threshold?"_

**Why it matters:** Representation alignment is gauge-dependent and brittle. Uncertainty reduction is task-invariant and measurable. You don't need to match KV spaces; you only need to transmit the bits that meaningfully shift B's output distribution. This reframes the problem from _geometric matching_ to _information bottleneck optimization_, which aligns cleanly with your anchor-preservation intuition and your saturation results on full latent alignment.

---

### 7. Smallest decisive experiment to test this?

**Experiment:** `Anchor-Preserving Parity Burst` (can run in <1 day on existing GSM8K32 contract)

1. Compute top-K principal directions across source latents for the 32 examples (anchors).
2. Transport only these K dimensions (or a 4-bit quantized version) via `dynalign`.
3. Target runs forward pass. Compute per-step next-token entropy.
4. If entropy > threshold on arithmetic/variable-assignment steps, inject next-batch of tail latents. Else, stop.
5. Compare accuracy & transmitted bytes vs. `dynalign + residrank16` and text-to-text.

**Why decisive:**

- Tests if information is concentrated in anchors (validates toy codec results).
- Tests if target uncertainty can gate communication (solves routing/sidecar saturation).
- Directly measures bytes/latency tradeoff.
- Requires only a lightweight entropy threshold + top-K extraction. No new architecture.

---

### Direct Critique of Current Assumptions

- **Token-wise alignment is a bottleneck.** You're fighting head permutations and tokenizer mismatch that will never fully disappear. Move to sequence-level or query-resampler interfaces.
- **Residual repair is fighting drift, not errors.** A static rank-16 residual assumes the mismatch is linear and stationary. It isn't. Predictor-corrector or event-triggered injection will outperform dense residuals at lower rank.
- **Toy gauge-fixing isn't cosmetic; it's the signal.** The fact that quotient/GPA/sparse dictionaries win in low-shot but lose with paired data means your current real benchmark is _too narrow_ to expose the true interface mismatch. Don't abandon it; use it as a filter, but validate structural gains via cross-task transfer or uncertainty gating.

If you implement the `Anchor-Preserving Parity Burst` with entropy gating and run the cross-task zero-shot ablation, you will either falsify the current lift in 48 hours or have a clean path to a unified, benchmark-ready method.

This is a beautifully characterized problem. You have done the hard work of carving away the wrong hypotheses. What remains is a narrow wedge of signal surrounded by a lot of negative space, which is exactly where good research lives.

Here is my read, drawing from several domains.

---

## 1. The Bet: Anchor-Preserving Value-Side Tail Repair

**If I had to bet on one branch, I would bet on explicitly separating the transport into anchor preservation (K-space) and value-side tail repair (V-space), with the tail coded against a toy-derived shared dictionary rather than a dense residual.**

**Why:** Your data already whispers this story but you have not formalized the split.

- `dynalign_value_routed + resid16` _preserves_ the 0.1250 lift. This means the entire gain is V-side. The value route is not "which expert" — it is "which side of the attention computation."
- Meanwhile, K-only/V-only transport ablations from Phase 1, the QK-template transport signal from Phase 2, and the quotient/gauge toy story all point to the same thing: **the key/query structure is the anchor.** It defines the coordinate system of reasoning. The value structure is the tail — it fills in content conditioned on that coordinate system.
- Your `preserve_topk_uniform_tail` toy result is dramatic (MSE 0.7463 → 0.0284) because when you freeze the dominant directions, you are freezing the coordinate system and only asking the tail to carry content.
- The residual on dynalign works because dynalign has already aligned the K-coordinate systems between models well enough that a small V-side correction suffices. The residual on tokenbasis fails because tokenbasis misaligns the K-coordinates, so a V-only patch cannot fix it.

**The concrete architecture I would try:**

```
Transport = {Preserved Anchors} ∪ {Tail}
Preserved Anchors = Top-k K-directions by spectral importance, sent at full precision
Tail = Everything else, coded as:
  - Index into a small shared dictionary (from your GPA/sparse-dict toy)
  + Small per-token V-side residual (rank-8 or rank-4, not rank-16)
  + Gated by target's current query state (not a generic gate)
```

This merges your two disconnected stories. The toy story (shared dictionary, canonicalization, sidecars) becomes the tail codec. The real story (dynalign + residual) becomes the anchor preservation + V-correction.

---

## 2. The Fastest Falsification

**Ablation:** On your frozen GSM8K32 contract, run `dynalign_module_replace_residrank16` but with the residual _restricted to apply only on directions orthogonal to the top-10% spectral directions of the source KV_.

If the 0.1250 result survives, your residual was already tail-only and the anchor/tail formalization is correct. If it collapses to 0.0625 or below, the residual was doing critical anchor-level work and the V-only story is wrong.

This costs you one benchmark row and tells you which side of the bet to pursue.

---

## 3. Telemetry That Would Convince Me

I would need to see **three non-negotiable signals** together:

| Signal                                 | What it proves                                                    | How to get it                                                                                                     |
| -------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **V-only residual concentration**      | The gain lives in value space, not key space                      | Per-head cosine similarity of source/target K vs V pre/post repair; K should already match before residual fires  |
| **Anchor stability under compression** | The top-k directions are genuinely preserved                      | Bit-flip the tail aggressively (quantize to 2-bit) and show anchors stay at >0.99 cosine; accuracy holds          |
| **Reasoning-step localization**        | The repair fires at specific reasoning transitions, not uniformly | Token-level heatmap of residual L2 norm against CoT structure (e.g., residual spikes after "therefore" or "wait") |

If the residual is uniformly distributed across tokens, it is probably calibration noise. If it clusters at reasoning boundaries, it is structural.

---

## 4. Alternative Interface to Try First

Before more basis surgery, I would try a **query-conditioned resampler transport** (Perceiver-style).

Instead of: `source_KV → align → send → target_injects`

Try: `source_tokens → small resampler → target_queries → received_latent`

The target does not passively receive aligned KV. It actively queries a lightweight resampler network that has seen the source's full token sequence. The transport is not a state dump; it is an interactive lookup.

**Why this matters for your problem:** Your entire current framework assumes the source must _send its state_ and the target must _adopt it_. But in multimodal literature (Vision Wormhole, MM1, Libra), the breakthrough was realizing that cross-modal transport works better as **query-response** than as **state alignment**. The target asks; the source answers.

Smallest experiment: Replace the KV alignment in dynalign with a 2-layer Perceiver resampler that takes source hidden states as input and is queried by the target's own current-layer hidden states. Keep everything else frozen. If this beats 0.0938, the interface itself was the bottleneck.

---

## 5. What You Are Obviously Missing (Cross-Domain Analogies)

**From feedback control / communications (my strongest conviction): You are running an open-loop protocol.**

Every communication system that survives channel mismatch uses feedback. Source → channel → receiver → ACK/NACK → retransmit. You have none of this. The source sends its state once and hopes.

The smallest form of feedback: let the target send back a _gradient signal_ on a tiny validation slice. Even one step of backprop from target loss to the transport interface would tell the source's sidecar "send more of this, less of that." This is not full training — it is interface adaptation.

**From joint source-channel coding (information theory):**

You are trying to do source coding (compress the source state) and channel coding (survive representation mismatch) separately. The optimal approach is joint: design the codebook knowing the channel. Your "shared dictionary + sidecar" is the right intuition but not pushed far enough. The sidecar should carry _correction information about the dictionary_, not just raw bytes.

**From differential geometry:**

You are trying to align points on two manifolds. But reasoning is a _flow_ — a trajectory on the manifold. The right object to transport is not the point (the KV state) but the **tangent vector** (the direction the source was about to move). If the target knows "the source was about to conclude X," it can integrate that from its own position.

**From program verification:**

Your verifier sidecar fails because it is a boolean classifier (good/bad). What you need is a **repair oracle** — not "is this wrong?" but "what delta would make it right?" This is the difference between Hoare logic (verify) andweakest precondition calculus (repair).

---

## 6. If You Are Asking the Wrong Question

I think you might be. The current question is:

> "How do we align the latent spaces so model A's KV state is useful to model B?"

A better question might be:

> **"What is the minimal sufficient statistic for reasoning continuation, and how do we make it model-agnostic by construction?"**

The difference is crucial. Your current framing assumes the source _has_ a state that the target _needs_. But the target is a competent reasoner on its own (it gets 2/32 alone). The source's job is not to transmit its full consciousness. It is to transmit **the one thing the target would not have thought of**.

This reframing suggests:

- The transport should be _sparse and targeted_, not dense alignment
- The interface should be _advice_, not _state adoption_
- The metric should be "what does the target do differently?" not "how well do the KV spaces align?"

---

## 7. Smallest Decisive Experiments

**Experiment A (tests anchor/tail split, 1 GPU day):**

1. Take dynalign+resid16 (your 0.1250 row)
2. Split K and V at the transport boundary
3. Send K anchors at full precision (top-10% directions)
4. Apply rank-8 residual _only_ to V-tail, gated by target query
5. Run on GSM8K32

Prediction: If ≥ 0.1250, the split is real and you can compress the tail aggressively. If < 0.0938, the residual was doing K-work and the V-only story is wrong.

**Experiment B (tests query-response interface, 2 GPU days):**

1. Freeze source and target models
2. Replace KV alignment with a 2-layer Perceiver resampler (latent dim 64)
3. Target queries the resampler with its own current hidden state
4. Fine-tune only the resampler on your 32-example contract
5. Compare to dynalign baseline

Prediction: If ≥ 0.0938, the interface redesign is higher-leverage than more alignment surgery. If < 0.0625, the state-dump model is actually correct for same-pair.

**Experiment C (tests feedback, 1 GPU day):**

1. Take your best transport (dynalign+resid16)
2. After target generates, compute target's own loss gradient w.r.t. the received latent
3. Send one gradient step back through the alignment layer
4. Measure if the _next_ example's accuracy improves

Prediction: Any improvement at all means feedback is the missing piece and open-loop is your core problem.

---

## Summary Ranking

| Branch                                     | Confidence                                    | Cost to Test |
| ------------------------------------------ | --------------------------------------------- | ------------ |
| Anchor/tail V-split (Experiment A)         | High                                          | 1 day        |
| Query-conditioned resampler (Experiment B) | Medium-High                                   | 2 days       |
| Feedback/closed-loop (Experiment C)        | High if you have a way to send gradients back | 1 day        |
| Multi-expert value repair                  | Medium                                        | 3 days       |
| Verifier-gated repair                      | Low-Medium                                    | 2 days       |

My recommendation: Run A and B in parallel. They are independent and each falsifies a different assumption. If both fail, then the answer is C — you need feedback.

The deeper pattern in your data is that **narrow wins survive and broad wins collapse**. The 0.1250 result is the narrowest positive signal you have. The right strategy is to make it even more narrow — understand exactly _which_ directions, _which_ heads, _which_ tokens — and then hard-code that structure into the architecture. Do not generalize from it until you have mechanistically understood it.

The following is the most important feedback, I'll paste it twice

My blunt read: the missing idea is that you are still mostly treating this as **latent alignment/compression**, when the right abstraction is **conditional communication with decoder side information**. The target already has a strong prior from its own hidden state and the prompt. In that regime, the worst mistake is to preserve the sender’s biggest or cleanest components just because they are big or clean; those are often the **most predictable** parts. The right things to preserve are the components that are both **unpredictable from the target** and **causally important for the target’s answer**. That fits your evidence unusually well: the real clue only appears when alignment is output-aware, while the toy clue appears when the interface has a small stable shared alphabet.

**1. If I had to bet on one next branch, I’d bet on a decoder-conditioned innovation codec** _(interface / transport + quantization / compression)._
Not more basis surgery. Not another generic residual on the raw aligned state. Concretely: use a small public interface, let the target query it, predict from target-alone what the source message should have been, and transmit only the **innovation residual**. Then protect the top innovation atoms and progressively refine the tail. Architecturally this looks closer to BLIP-2/Flamingo-style small querying connectors between frozen modules than to global latent matching, and codec-wise it should look more like a progressive residual codec than a one-shot repair MLP. QINCo is relevant because it adapts codebooks to residual structure and supports multirate/prefix-style decoding. ([arXiv][1])

The correction I would make to your current “anchor-preserving tail” instinct is: **preserve conditional innovation anchors, not absolute anchors**. “Top-k by norm,” “top-k by generic saliency,” and “important heads” are the wrong selectors if the decoder can already predict most of them. The right score is something like
`innovation magnitude × answer sensitivity`
where innovation is relative to a decoder-side predictor, not relative to zero. That is why I would not spend more time on preserve-tail or MoE variants over the raw aligned state. You are probably asking routers and quantizers to separate signal from a message that still contains too much common information.

**2. The single ablation that would falsify your current story fastest is cross-example message shuffling.** _(benchmarking / telemetry)_
Keep the exact target input, same transport budget, same norms/spectrum, same decode path, but swap the transmitted message across matched examples. If the live gain survives, you do not have communication; you have a calibration prior induced by the bridge. If it collapses, at least the channel is example-specific. I would do this before another week of experts, verifiers, or wrappers.

**3. What would convince me the gain is structural rather than noise?** _(benchmarking / telemetry)_
I would want four numbers more than I want another leaderboard row. First, a **specificity gap**: real message versus shuffled message. Second, a **predictability gap**: real message versus a decoder-predicted ghost message `ĝ(h_t, q)`. Third, paired gain in **gold-answer log-prob / numeric-token log-prob**, not just exact accuracy on 32 items. Fourth, a **rate curve**: layer 1 helps a bit, layer 2 helps mostly on high-entropy cases, and active-slot ablations remove wins when they should. If you tell an MoE story, expert usage should line up with innovation type or uncertainty, not just token position, sequence length, or feature norm. ([arXiv][2])

**4. The alternative interface I would try before investing more in the current basis is a receiver-conditioned slot interface.** _(interface / transport)_
Give the source 8–16 learned slots extracted from its sequence/KV by target queries, then hand only those slots over, optionally with a byte/sequence sidecar for hard mismatch. Let your gauge/canonicalization/shared-dictionary machinery initialize or regularize the public slot space, but do not force the whole handoff to live inside that space. The main point is that the receiver should decide what to read, because your own results say output-aware alignment matters more than geometry. If you want this to scale beyond pairwise collaboration, make the public code multiview from the start; DGCCA is the right family of precedent there. ([arXiv][1])

This is **not** a vote for “more xattn” in the generic sense. You likely already tried the weak form. The key difference is the objective: the slots should be optimized for **conditional utility to the target**, not unconditional source reconstruction.

**5. The mature analogy you are most obviously missing is Wyner-Ziv / distributed source coding, with a control-theory backup of observer innovations.** _(information theory + control)_
From communications: the decoder already has correlated side information, so the encoder should code conditionally, not absolutely; Neural Distributed Source Coding and V2X-DSC make that point directly in learned systems. From control: a distributed observer reconstructs what is locally detectable and only communicates the part that is not; that is almost exactly the decomposition you want. From compression: system-aware compression spends bits on components that matter at the downstream output and not on downstream null spaces. All three point the same way.

**6. I do think you are slightly asking the wrong question.** _(problem formulation)_
The better question is not, “Can A map its hidden state into B’s hidden state?” It is, “Given B’s current state and a rate budget, what message maximally changes B’s posterior on the correct answer?” In symbols, something like
`maximize Δ log p_B(y* | h_t, m) – λ · bits(m)`
with `m = f(h_s, h_t, q)`.
That change matters because it demotes alignment quality, latent MSE, and even exact hidden-state fidelity to means rather than ends. It also means your frozen GSM8K32 slice should stay a smoke gate, but **not** the main ranker for nearby variants. A one- or two-example swing on 32 items is too coarse to distinguish subtle ideas; use paired log-prob plus one adjacent frozen slice for that.

**7. The smallest decisive experiment I would actually run is this.** _(quantization / compression + verification / repair + benchmarking)_
Keep your current dynalign basis only as scaffolding. Fit a tiny predictor
`ĝ(h_t, q) -> z_s_aligned`.
Define residual
`r = z_s_aligned – ĝ(h_t, q)`.
Score each token/channel/slot with
`S_i = |r_i| × answer_sensitivity_i`,
where answer sensitivity is a gradient/Fisher proxy on the gold answer under the target. Preserve the top-k at high precision, quantize the tail with an RVQ/QINCo-style residual codec, and let a verifier do only one job: decide whether to send **refinement layer 2**, not whether to send the whole message. Then compare, at matched rate, against four baselines: dense resid16, absolute top-k preserve, predictor-only ghost message, and shuffled message. If innovation-top-k wins, you have a real unifying story. If predictor-only nearly matches, your communication claim is weak. If dense resid still wins and innovation-top-k does not survive one neighboring frozen 32-example slice, I would stop investing in anchor-preserving tails on this pair.

One last blunt point: I would not conclude from your routed-bank failures that **expert repair is dead**. I would conclude that you likely tried to route **before** isolating the innovation. Route the residual type after subtracting the decoder’s prediction, not the raw aligned state. Same for verification: gate refinement, not the base channel.

In one line: **don’t preserve anchors; preserve innovations.**

[1]: https://arxiv.org/abs/2301.12597 '[2301.12597] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models'
[2]: https://arxiv.org/html/2602.12866 'Model-Aware Rate–Distortion Limits for Task–Oriented Source Coding'

My blunt read: the missing idea is that you are still mostly treating this as **latent alignment/compression**, when the right abstraction is **conditional communication with decoder side information**. The target already has a strong prior from its own hidden state and the prompt. In that regime, the worst mistake is to preserve the sender’s biggest or cleanest components just because they are big or clean; those are often the **most predictable** parts. The right things to preserve are the components that are both **unpredictable from the target** and **causally important for the target’s answer**. That fits your evidence unusually well: the real clue only appears when alignment is output-aware, while the toy clue appears when the interface has a small stable shared alphabet.

**1. If I had to bet on one next branch, I’d bet on a decoder-conditioned innovation codec** _(interface / transport + quantization / compression)._
Not more basis surgery. Not another generic residual on the raw aligned state. Concretely: use a small public interface, let the target query it, predict from target-alone what the source message should have been, and transmit only the **innovation residual**. Then protect the top innovation atoms and progressively refine the tail. Architecturally this looks closer to BLIP-2/Flamingo-style small querying connectors between frozen modules than to global latent matching, and codec-wise it should look more like a progressive residual codec than a one-shot repair MLP. QINCo is relevant because it adapts codebooks to residual structure and supports multirate/prefix-style decoding. ([arXiv][1])

The correction I would make to your current “anchor-preserving tail” instinct is: **preserve conditional innovation anchors, not absolute anchors**. “Top-k by norm,” “top-k by generic saliency,” and “important heads” are the wrong selectors if the decoder can already predict most of them. The right score is something like
`innovation magnitude × answer sensitivity`
where innovation is relative to a decoder-side predictor, not relative to zero. That is why I would not spend more time on preserve-tail or MoE variants over the raw aligned state. You are probably asking routers and quantizers to separate signal from a message that still contains too much common information.

**2. The single ablation that would falsify your current story fastest is cross-example message shuffling.** _(benchmarking / telemetry)_
Keep the exact target input, same transport budget, same norms/spectrum, same decode path, but swap the transmitted message across matched examples. If the live gain survives, you do not have communication; you have a calibration prior induced by the bridge. If it collapses, at least the channel is example-specific. I would do this before another week of experts, verifiers, or wrappers.

**3. What would convince me the gain is structural rather than noise?** _(benchmarking / telemetry)_
I would want four numbers more than I want another leaderboard row. First, a **specificity gap**: real message versus shuffled message. Second, a **predictability gap**: real message versus a decoder-predicted ghost message `ĝ(h_t, q)`. Third, paired gain in **gold-answer log-prob / numeric-token log-prob**, not just exact accuracy on 32 items. Fourth, a **rate curve**: layer 1 helps a bit, layer 2 helps mostly on high-entropy cases, and active-slot ablations remove wins when they should. If you tell an MoE story, expert usage should line up with innovation type or uncertainty, not just token position, sequence length, or feature norm. ([arXiv][2])

**4. The alternative interface I would try before investing more in the current basis is a receiver-conditioned slot interface.** _(interface / transport)_
Give the source 8–16 learned slots extracted from its sequence/KV by target queries, then hand only those slots over, optionally with a byte/sequence sidecar for hard mismatch. Let your gauge/canonicalization/shared-dictionary machinery initialize or regularize the public slot space, but do not force the whole handoff to live inside that space. The main point is that the receiver should decide what to read, because your own results say output-aware alignment matters more than geometry. If you want this to scale beyond pairwise collaboration, make the public code multiview from the start; DGCCA is the right family of precedent there. ([arXiv][1])

This is **not** a vote for “more xattn” in the generic sense. You likely already tried the weak form. The key difference is the objective: the slots should be optimized for **conditional utility to the target**, not unconditional source reconstruction.

**5. The mature analogy you are most obviously missing is Wyner-Ziv / distributed source coding, with a control-theory backup of observer innovations.** _(information theory + control)_
From communications: the decoder already has correlated side information, so the encoder should code conditionally, not absolutely; Neural Distributed Source Coding and V2X-DSC make that point directly in learned systems. From control: a distributed observer reconstructs what is locally detectable and only communicates the part that is not; that is almost exactly the decomposition you want. From compression: system-aware compression spends bits on components that matter at the downstream output and not on downstream null spaces. All three point the same way.

**6. I do think you are slightly asking the wrong question.** _(problem formulation)_
The better question is not, “Can A map its hidden state into B’s hidden state?” It is, “Given B’s current state and a rate budget, what message maximally changes B’s posterior on the correct answer?” In symbols, something like
`maximize Δ log p_B(y* | h_t, m) – λ · bits(m)`
with `m = f(h_s, h_t, q)`.
That change matters because it demotes alignment quality, latent MSE, and even exact hidden-state fidelity to means rather than ends. It also means your frozen GSM8K32 slice should stay a smoke gate, but **not** the main ranker for nearby variants. A one- or two-example swing on 32 items is too coarse to distinguish subtle ideas; use paired log-prob plus one adjacent frozen slice for that.

**7. The smallest decisive experiment I would actually run is this.** _(quantization / compression + verification / repair + benchmarking)_
Keep your current dynalign basis only as scaffolding. Fit a tiny predictor
`ĝ(h_t, q) -> z_s_aligned`.
Define residual
`r = z_s_aligned – ĝ(h_t, q)`.
Score each token/channel/slot with
`S_i = |r_i| × answer_sensitivity_i`,
where answer sensitivity is a gradient/Fisher proxy on the gold answer under the target. Preserve the top-k at high precision, quantize the tail with an RVQ/QINCo-style residual codec, and let a verifier do only one job: decide whether to send **refinement layer 2**, not whether to send the whole message. Then compare, at matched rate, against four baselines: dense resid16, absolute top-k preserve, predictor-only ghost message, and shuffled message. If innovation-top-k wins, you have a real unifying story. If predictor-only nearly matches, your communication claim is weak. If dense resid still wins and innovation-top-k does not survive one neighboring frozen 32-example slice, I would stop investing in anchor-preserving tails on this pair.

One last blunt point: I would not conclude from your routed-bank failures that **expert repair is dead**. I would conclude that you likely tried to route **before** isolating the innovation. Route the residual type after subtracting the decoder’s prediction, not the raw aligned state. Same for verification: gate refinement, not the base channel.

In one line: **don’t preserve anchors; preserve innovations.**

[1]: https://arxiv.org/abs/2301.12597 '[2301.12597] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models'
[2]: https://arxiv.org/html/2602.12866 'Model-Aware Rate–Distortion Limits for Task–Oriented Source Coding'
