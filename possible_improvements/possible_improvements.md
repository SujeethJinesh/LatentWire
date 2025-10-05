1. Anchor-based Large Language Models – PDF: arxiv:2402.07616. Summary: Introduces Anchor LLMs (AnLLMs) which use an Anchor-based Self-Attention Network (AnSAN) to force the model to compress all contextual information into special anchor tokens (typically the last token of each segment). During training, attention masks are modified so that each anchor token attends only to its own segment and preceding anchors, while non-anchor tokens attend to earlier tokens and prior anchors
   openreview.net
   openreview.net
   . This effectively condenses the sequence’s information into a few tokens, yielding up to 99% reduction in stored keys/values with minimal loss in performance
   arxiv.org
   . Suggested Architectural Idea: Incorporate anchor tokens into your model’s sequence and train with an adjusted attention mask so that these tokens “summarize” preceding content. At inference, retain only the anchor’s key/value vectors as the latent. Why it helps: By bottlenecking the model’s memory into a handful of learned tokens, you obtain a compact latent interlingua that preserves most of the sequence’s meaning
   arxiv.org
   – ideal for passing information between models or stages.

2. Deliberation in Latent Space via Differentiable Cache Augmentation – PDF: arxiv:2412.17747. Summary: Liu et al. (2024) propose a two-module architecture: a frozen base LLM (decoder) and a trainable coprocessor that operates on the base’s attention cache. The coprocessor writes a set of latent embeddings into the base model’s key-value cache, providing extra “thinking” vectors that the base can attend to
   arxiv.org
   . It is trained end-to-end by the language modeling loss while the base LLM remains frozen
   arxiv.org
   . This teaches the coprocessor to insert useful latent computation that improves the base’s next-token predictions (e.g. lower perplexity on downstream tokens)
   arxiv.org
   . Suggested Architectural Idea: Attach a latent-insertion module that takes the LLM’s current hidden state (or past cache) and generates additional key/value pairs (the latent) to augment the model’s self-attention at all layers. Train this module on unlabeled text by next-word prediction, keeping the main model fixed
   arxiv.org
   . Why it helps: This approach adds a learned latent “deliberation” step to a frozen LLM
   arxiv.org
   . The base model learns to incorporate the injected latent information, improving fidelity and reasoning without modifying its weights. It’s directly applicable to your goal since you can freeze the big LLM and let a smaller network handle the latent generation.

3. Training Large Language Models to Reason in a Continuous Latent Space (Coconut) – PDF: arxiv:2412.06769. Summary: This work (Hao et al., 2024) introduces Coconut (Chain-of-Continuous-Thought), enabling LLMs to perform multi-step reasoning internally using continuous vectors instead of language tokens. The method feeds the model’s last hidden state back into the model as the next-step input, rather than outputting it as text
   arxiv.org
   . In effect, the model generates a sequence of latent states (“continuous thoughts”) to traverse a reasoning problem. This allows exploration of multiple reasoning paths in parallel – e.g. the latent can encode multiple alternative next steps, enabling a breadth-first search through solution space
   arxiv.org
   . Coconut showed improved logical reasoning on tasks requiring backtracking, even outperforming explicit chain-of-thought prompting, while using fewer tokens
   arxiv.org
   . Suggested Architectural Idea: Implement a latent recurrence loop: after encoding a query, take the final hidden state (or a projection of it) and insert it back into the model (e.g. as a special prefix embedding) for another forward pass. Repeat this for a fixed number of “thought” iterations, then decode the answer. You can train this behavior by supervising the final answer (and possibly intermediate latent consistency). Why it helps: It forces the model to process information iteratively in latent space, encouraging a form of internal reasoning and compression. The latent becomes an expressive intermediate that both encodes and conveys rich semantics between passes
   arxiv.org
   – a useful property for a shared interlingua.

4. Exploring System 1 and 2 Communication for Latent Reasoning in LLMs – PDF: arxiv:2510.00494. Summary: Coda-Forno et al. (Oct 2025) investigate how a two-model “reasoning via latents” setup can be improved. They revisit the idea of a frozen base LLM (System-1) plus a Coprocessor (System-2) that exchanges latent messages. Crucially, they find that merely increasing latent size or capacity yields modest gains, whereas jointly fine-tuning the base model to “listen” to the latents yields far larger improvements
   arxiv.org
   arxiv.org
   . In their experiments, a co-trained base+Coprocessor significantly outperformed a frozen base setup under the same latent budget
   arxiv.org
   . They also propose writing latents deeper into the model (editing the cache at multiple layers) to ensure the information reaches all parts of the network
   arxiv.org
   . Suggested Architectural Idea: Instead of keeping your LLM completely frozen, use a lightweight tuning approach (e.g. LoRA or adapters) on the base model in tandem with training the latent encoder/decoder. For example, apply LoRA to attention layers so the base model can better incorporate the injected latent. Additionally, inject the latent not only at input but at multiple layers (e.g. via adapters that feed latent vectors into intermediate layers’ activations). Why it helps: Minor fine-tuning of the base will teach the LLM to pay attention to the injected latent info, making communication far more effective
   arxiv.org
   . Likewise, providing latent inputs at various depths ensures the model’s internal layers consistently integrate that information, improving reliability of latent decoding.

5. Prefix-Tuning+ (Modernizing Prefix-Tuning) – PDF: arxiv:2506.13674. Summary: Prefix-Tuning+ (Wang et al., 2025) is an improved variant of prefix-tuning designed for modern LLMs. It addresses a key limitation of vanilla prefix-tuning: in standard prefix-tuning, the prepended vectors reside within the self-attention mechanism, which creates a tug-of-war between attending to the prefix vs. the actual input
   arxiv.org
   . Prefix-Tuning+ instead moves the prefix out of the attention heads and into a dedicated module, decoupling prefix processing from main attention
   arxiv.org
   . Concretely, it learns an external projection that generates an “attention-independent” influence of the prefix. This change avoids the trade-off where a long prefix overshadows the input or vice-versa
   arxiv.org
   . As a result, Prefix-Tuning+ achieves consistently better performance than vanilla prefix-tuning on multiple tasks and even matches LoRA on some benchmarks
   arxiv.org
   arxiv.org
   . Suggested Architectural Idea: When feeding a latent as a prefix, don’t just prepend it as extra attention keys/values. Instead, learn a small module that takes the latent and produces an additive bias or intermediate representation applied outside the self-attention (for example, add it to the residual stream or as a modifier to layer norms). This follows the Prefix+ approach of isolating prefix influence. Why it helps: By decoupling the latent from direct token competition in attention
   arxiv.org
   , the model can incorporate the latent’s information more reliably. In short, the latent won’t “steal” attention away from the actual text – avoiding instability – but will still guide the model through the external module. This yields more robust latent conditioning, especially in deeper LLMs
   arxiv.org
   .

6. IAA: Inner-Adaptor Architecture for Frozen LLMs – PDF: arxiv:2408.12902. Summary: Wang et al. (AAAI 2025) propose Inner-Adaptor Architecture (IAA) to add new capabilities to a large LLM without fine-tuning it. Specifically, they focus on injecting visual understanding into a frozen text model by inserting multiple multimodal adapter modules at different transformer layers
   arxiv.org
   . Each adapter at layer $i$ takes in features from an external modality (e.g. image embeddings) and merges them with the LLM’s hidden state at that layer. By distributing these adapters throughout the stack, the LLM gains direct multimodal inputs at varying depths, which greatly improved its visual grounding performance
   arxiv.org
   . Notably, IAA achieved SOTA on vision-language tasks while keeping the LLM’s language ability intact, since the base weights stay frozen
   arxiv.org
   . Suggested Architectural Idea: Use a similar multi-depth adapter strategy for your latent. Instead of only feeding the latent at the embedding layer, insert small adapter blocks at several layers of the LLM. For example, at layers 5, 10, 15, have an adapter that takes the latent vector (or an encoding of it) and combines it with the layer’s hidden representation (via cross-attention or an MLP). The base model remains frozen, and only these adapters (and possibly a tiny latent encoder) are trained. Why it helps: Providing the latent information at multiple levels of abstraction helps the frozen model integrate and understand the latent more effectively. Early layers can extract low-level signals from the latent, while later layers get higher-level latent context, analogous to how IAA enabled a frozen LLM to deeply absorb image features
   arxiv.org
   . This should make the model’s latent-conditioned generation much more reliable.

7. Compressed Chain-of-Thought (CCoT) – PDF: arxiv:2412.13171. Summary: Cheng and Van Durme (2024) introduce a framework for compressing reasoning paths into dense vectors. Standard chain-of-thought (CoT) prompting forces models to generate long, token-level rationales. In contrast, CCoT trains the model to generate a small number of continuous “contemplation” tokens that internally represent the reasoning process
   emergentmind.com
   . These contemplation tokens are contentful latent vectors that summarize the logical steps, rather than an explicit text explanation. At inference, the LLM can generate a handful of such latent tokens, use them to compute an answer, and then discard them. Experiments showed that with about a 90% compression (e.g. using 10% of the tokens of an explicit CoT), the model improved exact-match accuracy by ~9 points on reasoning benchmarks
   emergentmind.com
   . The method uses LoRA to inject this capability into a pretrained model, incurring minimal overhead
   emergentmind.com
   . Suggested Architectural Idea: Train your model to produce learned latent “thought vectors” in the middle of its generation. For example, you could insert special markers where the model should output $k$ latent tokens (as a block) that represent the gist of its reasoning, then condition on them to produce the final answer. Use a training procedure to encourage these latent tokens to capture information needed for the answer (perhaps by distilling from a model that uses full CoT). Implementing this with LoRA or adapters will keep the base model mostly frozen. Why it helps: It gives you a way to compress complex reasoning or input information into a small latent space
   emergentmind.com
   . Those few latent tokens act as a distilled representation that the model can carry between models or across long contexts without large token overhead. In essence, it’s building an internal interlingua for reasoning, which is directly aligned with your shared latent goal.

8. TALL: Trainable Architecture for Low-Resource Language LLM Adaptation – PDF: arxiv:2506.05057. Summary: Ofer et al. (2025) present an architecture (TALL) that bridges a pretrained LLM with other language models via small adapters. In their case, they connect a high-resource language LLM (English) with machine translation models for a low-resource language (Hebrew)
   arxiv.org
   . The low-resource input is first translated (roughly) into the high-resource language embedding space, passed through the frozen LLM, and then translated back to the target language. Key to this are dimension alignment adapters that map between the different models’ representation spaces
   arxiv.org
   . The system uses parameter-efficient modules at each interface and keeps the large models frozen
   arxiv.org
   arxiv.org
   . TALL significantly outperformed direct prompting or naive translation pipelines on the low-resource tasks
   arxiv.org
   . Suggested Architectural Idea: Use the same principle to connect different LLMs (e.g. LLaMA and Qwen). For example, train a small encoder adapter that takes LLaMA’s latent output and transforms it into a form that Qwen can accept (perhaps as prefix tokens in Qwen’s embedding space). Likewise, a decoder adapter could map Qwen’s outputs back to LLaMA’s token space if needed. The base models remain unchanged; the adapters learn the interlingua mapping. If parallel data is available (or you generate it via one model), train the adapters to minimize reconstruction or prediction error. Why it helps: TALL demonstrates that with minimal adapters, one model’s knowledge can be channeled through another model
   arxiv.org
   arxiv.org
   . By aligning their latent representations, you essentially create a shared language between the models. This is exactly what you need for a cross-LLM interlingua, and it can be done without full re-training of either model.

9. Learning a Global Controller in Latent Space for PEFT – PDF: ACL 2024 paper (Zeqi Tan et al.). Summary: This work introduces a set of trainable global latent units that interact with a frozen LLM at all layers, acting as a smart controller for downstream tasks. The method adds a small number of learned vectors (the “controller”) that persist through the model’s forward pass and are updated via asymmetric cross-attention with each transformer layer
   openreview.net
   . Essentially, at every layer the latent controller attends to the LLM’s hidden state and vice-versa, gradually extracting and refining relevant features. They also apply knowledge distillation on hidden states to ensure the controller captures essential information
   openreview.net
   . Because the controller is much smaller than the model’s activations, this approach is memory-efficient. Empirically it achieved state-of-the-art results on multiple NLP tasks using far fewer trainable parameters than full fine-tuning
   openreview.net
   . Suggested Architectural Idea: Introduce a persistent latent vector (or a set of vectors) that accompanies the model’s computation from layer 1 to $N$. At each layer, use a cross-attention: the latent queries the layer’s hidden state and vice versa, and they exchange information. The latent vectors are the only components you update during training (along with cross-attention projections), the rest of the model stays fixed. This will effectively create a globally-aware bottleneck through which the model can route important information. Why it helps: The global controller serves as an iterative bottleneck that forces the model to summarize and propagate meaningful content in a small latent
   openreview.net
   . This can improve generalization and focus. For your use case, such a controller could continuously distill the input or context into a shared latent representation as the input is processed. That distilled latent can then be easily transferred or decoded by another model, since it concentrates the needed information in a compact form.

10. Quiet-STaR: LMs Can Teach Themselves to Think Before Speaking – PDF: arxiv:2403.09629. Summary: Quiet-STaR (Zelikman et al., 2024) is a training paradigm where language models learn to generate and use internal reasoning steps without outputting them as natural language. It builds on the Self-Taught Reasoner approach: the model is first encouraged to produce chain-of-thought explanations, then it is trained (via a form of knowledge distillation) to solve problems without explicitly showing the thoughts. In Quiet-STaR, the model uses special tokens to delimit an internal rationale at each position, effectively inferring “unstated” thoughts between the words of any text it processes
    arxiv.org
    . A novel token-wise parallel sampling and an extended teacher-forcing method are used to make this efficient
    arxiv.org
    . After a continued pre-training on open text with this method, the LM saw significant zero-shot gains on reasoning benchmarks (e.g. GSM8K math +5% absolute) without any chain-of-thought prompts, and improved perplexity on difficult tokens in natural text
    arxiv.org
    . Suggested Architectural Idea: Apply a two-stage training: first, use one of your models (or a larger teacher) to generate latent explanations or “gist” tokens for inputs; then train the other model to consume these latents and produce the output without needing the full explanation. For instance, you might generate step-by-step solutions or summaries with LLaMA, then fine-tune Qwen to take in just a compact latent (from those steps) and output the answer directly. This is analogous to learning to “think quietly.” You could also introduce special delimiter tokens and teach the model to generate a latent rationale in between, which is then used implicitly. Why it helps: It’s essentially a form of knowledge distillation into latent space – the model learns to encode the process that a verbose reasoning would follow, but in a concise internal form
    arxiv.org
    . This should improve the model’s ability to accept and make use of a given latent, because the model has been trained to align those latent vectors with meaningful intermediate computations. In short, it trains the model to understand the latent as if it were its own thought, boosting generalization and correctness.

Ranking of Approaches by Potential Uplift:

Differentiable Cache Augmentation (Latent Coprocessor) – Likely to yield the biggest gain as it directly enables a frozen model to leverage a learned latent channel
arxiv.org
. It addresses your core use-case (passing a latent between models) and has shown strong perplexity and task improvements with minimal changes to the base model.

Anchor Tokens for Latent Compression – By compressing large contexts into a few anchor tokens
arxiv.org
, you can drastically reduce the latent size while retaining meaning. This is critical for an efficient interlingua and has empirical support (99% context reduction with minor performance drop).

Compressed CoT with LoRA – Training your model to internalize reasoning into dense vectors
emergentmind.com
will directly improve its ability to generate and use a shared latent. CCoT demonstrated substantial accuracy gains on reasoning tasks with only a small latent, indicating high relevance to shared latent decoding.

Co-Finetuning Base Model to “Listen” – Partially fine-tuning or adapter-tuning your base LLM alongside the latent injector will significantly improve integration
arxiv.org
. Though it slightly violates “freeze,” the payoff in latent utilization and generalization (as seen in System2 communication results) is huge.

Multi-Depth Adaptors (IAA) – Inserting latent adapters at multiple layers will make the LLM more receptive to latent information at all levels
arxiv.org
. This architectural change can markedly improve reliability of latent transfer, as it did for multimodal inputs to frozen LLMs.

TALL Adapters for Cross-Model Alignment – When you move to multi-model, TALL’s approach of small alignment adapters between models will be invaluable
arxiv.org
arxiv.org
. It ensures that a latent produced by one model can be understood by another. This is a proven recipe for bridging different model architectures.

Prefix-Tuning+ Module – Adopting the Prefix-Tuning+ design will make any prefix/latent you use more effective by avoiding attention dilution
arxiv.org
. It’s a relatively easy swap that can boost latent injection performance, though the gains are more about stability than a new capability.

Global Latent Controller – Adding a global latent vector that persists through the network can help distill and propagate the latent information
openreview.net
. This is a more experimental idea, but it could improve the model’s handling of the latent by continuously refining it.

Coconut (Latent Iterative Reasoning) – Useful for complex tasks, it allows the model to elaborate on the latent internally via multiple passes
arxiv.org
. It’s powerful, but more complex to implement and may yield diminishing returns for simpler latent transfer scenarios (it shines mainly on tricky reasoning problems).

Quiet-STaR Distillation – While promising for enhancing reasoning, it requires a complex two-phase training and significant compute (continued pre-training)
arxiv.org
. It’s slightly tangential to straightforward latent architecture changes. Consider this a longer-term refinement: it can boost the model’s latent “thinking” ability, but the above architectural changes will be more immediately impactful for building your shared interlingua.

Great. Now for each of these ranked approaches, please provide a corresponding before and after set of diagrams showing how this changes our current approach and what needs to be added. Are these ideas aligned with our shared interlingua and approach? I want you to provide each with a downloadable design doc as well and sample code in a zip file. There should be 1 for each of the top 5 ideas you have.

Below are before/after diagrams, a short rationale, and downloadable design bundles (with a DESIGN.md + sample code + config snippets) for the top 5 ideas we ranked. Each bundle contains:

DESIGN.md with Mermaid before/after diagrams, an explanation of what changes, training signals, expected benefits/risks, and step‑by‑step integration.

sample_code/module.py with Python skeletons you can drop into your repo to prototype the idea.

config/snippets.yaml with flag examples.

PATCH_NOTES.md highlighting integration points and logging to add.

Note on your repo files: Some previously shared files (e.g., paper.tex) may have expired from my context. The downloadable bundles below are self‑contained. If you’d like me to tailor the code to your exact file layout (train.py, models.py, etc.), re‑upload the latest tree and I’ll produce exact diffs.

1. Latent Coprocessor (Differentiable Cache Augmentation) — Highest impact

Idea: Keep the base LLM frozen. Train a small coprocessor that turns the latent Z (and optionally the current hidden state) into per-layer {K,V} deltas and writes them into the LLM cache each step—so the LLM can attend to the latent at every depth.

Before (current)
(shallow prefix; LLM rarely “accepts” the latent)

flowchart LR
subgraph Current Pipeline (Before)
A[Input text] --> E[Latent Encoder (frozen/lt)]
E -->|Z (M×d_z)| Z[(Shared latent wire)]
Z --> AM1[Per-model Adapter (MLP)]
AM1 --> P1[Prefix at input-only (shallow)]
P1 --> T1[Chat Template]
T1 --> LLM1[LLM (frozen)]
end
LLM1 --> O[Answer]

After (coprocessor writes to cache)

flowchart LR
subgraph After: Latent Coprocessor
A[Input text] --> E[Latent Encoder]
E -->|Z| COP[Latent Coprocessor]
LLM[Frozen LLM]
COP -. writes .-> KV[(Augmented KV cache at multiple layers)]
LLM -->|attend to KV| LLM
LLM --> O[Answer]
end

Why this aligns with the interlingua: It makes Z a first‑class latent channel that the LLM consults at every layer, with the base model frozen—exactly what we want for a portable, shared interlingua.

What to add (high level):

LatentCoprocessor(z)->{ΔK_l, ΔV_l} for each layer l.

Hook to augment past_key_values before the LLM forward.

Train with first‑token CE, K‑token CE, and KD (teacher = text baseline with adapters disabled).

Download:
Latent Coprocessor bundle

2. Anchor Tokens for Latent Compression

Idea: Insert anchor tokens at segment boundaries and restrict attention so segments must compress into their anchor. At inference, keep only anchors’ KV as the latent—90%+ KV reduction reported.

After (anchors summarize segments)

flowchart LR
subgraph After: Anchor Tokens
T[Token stream] -->|segmentation| S1[Seg1] --> A1[(Anchor1)]
S2[Seg2] --> A2[(Anchor2)]
S3[Seg3] --> A3[(Anchor3)]
A1 --> KV[(Store only anchors' KV)]
A2 --> KV
A3 --> KV
KV --> LLM[LLM attends anchors only]
LLM --> O[Answer]
end

Interlingua fit: The anchors’ KV are the compact latent; other models can be taught to read these anchors as a shared code.

What to add:

Anchor‑aware attention mask (segment‑local for non‑anchors; anchors attend prior anchors).

KV selection to retain anchors only at inference.

Download:
Anchor Tokens bundle

3. Compressed Chain‑of‑Thought (CCoT) with LoRA

Idea: Teach the model to emit k latent “thought vectors” (continuous, not text), then consume them to answer. This compresses reasoning like CoT but with few latent tokens.

After (latent thoughts round‑trip)

flowchart LR
subgraph After: Compressed CoT (CCoT)
A[Input] --> LLM[LLM]
LLM --> T[(k latent thought tokens)]
T --> LLM
LLM --> O[Answer]
end

Interlingua fit: The thought vectors are a structured latent that can be handed to another model with a small adapter, serving as a portable interlingua for reasoning.

What to add:

A ThoughtHead producing k vectors from a hidden state.

Injection path to re‑insert those vectors (as prefix or cache) before decoding the answer.

Optional distillation from verbose CoT teacher.

Download:
CCoT bundle

4. Teach the Base to Listen (Tiny LoRA + KD)

Idea: Keep deep prefix (or latent channel), but add small LoRA (r=8–16) on early attention so the base LLM learns to listen. Use constant high first‑token CE + KD (τ≈2, K=8).

After (tiny LoRA strengthens acceptance)

flowchart LR
subgraph After: Teach Base to Listen
Z[(Shared latent)] --> P[Prefix (deep)]
P --> LORA[Tiny LoRA (early attn)]
LORA --> LLM[LLM (mostly frozen)]
LLM --> O[Answer]
end

Interlingua fit: We still pass a shared Z, but this makes per‑model interfaces robust with minimal trainables.

What to add:

Apply PEFT LoRA to Q/K/V in first 8–12 layers.

Optional adapter gating (disable LoRA on text-only batches).

KD teacher = base model with adapters disabled.

Download:
Teach Base to Listen bundle

5. Multi‑Depth Latent Adapters (IAA‑style)

Idea: Don’t inject the latent only at the input. Add small adapters at several layers (e.g., 5, 10, 15) that cross‑attend from hidden states to the latent, then write a small residual.

After (latent at multiple depths)

flowchart LR
subgraph After: Multi-Depth Adapters (IAA-style)
Z[(Shared latent)] --> ENC[Small projection]
ENC --> A5[Adapter@L5]
ENC --> A10[Adapter@L10]
ENC --> A15[Adapter@L15]
A5 --> LLM[LLM (frozen)]
A10 --> LLM
A15 --> LLM
LLM --> O[Answer]
end

Interlingua fit: Great—Z is reused across depths, improving integration in a frozen LLM.

What to add:

LatentAdapterBlock with LN → cross‑attn(h, z) → small MLP → residual.

Insert in 2–3 layers; train adapters + latent projection only.

Download:
Multi‑Depth Adapters bundle

Are these aligned with our shared interlingua?

Yes. All five preserve the core abstraction—a compact, portable latent Z—and make the LLM accept and use it more reliably:

Coprocessor and multi‑depth adapters put Z inside the attention pathway across layers (strongest acceptance).

Anchor tokens provide a highly compressed latent with explicit training pressure to summarize.

CCoT gives Z structure (thought vectors) that can be shared and decoded across models.

Tiny LoRA is a pragmatic bridge to help the base model learn to listen without abandoning the “mostly frozen” constraint.

How to choose for the next run

If you want minimal code changes and a big acceptance boost first: start with #4 Teach Base to Listen (tiny LoRA + KD), then layer #5 Multi‑Depth Adapters or #1 Coprocessor.

If you want compression wins quickly: try #2 Anchor Tokens (KV‑only anchors) once acceptance is steady.

If you want reasoning‑heavy tasks: add #3 CCoT (k latent thoughts) after acceptance stabilizes.

Notes on integration & evaluation

Keep chat templates for both models (apply_chat_template(..., add_generation_prompt=True)), and avoid double adding special tokens.

For KD: use τ≈2.0 and K=8 (first 8 steps), teacher = base model with adapters disabled.

Track: first‑token top‑1, EM/F1, KD loss, and per‑layer adapter/coprocessor norms.

If you want, I can now generate repo‑specific diffs against your train.py, models.py, and eval.py for one of these options (e.g., Latent Coprocessor or Teach Base to Listen). Just upload the current versions and tell me which option to wire first.
