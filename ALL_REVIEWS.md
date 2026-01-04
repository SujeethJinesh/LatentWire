# Claude

# LatentWire Peer Review Report

This is a comprehensive peer review of your MLSys 2025 submission. I've assembled 10 expert reviewers with diverse perspectives to provide detailed, actionable feedback.

---

## Individual Reviews

---

### Reviewer 1: MLSys Systems Expert

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 4/5
- **Recommendation**: Weak Accept

**Summary**
This paper presents LatentWire, a system that enables heterogeneous LLMs to communicate through learned continuous embeddings rather than text serialization. The core contribution is a PerceiverResampler-based bridge that compresses prompts into 8 soft tokens, achieving 27× speedup over text-relay methods on classification tasks.

**Strengths**

- Clear systems motivation: The prefill bottleneck analysis (Section 1.1) correctly identifies a real pain point in multi-LLM deployments
- Impressive throughput scaling: Figure 1 shows Bridge achieving 109 samples/sec at batch=16, nearly matching direct inference
- Text-relay batching limitation is a genuine insight: The observation that text-relay cannot effectively batch due to sequential generation is valuable
- Clean wire protocol abstraction: Constant 8KB payload regardless of input length is architecturally elegant
- Practical implementation details: Gradient checkpointing, mixed precision training, and GPU sharding are well-documented

**Weaknesses**

- The 27× speedup claim is misleading: This compares soft-token injection against _summarization + retokenization_ (text-relay), not against a fair baseline. The text-relay approach is pathologically slow by design
- Missing comparison to C2C's 2× speedup: C2C operates at KV-cache level and reports more modest but realistic speedups. Why is LatentWire 13× faster than C2C's claimed speedup?
- No end-to-end system latency breakdown: Where does time go? Encoder, adapter, prefill, generation? Table 14 shows only aggregate latencies
- Memory overhead not reported: The 537M parameter bridge must live somewhere. What's the VRAM overhead?
- No multi-GPU communication analysis: For distributed deployments, is the soft-token transfer actually faster than text when accounting for network latency?

**Detailed Critique**

_5.1 Novelty Assessment_
The systems contribution is incremental. The insight that continuous embeddings batch better than variable-length text is known. The PerceiverResampler is borrowed from Flamingo. The main novelty is applying this to LLM-to-LLM transfer, but C2C already demonstrated heterogeneous transfer via a different mechanism.

_5.2 Experimental Evaluation_
The 27× speedup needs methodological scrutiny:

- Text-relay requires Llama to _generate_ a summary (~1000ms), then Mistral to classify it
- Bridge requires only forward passes
- This comparison conflates generation vs. classification latency

A fairer comparison would be: time for Mistral to classify given (a) full text input, (b) truncated text input, (c) soft tokens. Table 14 should include direct Mistral classification as a baseline.

_5.3 Ablation Studies_
Missing systems ablations:

- Latency vs. M (soft token count)
- Memory footprint vs. model scale
- Network transfer time for distributed deployment

_5.4 Technical Soundness_
The calibration section (Equation 15) is correct but underspecified. What happens when embedding RMS varies significantly across input samples? Is per-sample calibration used?

_5.5 Writing Quality_
The systems aspects are generally well-written, but the efficiency claims in the abstract and introduction are overstated relative to the experimental evidence.

**Questions for Authors**

1. Can you provide a latency breakdown showing encoder, adapter, and generation time separately?
2. What is the VRAM overhead of the 537M bridge during inference?
3. How does latency scale with M? Is there a sweet spot?
4. Have you measured latency in a distributed setting with network transfer?
5. Why not compare against KV-cache compression methods like StreamingLLM?

**Missing References**

- StreamingLLM (Xiao et al., 2023) - efficient KV-cache management
- Flash Attention - relevant for prefill optimization context
- PagedAttention/vLLM - production inference systems

**Suggestions for Improvement**

1. Add direct classification baseline (Mistral on full text) to latency comparisons
2. Report memory overhead explicitly
3. Provide latency breakdown by component
4. Tone down "27× speedup" claims or add caveats about baseline choice
5. Add distributed systems experiments if claiming multi-LLM system benefits

---

### Reviewer 2: Prompt Tuning / Soft Prompts Specialist

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 5/5
- **Recommendation**: Borderline

**Summary**
LatentWire extends soft prompt methods to cross-model transfer, using a PerceiverResampler to compress sender hidden states into soft tokens that a receiver model can process. The paper achieves competitive classification accuracy while demonstrating bidirectional transfer.

**Strengths**

- First systematic study of cross-model soft token transfer via embedding injection (distinct from KV-cache fusion)
- Comprehensive training procedure documentation: exposure bias fix, mixed warm-up alignment, and calibration are valuable contributions
- The first-token objective (Equation 17) is a practical insight that others can use
- Strong empirical results on multi-class tasks (AG News 90.3%, TREC 94.5%)
- Adapter regularization preventing signal collapse is a useful finding

**Weaknesses**

- The 3B threshold claim is not novel: Lester et al. (2021) established ~1B+ threshold for prompt tuning efficacy. The paper claims 3B for _cross-model_ decoding but doesn't rigorously distinguish this from prior findings
- No comparison to Gist Tokens: Mu et al. (2023) achieve 26× compression within one model. Can Gist tokens be adapted for cross-model transfer? This baseline is essential
- No comparison to ICAE: Ge et al. (2024) compress to memory slots. Why not use ICAE as encoder?
- The prompt-tuning baseline is weak: Table 13 shows prompt-tuning with high variance (±4.5-9.6%). Were these properly tuned?
- Binary classification failure mode reveals fragility: The need for completely different hyperparameters (diversity loss 0.0, different M, different layer) for SST-2 suggests the method is not robust

**Detailed Critique**

_5.1 Novelty Assessment_
The PerceiverResampler for LLM-to-LLM transfer is a reasonable contribution, but the paper oversells novelty:

- Su et al. (2022) already demonstrated cross-model soft prompt transfer (RoBERTa → T5) with learned projectors
- The architectural difference (PerceiverResampler vs. linear projector) is incremental
- The "interlingua" framing suggests universality, but Section 6.12.5 shows Qwen→Mistral completely fails

The 3B threshold needs better contextualization. Lester et al. showed prompt tuning requires ~1B+ parameters. Is the 3B threshold specific to (a) cross-model transfer, (b) the PerceiverResampler architecture, or (c) soft prompt injection via inputs_embeds? Without controlled ablations isolating these factors, the threshold claim is uninterpretable.

_5.2 Experimental Evaluation_
The baselines are incomplete:

- No Gist Tokens comparison (most similar prior work)
- No ICAE comparison
- No LLMLingua comparison (discrete compression baseline)
- The "Text-Relay" baseline is artificially weak—it uses summarization rather than direct relay

The prompt-tuning baseline variance (±4.5-9.6%) is concerning. Were these run with 3 seeds like Bridge? The comparison isn't apples-to-apples if prompt-tuning used fewer seeds or less tuning.

_5.3 Ablation Studies_
Missing ablations:

- Number of PerceiverResampler layers
- Cross-attention vs. self-attention in encoder
- Source layer sweep should use multiple seeds (Table 19 appears single-seed)
- What if you used Gist-style compression on sender, then project?

_5.4 Technical Soundness_
Equation 15 (calibration) is described but the interaction with learned adapter scale (si in Equation 9) is unclear. Are both applied? In what order?

The diversity loss ablation is only for binary classification. Is diversity loss helping or hurting multi-class tasks?

_5.5 Writing Quality_
Section 3.4 is excellent—documenting training failures is valuable. However, the paper reads as two papers merged: (1) cross-model classification, (2) QA compression. The SQuAD/HotpotQA results (Section 6.10) use a different architecture and seem disconnected.

**Questions for Authors**

1. How does Gist Tokens + linear projection compare to PerceiverResampler?
2. Is the 3B threshold specific to cross-model transfer or to inputs_embeds injection?
3. Why does Qwen→Mistral fail completely? What's different about Qwen's representations?
4. Would ICAE work as an alternative encoder?
5. What happens with deeper (multi-layer) PerceiverResampler?

**Missing References**

- On Transferability of Prompt Tuning (Su et al., 2022) - directly relevant, cited but insufficiently compared
- SPoT: Better Frozen Model Adaptation (Vu et al., 2022) - prompt transfer across tasks
- Prompt Tuning for Generative Multimodal Pretrained Models (Liu et al., 2023)

**Suggestions for Improvement**

1. Add Gist Tokens baseline with cross-model projection
2. Conduct controlled experiment isolating 3B threshold cause
3. Report prompt-tuning baseline with same 3-seed protocol
4. Remove or de-emphasize disconnected SQuAD/HotpotQA results
5. Explain Qwen→Mistral failure with representation analysis

---

### Reviewer 3: Multi-Agent Systems Researcher

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 4/5
- **Recommendation**: Weak Reject

**Summary**
LatentWire proposes replacing text-based communication in multi-agent LLM systems with learned continuous embeddings. The paper demonstrates cross-model classification transfer but does not evaluate the method on actual multi-agent coordination tasks.

**Strengths**

- Addresses a real problem: text serialization is inefficient for multi-agent systems
- Constant-size communication overhead is valuable for multi-turn conversations
- Bidirectional transfer (Table 18) suggests the approach could enable peer-to-peer agent communication
- Training cost is reasonable (0.5 GPU-hours for full suite)
- The efficiency gains would be valuable for latency-sensitive multi-agent applications

**Weaknesses**

- No evaluation on multi-agent coordination tasks: The paper claims multi-agent applicability but tests only single-hop classification. No debate, negotiation, planning, or multi-turn reasoning experiments
- No comparison to LatentMAS: Zou et al. (2025) evaluate on MATH, LiveCodeBench, and HumanEval. These are the relevant benchmarks for multi-agent systems
- Classification tasks are irrelevant for agent systems: Agents need to transfer reasoning chains, plans, and structured arguments—not sentiment labels
- Per-task training requirement is impractical: Training separate bridges for each task defeats the purpose of general-purpose agent communication
- Only two-agent scenarios: Real multi-agent systems involve 3+ agents with complex communication patterns

**Detailed Critique**

_5.1 Novelty Assessment_
The multi-agent framing is misleading. The paper is really about cross-model prompt compression for classification. The introduction mentions "multi-turn conversation" but no multi-turn experiments exist.

LatentMAS achieves 4-4.3× speedup on actual reasoning benchmarks (MATH, coding) with training-free methods. LatentWire requires task-specific training and is only evaluated on classification.

_5.2 Experimental Evaluation_
The benchmark selection undermines the multi-agent claims:

- SST-2: Binary sentiment - no agent would need this
- AG News: Topic classification - same issue
- TREC: Question classification - not agent-relevant

Missing experiments that would validate multi-agent applicability:

- Multi-turn reasoning (MMLU-Pro, ARC-Challenge)
- Code generation (HumanEval, MBPP)
- Mathematical reasoning (GSM8K, MATH)
- Planning/coordination (WebArena, AgentBench)
- Multi-agent debate (established debate benchmarks)

_5.3 Ablation Studies_
No ablations relevant to multi-agent systems:

- How does performance degrade over multiple communication rounds?
- Can 3+ agents share the same interlingua?
- What's the information loss in round-trip (A→B→A) communication?

_5.4 Technical Soundness_
The semantic compression nature (Section 8.4) is a fundamental limitation for agents. Agents often need to communicate:

- Exact values (tool parameters, API arguments)
- Structured data (JSON, code)
- Precise logical constraints

The paper acknowledges this but doesn't address it.

_5.5 Writing Quality_
The abstract and introduction oversell multi-agent applicability. Phrases like "multi-LLM systems," "telepathically," and "wire protocol" suggest general applicability, but experiments are narrow.

**Questions for Authors**

1. Have you evaluated on any reasoning benchmarks (GSM8K, MATH)?
2. What happens with 3+ agents sharing the interlingua?
3. How does information degrade over multiple round-trips?
4. Can you show any multi-turn experiment where agents genuinely coordinate?
5. What's the plan for handling exact value transfer alongside semantic transfer?

**Missing References**

- CAMEL (Li et al., 2023) - role-playing multi-agent communication
- AgentVerse (Chen et al., 2023) - multi-agent collaboration
- AutoGen (Wu et al., 2023) - multi-agent conversation framework
- Debate improves reasoning (Du et al., 2023)

**Suggestions for Improvement**

1. Either (a) add multi-agent experiments or (b) remove multi-agent framing from title/abstract
2. Evaluate on GSM8K or MATH to compare with LatentMAS
3. Test multi-turn conversation with cumulative compression
4. Add 3-agent experiment with shared interlingua
5. Develop hybrid approach for semantic + exact communication

---

### Reviewer 4: Representation Learning Theorist

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 4/5
- **Recommendation**: Weak Accept

**Summary**
LatentWire demonstrates that learned soft tokens can transfer task-relevant information across heterogeneous LLM families. The paper invokes the Platonic Representation Hypothesis to explain why cross-model mapping succeeds and proposes a 3B parameter capacity threshold.

**Strengths**

- The t-SNE visualization (Figure 2) provides evidence that the learned latent space is semantically organized
- Empirical validation of cross-family transfer (Llama→Mistral, Mistral→Llama) is valuable
- The capacity threshold finding, while perhaps not novel, is systematically documented
- The paper correctly notes that discrete quantization (VQ-VAE) fails due to manifold structure destruction
- Bidirectional asymmetry (97.0% vs. 91.5% on SST-2) is an interesting empirical observation

**Weaknesses**

- The Platonic Representation connection is superficial: The paper cites Huh et al. but doesn't measure representation alignment. No CKA, SVCCA, or linear probe analyses
- No analysis of what makes representations "compatible": Llama↔Mistral works, Qwen→Mistral fails. Why? The paper offers no representational explanation
- The 3B threshold is not theoretically grounded: Equation 22 is speculative ("we hypothesize") with no derivation or validation
- Missing intrinsic dimension analysis: What's the intrinsic dimension of the soft token space? Is it lower for successful transfers?
- No cross-model representation similarity analysis: Are Llama and Mistral representations actually aligned before training?

**Detailed Critique**

_5.1 Novelty Assessment_
The representation learning contribution is limited. The paper demonstrates cross-model transfer empirically but doesn't analyze _why_ it works. Key questions unanswered:

- What's the representation similarity between Llama and Mistral before/after bridge training?
- Is the bridge learning an alignment, or something more complex?
- What features does the bridge preserve vs. discard?

Moschella et al. (2023) showed that relative representations enable zero-shot stitching. How does LatentWire compare? Could their method work here without training?

_5.2 Experimental Evaluation_
Missing analyses:

- Centered Kernel Alignment (CKA) between sender and receiver representations
- Linear probing of soft tokens for semantic features
- Representation similarity analysis (before/after bridge)
- Comparison to zero-shot alignment methods (Moschella et al.)

_5.3 Ablation Studies_
The source layer ablation (Table 19) is valuable but needs theoretical interpretation. Why does layer 16 work best for multi-class but layer 28 for binary? What does this suggest about representation structure?

_5.4 Technical Soundness_
The calibration procedure (matching RMS) is empirically motivated but theoretically unmotivated. Why should RMS matching work? Is it because:

- Embedding norms correlate with semantic salience?
- Models use similar activation scales?
- Something else?

_5.5 Writing Quality_
Section 2.5 mentions the Platonic Representation Hypothesis but doesn't follow through. Either remove this connection or add representation analysis to support it.

**Questions for Authors**

1. What's the CKA similarity between Llama layer 16 and Mistral embedding space?
2. Can you perform linear probing on the 8 soft tokens to verify semantic content?
3. Why does Qwen→Mistral fail? Is Qwen's representation space structured differently?
4. Could Relative Representations (Moschella et al.) provide a training-free baseline?
5. What's the intrinsic dimension of successful vs. failed soft token spaces?

**Missing References**

- Understanding Learned Representations (Kornblith et al., 2019) - CKA analysis
- Similarity of Neural Network Representations (Raghu et al., 2017) - SVCCA
- CLIP's representation structure (Radford et al., 2021)
- Cross-model Transferability of Steering Vectors (arXiv:2501.02009) - 90% cross-family success

**Suggestions for Improvement**

1. Add CKA analysis between sender and receiver representations
2. Perform linear probing on soft tokens
3. Analyze Qwen→Mistral failure with representation tools
4. Either substantiate Platonic Representation connection or remove it
5. Compare to zero-shot alignment methods

---

### Reviewer 5: Skeptical Senior Reviewer

**Overall Assessment**

- **Score**: 4/10
- **Confidence**: 5/5
- **Recommendation**: Reject

**Summary**
This paper applies known techniques (PerceiverResampler, soft prompts, embedding injection) to a new setting (cross-model transfer) and demonstrates classification accuracy on three simple benchmarks. The novelty is incremental and several claims are overstated.

**Strengths**

- Thorough documentation of training challenges and solutions
- The first-token objective fix is a practical contribution
- Bidirectional transfer is demonstrated
- Code/reproducibility details appear sufficient
- Training efficiency is reasonable

**Weaknesses**

- "Interlingua" framing is overclaimed: An interlingua should be universal. Section 6.12.5 shows Qwen→Mistral fails completely. This isn't an interlingua; it's a model-specific bridge
- The 27× speedup is against a straw-man baseline: Text-relay requires generation; Bridge requires only forward passes. This is not a fair comparison
- Classification tasks are too simple to demonstrate the claimed benefits: Any decent embedding + linear probe would achieve similar results
- The 3B threshold is not rigorously established: Table 4 shows 0.5B→1.1B→3B with only one model per size. Is this a threshold or sampling variance?
- High variance in results: Table 13 shows ±4-9.6% standard deviation across seeds, which undermines precision claims

**Detailed Critique**

_5.1 Novelty Assessment_
I dispute the novelty claims:

1. "Cross-model soft token transfer via embedding injection" - Su et al. (2022) already did this with linear projectors. The PerceiverResampler is borrowed from Flamingo. The combination is engineering, not research contribution.

2. "27× faster inference" - This compares soft-token forward pass against text generation + classification. Of course forward passes are faster than generation.

3. "3B parameter threshold" - Lester et al. (2021) showed prompt tuning works above ~1B parameters. The paper doesn't isolate whether 3B is specific to cross-model transfer.

4. "Interlingua" - An interlingua is universal by definition. Qwen→Mistral fails. This is a Llama↔Mistral bridge.

_5.2 Experimental Evaluation_
The evaluation has significant gaps:

**Missing baselines:**

- Linear probe on sender hidden states (∼4K params, likely achieves similar accuracy)
- Gist Tokens + projection
- Direct Mistral few-shot (Table 13 shows 5-shot but only for Mistral)
- ICAE compression baseline

**Benchmark concerns:**

- SST-2, AG News, TREC are solved benchmarks. State-of-the-art is 97%+
- No reasoning tasks (GSM8K, MATH) where compression might matter
- No generation tasks—only classification

**Statistical rigor:**

- 3 seeds is minimal
- ±5% variance means results could flip with different seeds
- No significance testing

_5.3 Ablation Studies_
The ablations are incomplete:

- Table 19 (source layer) appears to be single-seed
- No ablation on PerceiverResampler depth
- No ablation on training data size
- The binary classification "fix" required 6 simultaneous changes—which one(s) actually mattered?

_5.4 Technical Soundness_
Several technical concerns:

- The zero-prefix control (Table 3) shows models generate coherent text from anchor alone. How do we know the soft tokens are actually used?
- Attention analysis is missing: Do receiver attention heads attend to soft tokens?
- The calibration + adapter scale interaction is confusing

_5.5 Writing Quality_
The paper oversells throughout:

- Abstract: "telepathically" is hyperbolic
- "Wire protocol" suggests standardization that doesn't exist
- "Interlingua" claims universality that isn't demonstrated

**Questions for Authors**

1. How does a linear probe on Llama hidden states compare to the full Bridge?
2. Can you show attention patterns proving soft tokens are attended to?
3. Why not report significance tests (p-values)?
4. With 6 changes needed for SST-2, how do you know which mattered?
5. If Qwen→Mistral fails, how is this an "interlingua"?

**Missing References**

- Linear probing literature (Hewitt & Manning, 2019)
- Attention analysis methods (Clark et al., 2019)

**Suggestions for Improvement**

1. Add linear probe baseline—if it matches Bridge, the contribution collapses
2. Remove "interlingua" framing or demonstrate Qwen/Gemma/Phi-3 compatibility
3. Add attention analysis showing soft tokens are used
4. Report significance tests
5. Evaluate on a reasoning benchmark (GSM8K)
6. Tone down speedup claims or add fairer baselines

---

### Reviewer 6: Reproducibility & Methodology Expert

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 5/5
- **Recommendation**: Weak Accept

**Summary**
The paper presents LatentWire with reasonable reproducibility details but has several methodological concerns regarding baseline fairness, seed selection, and ablation design.

**Strengths**

- Training hyperparameters fully specified (Table 22)
- Model versions explicitly stated (Section A.5)
- Training dynamics documented (Section A.2)
- Computational requirements reported (Table 23)
- Random seed specified (42 for main experiments)

**Weaknesses**

- Only 3 seeds for main results—minimum acceptable but not robust
- Ablations appear to use single seeds (Tables 17, 19, 20)
- Binary classification required 6 simultaneous changes with no factorial analysis
- Prompt-tuning baseline variance (±4.5-9.6%) suggests insufficient tuning
- Text-relay baseline is not fairly constructed

**Detailed Critique**

_5.1 Novelty Assessment_
N/A for this reviewer.

_5.2 Experimental Evaluation_
**Baseline fairness concerns:**

The text-relay baseline is artificially weak:

- It requires Llama to _generate_ a summary (slow)
- Then Mistral classifies the summary
- This conflates generation latency with the core comparison

A fairer baseline would be:

- Mistral classifies full text (no relay)
- Mistral classifies truncated text (matched token budget)
- Bridge

**Seed selection:**

- Main results (Table 13) use 3 seeds—acceptable minimum
- Ablations (Tables 17, 19, 20, 21) appear single-seed—insufficient
- Table 17 (soft token scaling) shows SST-2 at 86.5% for ALL values of M (2, 4, 8, 16, 32). This is suspicious—either the metric is insensitive or there's an error

**Variance handling:**

- ±5% variance is high for classification benchmarks
- No confidence intervals or significance tests
- Prompt-tuning baseline variance (±4.5-9.6%) exceeds Bridge variance—was it properly tuned?

_5.3 Ablation Studies_
**Factorial design issues:**

The SST-2 fix involved 6 simultaneous changes:

1. Diversity loss: 0.0
2. Soft tokens: M=4
3. Learning rate: 5e-4
4. Training steps: 4000
5. Source layer: 24
6. Class-balanced sampling

Without factorial or sequential ablation, we don't know which change(s) mattered. The paper acknowledges this ("we did not conduct rigorous single-factor ablations") but this is a significant gap.

**Missing ablations:**

- PerceiverResampler depth
- Training data size
- Different random seeds for each ablation

_5.4 Technical Soundness_
The zero-prefix control is valuable but incomplete. It shows models generate coherent text when prefix is zeroed, but doesn't prove soft tokens are actively used when present. Attention analysis would strengthen this.

_5.5 Writing Quality_
Documentation is generally good. Section 3.4 (training challenges) is exemplary—more papers should document failures.

**Questions for Authors**

1. Can you run ablations with 3 seeds and report variance?
2. Can you do a factorial analysis of the 6 SST-2 changes?
3. Why does SST-2 achieve identical accuracy (86.5%) for all M values in Table 17?
4. How was the prompt-tuning baseline tuned? Why such high variance?
5. Can you add attention analysis to prove soft tokens are attended to?

**Missing References**

- ML Reproducibility Checklist (NeurIPS guidelines)
- Dodge et al. (2020) - fine-tuning variance in NLP

**Suggestions for Improvement**

1. Re-run all ablations with 3 seeds
2. Add factorial analysis of SST-2 fix
3. Add fairer text baseline (direct classification)
4. Report confidence intervals
5. Add attention visualization

---

### Reviewer 7: NLP Applications Researcher

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 5/5
- **Recommendation**: Borderline

**Summary**
LatentWire demonstrates cross-model classification transfer on SST-2, AG News, and TREC benchmarks. While results are competitive, the benchmark selection is narrow and doesn't demonstrate the claimed benefits for multi-LLM systems.

**Strengths**

- Strong AG News performance (90.3%) exceeds prompt-tuning baseline
- TREC performance (94.5%) is impressive for cross-model transfer
- Bidirectional transfer works (Mistral→Llama achieves 97.0% on SST-2)
- Task-aware configuration (binary vs. multi-class) is practical
- t-SNE visualization (Figure 2) shows semantic structure

**Weaknesses**

- Benchmark selection is too narrow: SST-2, AG News, TREC are simple classification tasks from 10+ years ago
- No NLU benchmarks: GLUE/SuperGLUE would provide standardized comparison
- No generation tasks: The paper claims "wire protocol" applicability but only tests classification
- No reasoning tasks: GSM8K, MATH, ARC would test semantic compression limits
- The Qwen→Mistral failure (random chance accuracy) limits practical applicability

**Detailed Critique**

_5.1 Novelty Assessment_
From an NLP perspective, the novelty is limited:

- Cross-model transfer has been studied (Su et al., 2022)
- Soft prompts are well-established
- The application to classification is straightforward

_5.2 Experimental Evaluation_
**Benchmark concerns:**

SST-2:

- Binary sentiment is trivial—BERT achieved 94% in 2018
- SOTA is 97%+
- 91.5% is not competitive with modern methods

AG News:

- 4-class topic classification
- SOTA is 95%+
- 90.3% is reasonable but not impressive

TREC:

- 6-class question classification
- Small test set (500 questions)
- 94.5% is strong

**Missing evaluations:**

- GLUE benchmark (standardized NLU comparison)
- MMLU (knowledge/reasoning)
- HellaSwag (commonsense)
- Reading comprehension (SQuAD results in paper are weak)
- Generation quality (summarization, translation)

_5.3 Ablation Studies_
The source layer ablation (Table 19) is useful for understanding what information different layers encode. However, the NLP interpretation is missing—what linguistic features are captured at layer 16 vs. layer 24?

_5.4 Technical Soundness_
The method is sound for classification. The claim that this is a "wire protocol" for arbitrary communication is not supported by the evaluation.

_5.5 Writing Quality_
The paper could better contextualize results against NLP SOTA. Simply stating "91.5% on SST-2" without noting that SOTA is 97% is misleading.

**Questions for Authors**

1. How does Bridge perform on GLUE benchmark?
2. Can you show any generation task (summarization, translation)?
3. What linguistic features are captured in the soft tokens?
4. How does performance scale with number of classes beyond 6?
5. Would the method work for token-level tasks (NER, POS)?

**Missing References**

- BERT (Devlin et al., 2019) - SST-2 performance context
- GLUE/SuperGLUE benchmarks
- Sentence-BERT (Reimers & Gurevych, 2019) - alternative embedding approach
- SetFit (Tunstall et al., 2022) - few-shot classification baseline

**Suggestions for Improvement**

1. Add GLUE benchmark evaluation
2. Include at least one generation task
3. Context results against SOTA (e.g., "91.5% vs. 97% SOTA on SST-2")
4. Add linguistic analysis of what soft tokens encode
5. Test on token-level tasks

---

### Reviewer 8: Efficiency/Compression Specialist

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 4/5
- **Recommendation**: Weak Accept

**Summary**
LatentWire achieves 15-30× prompt compression by learning soft token representations that transfer across model families. The compression-quality tradeoff is competitive with single-model methods while enabling cross-model communication.

**Strengths**

- 15-30× compression while maintaining classification accuracy is impressive
- Comparison to PCA baseline (Table 11) validates the need for learned compression
- The training-free baseline (Table 21) proves learning is essential
- Memory overhead analysis (Appendix A.6) is thorough
- Break-even analysis is practical

**Weaknesses**

- No comparison to LLMLingua: Discrete compression achieves up to 20× with black-box APIs. How does LatentWire compare?
- No comparison to 500xCompressor: Li et al. achieve 6-480× compression
- The 537M bridge is expensive: This is 7% of Mistral-7B. Is the compression worth the parameter overhead?
- Fixed M=8 is inflexible: Variable-length compression would be more practical
- No quality-compression curve: What's the accuracy at 4×, 8×, 16×, 32× compression?

**Detailed Critique**

_5.1 Novelty Assessment_
The compression contribution is incremental over single-model methods:

- Gist Tokens achieve 26× within one model
- ICAE achieves 4-15×
- 500xCompressor achieves 6-480×

LatentWire's contribution is enabling cross-model transfer, not better compression.

_5.2 Experimental Evaluation_
**Missing baselines:**

- LLMLingua (discrete compression, black-box compatible)
- LLMLingua-2 (faster compression)
- 500xCompressor (higher compression)
- AutoCompressors (learned summary vectors)

**Missing analyses:**

- Quality vs. compression curve (vary M from 2 to 64)
- Compression overhead breakdown (encoder time vs. savings)
- Comparison of actual information content (bits per sample)

_5.3 Ablation Studies_
Table 17 (soft token scaling) is useful but limited:

- Only tests M ∈ {2, 4, 8, 16, 32}
- Single seed
- SST-2 shows identical accuracy for all M—suspicious

_5.4 Technical Soundness_
The claim of "15-30× compression" needs clarification:

- Text baseline: ~300 tokens
- Soft tokens: 8 vectors
- Token-count compression: 300/8 = 37.5×
- Information compression: 8 × 4096 floats vs. 300 tokens

Are we comparing apples to apples? 8 × 4096 = 32,768 float values is more raw information than 300 discrete tokens.

_5.5 Writing Quality_
The efficiency claims could be clearer about what exactly is being compressed (tokens, bits, information).

**Questions for Authors**

1. How does LatentWire compare to LLMLingua at similar compression ratios?
2. What's the accuracy at 64× compression (M=4, dz=256)?
3. Is the 537M bridge parameter overhead accounted for in efficiency claims?
4. Can you plot a Pareto curve of accuracy vs. compression?
5. What's the actual bits-per-sample compression ratio?

**Missing References**

- LLMLingua (Jiang et al., 2023) - discrete compression
- LLMLingua-2 (Pan et al., 2024) - cited but not compared
- 500xCompressor (Li et al., 2024) - cited but not compared
- CONTEXT: Learning Fine-Grained Concepts (Snell et al., 2022)

**Suggestions for Improvement**

1. Add LLMLingua and 500xCompressor comparisons
2. Plot accuracy vs. compression Pareto curve
3. Clarify what "compression" means (tokens, bits, floats)
4. Consider variable-M approach
5. Report bits-per-sample metrics

---

### Reviewer 9: Writing & Presentation Critic

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 4/5
- **Recommendation**: Borderline

**Summary**
The paper presents LatentWire with generally clear exposition but has significant writing issues including hyperbolic claims, structural inconsistencies, and missing contextualization of results.

**Strengths**

- Section 3.4 (Training Challenges and Solutions) is exemplary documentation
- Figures are generally clear
- Mathematical notation is consistent
- Appendices provide good supplementary detail
- Table formatting is professional

**Weaknesses**

- Abstract contains hyperbolic language ("telepathically")
- "Interlingua" framing is misleading given Qwen failure
- Results lack SOTA context
- Structural inconsistency between classification and QA sections
- Several sentences are awkward or unclear

**Detailed Critique**

_5.5 Writing Quality_

**Abstract issues:**

- "communicate telepathically" is hyperbolic and informal
- "Wire protocol" suggests a standard that doesn't exist
- Should mention Qwen→Mistral failure for honesty

**Title concerns:**

- "Interlingua" implies universality not demonstrated
- "Heterogeneous LLM Communication" is accurate but vague
- Suggested alternative: "LatentWire: Cross-Model Soft Token Transfer for LLM Classification"

**Sentence-level issues:**

Section 1 (Introduction):

- "What if models could communicate telepathically" — Too informal for academic paper. Suggest: "We demonstrate that models can share compressed semantic representations directly."

Section 1.2:

- "Our key contributions:" list has inconsistent parallel structure. Item 1 uses "via," items 2-3 use present tense, item 4 uses "We identify."

Section 2.5:

- "Our differentiation: LatentWire operates in a distinct architectural space" — This is vague. Be specific: "Unlike C2C, which fuses KV-caches, LatentWire injects soft tokens via inputs_embeds."

Table 13 caption:

- "Best results per dataset in bold" but SST-2's best is Mistral 5-shot (96.3%), not Bridge (91.5%)

Section 6.11:

- "Why Bridge beats Prompt-Tuning" — Claims victory on 2/3 tasks but presents this as universal insight

**Structural issues:**

- Sections 6.1-6.9 (SQuAD/HotpotQA) feel disconnected from 6.11 (classification)
- The paper reads like two separate papers merged
- Consider moving SQuAD results to appendix

**Missing context:**

- SST-2 SOTA is 97%+, not mentioned
- AG News SOTA is 95%+, not mentioned
- The 3B threshold should be compared to Lester et al.'s 1B finding

**Figure quality:**

- Figure 1: Y-axis doesn't start at 0, exaggerating differences
- Figure 2: t-SNE visualization is useful but lacks quantitative cluster quality metric

**Notation:**

- M is used inconsistently (sometimes soft token count, sometimes a set)
- dz vs di subscript convention is clear but could be stated explicitly

**Questions for Authors**

1. Will you remove "telepathically" from the abstract?
2. Can you reconcile "interlingua" framing with Qwen failure?
3. Will you add SOTA context for benchmark results?
4. Can Figure 1 y-axis start at 0?
5. Can you unify the paper's focus (classification OR QA, not both)?

**Suggestions for Improvement**

1. Remove informal language ("telepathically")
2. Either rename from "interlingua" or demonstrate broader model coverage
3. Add SOTA context (e.g., "91.5% vs. 97% SOTA")
4. Move SQuAD/HotpotQA to appendix to improve focus
5. Fix parallel structure in contribution list
6. Start Figure 1 y-axis at 0
7. Add quantitative cluster quality metric to Figure 2

---

### Reviewer 10: Industry Practitioner

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 4/5
- **Recommendation**: Weak Reject

**Summary**
LatentWire proposes replacing text-based multi-LLM communication with learned soft tokens. While the efficiency gains are attractive, significant practical limitations make deployment challenging.

**Strengths**

- Training cost is reasonable (30 min for 3 tasks on H100)
- 27× latency improvement would be valuable in production
- Constant-size payload simplifies infrastructure
- Frozen LLMs mean no model modification required
- Batch processing capability is important for throughput

**Weaknesses**

- Per-task training defeats the purpose: Production systems need general-purpose bridges, not task-specific ones
- Model pair specificity is a showstopper: Qwen→Mistral fails completely. What about other model pairs?
- 537M bridge is significant overhead: For Mistral-7B, this is 7.7% additional parameters
- 3B minimum is restrictive: Many edge deployments use smaller models (Phi-3, Gemma-2B)
- No API compatibility: The method requires inputs_embeds access, unavailable in most APIs

**Detailed Critique**

_Practical Deployment Concerns:_

**1. Model lock-in:**
Production systems need flexibility to swap models. LatentWire requires:

- Per-model-pair bridges
- Per-task training
- Validation of each new combination

If you want to add GPT-4 to your system, you need new bridges. This is impractical.

**2. API limitations:**
Most LLM APIs (OpenAI, Anthropic, Cohere) don't expose inputs_embeds. LatentWire only works with:

- Self-hosted models
- HuggingFace-style APIs
- Custom deployments

This limits applicability to organizations with significant ML infrastructure.

**3. Maintenance burden:**
The 537M bridge needs:

- Training compute for each task/model pair
- Validation infrastructure
- Version management
- Performance monitoring

For N models and T tasks, you need O(N²T) bridges. This doesn't scale.

**4. Failure modes:**
Qwen→Mistral produces random chance accuracy. In production, silent failures are dangerous. How would you detect when a bridge is failing?

**5. Debugging difficulty:**
Soft tokens are uninterpretable. When classification fails, how do you debug? Text-based communication is human-readable.

_Cost-Benefit Analysis:_

**Costs:**

- Training: 0.5 GPU-hours per task × model pairs
- Storage: 537M params × 4 bytes × model pairs
- Validation: Unknown but significant
- Monitoring: Need anomaly detection for soft token quality

**Benefits:**

- 27× latency reduction (but text baseline is artificial)
- Constant payload size
- Batch processing

**Break-even:**
For the benefits to outweigh costs, you need:

- High-volume multi-LLM communication
- Latency-sensitive applications
- Self-hosted models
- Willingness to maintain bridges

This is a narrow use case.

**Questions for Authors**

1. What's the failure detection strategy for bridges?
2. How would you handle model version updates?
3. Have you tested with API-based models at all?
4. What's the recommended deployment architecture?
5. How do you handle bridge training in production (online? offline?)

**Missing References**

- Production LLM deployment papers
- Model serving systems (vLLM, TensorRT-LLM)
- Multi-model orchestration (Ray Serve, BentoML)

**Suggestions for Improvement**

1. Add failure detection/monitoring discussion
2. Provide deployment architecture recommendations
3. Discuss API compatibility limitations explicitly
4. Add model version update strategy
5. Provide cost-benefit calculator for practitioners
6. Test with at least 4 model families to establish generality

---

## Meta-Review

### Consensus Summary

All reviewers agree on the following:

1. **The core idea is sound:** Cross-model soft token transfer via PerceiverResampler is a valid approach that demonstrably works for Llama↔Mistral classification tasks.

2. **Documentation is strong:** Section 3.4 (Training Challenges) is uniformly praised. The paper's honesty about failures (SST-2 initial failure, Qwen→Mistral failure) is appreciated.

3. **The 27× speedup claim is problematic:** The comparison against text-relay (which requires generation) is not a fair baseline. Reviewers want comparison against direct classification.

4. **Benchmark selection is too narrow:** SST-2, AG News, and TREC are simple classification tasks that don't demonstrate the claimed multi-agent applicability.

5. **Missing key baselines:** Gist Tokens, LLMLingua, ICAE, and linear probes are missing comparisons that would contextualize the contribution.

### Key Controversies

1. **Novelty significance:**

   - Reviewers 1, 4, 6 lean positive (incremental but useful)
   - Reviewers 5, 10 lean negative (engineering combination, not research)

2. **"Interlingua" framing:**

   - Reviewers 3, 5, 9, 10 object strongly (Qwen fails, so not universal)
   - Reviewers 2, 4 see it as acceptable with caveats

3. **3B threshold contribution:**

   - Reviewer 5 argues Lester et al. already established this
   - Reviewer 2 sees potential novelty if properly isolated to cross-model

4. **Venue appropriateness:**
   - Reviewers 1, 8 see MLSys fit (systems contribution)
   - Reviewers 3, 7 see better fit at ACL/EMNLP (NLP application)

### Critical Issues

**Must fix for any venue:**

1. **Add fairer baselines:**

   - Direct Mistral classification (no relay)
   - Linear probe on sender hidden states
   - Gist Tokens + projection

2. **Fix speedup claims:**

   - Either add caveats about baseline choice or use fairer comparison
   - Provide latency breakdown by component

3. **Address model pair specificity:**

   - Explain Qwen→Mistral failure
   - Either test more model pairs or remove "interlingua" framing

4. **Add statistical rigor:**
   - Run ablations with 3 seeds
   - Add significance tests
   - Report confidence intervals

### Venue Recommendation

**MLSys is marginally appropriate** but the systems contribution is limited. The paper's strength is the NLP application (cross-model transfer for classification), not systems innovation.

**Better venues:**

- **ACL/EMNLP** (primary recommendation): Cross-model transfer is an NLP contribution
- **NeurIPS** (if expanded): With broader evaluation on reasoning benchmarks
- **ICLR** (if representation analysis added): With CKA/SVCCA analysis

### Overall Verdict

**Weak Reject** with potential for acceptance after major revision.

The paper demonstrates a working system for cross-model classification transfer, which is a useful contribution. However, the overclaims (27× speedup, "interlingua," multi-agent applicability), missing baselines, and narrow evaluation significantly weaken the submission.

With the revisions outlined below, this could become a solid accept at ACL/EMNLP or a weak accept at MLSys.

### Revision Roadmap

**Critical (must fix for any venue):**

1. ☐ Add linear probe baseline on sender hidden states
2. ☐ Add direct Mistral classification baseline for latency comparison
3. ☐ Add Gist Tokens + projection baseline
4. ☐ Run all ablations with 3 seeds and report variance
5. ☐ Either test 2+ additional model families OR remove "interlingua" terminology
6. ☐ Add caveats to 27× speedup claim explaining baseline choice
7. ☐ Remove "telepathically" and other hyperbolic language

**Major (significantly strengthens paper):**

8. ☐ Add at least one reasoning benchmark (GSM8K or MATH)
9. ☐ Add attention visualization showing soft tokens are attended to
10. ☐ Analyze Qwen→Mistral failure with representation tools (CKA)
11. ☐ Add LLMLingua comparison at similar compression ratios
12. ☐ Provide factorial analysis of the 6 SST-2 fixes
13. ☐ Add SOTA context for benchmark results (e.g., "vs. 97% SOTA")
14. ☐ Move SQuAD/HotpotQA results to appendix (focus paper on classification)

**Minor (nice to have):**

15. ☐ Add latency breakdown by component
16. ☐ Report memory overhead explicitly
17. ☐ Add accuracy vs. compression Pareto curve
18. ☐ Fix Figure 1 y-axis to start at 0
19. ☐ Add cluster quality metric to Figure 2
20. ☐ Discuss failure detection for production deployment

---

I hope this comprehensive review helps strengthen your paper, Sujeeth. The core work is solid—the main issues are framing and missing baselines rather than fundamental problems with the approach. Happy to discuss any specific points in more detail.

# ChatGPT

# LatentWire Peer Review Report

_(Committee of 10 independent reviewers; each uses the same paper text but evaluates from a different lens. I cite specific parts of your PDF where relevant, and I also cite the related-work papers you listed when I can verify them online.)_

---

## Individual Reviews

### Reviewer 1: MLSys Systems Expert

#### 1. Overall Assessment

- **Score:** 6/10
- **Confidence:** 4/5
- **Recommendation:** Weak Accept

#### 2. Summary (2–3 sentences)

LatentWire proposes a “wire protocol” for multi-LLM systems: compress a sender’s hidden states into a constant-size set of learned soft tokens and inject them into a receiver via `inputs_embeds`, avoiding text relay. It reports large end-to-end latency gains versus an explicit Text-Relay baseline (≈27×) and strong accuracy on three classification datasets while keeping both LLMs frozen.

#### 3. Strengths (≥5)

- Clear **systems framing** (prefill bottleneck, batching limitations of text relay, constant-size communication).
- Strong **end-to-end latency numbers** vs Text-Relay with measurement notes and variance.
- Includes **throughput scaling with batch size**; identifies “uniform M” enabling batching.
- Concrete **training cost accounting** (minutes per task; GPU-hours).
- Cross-model communication is shown to sometimes **improve** downstream performance (AG News, TREC) vs prompt tuning.

#### 4. Weaknesses (≥5)

- The headline **27× speedup** is vs a Text-Relay pipeline that includes autoregressive summarization; it is not directly comparable to other “speedup” papers (e.g., KV-cache fusion).
- Evaluation uses **only 200 samples per dataset** in the main setup, which is too small for MLSys-grade claims about accuracy/robustness.
- The bridge is **large** (≈537M params) and the paper sometimes sells it as “small external adapters,” which will raise eyebrows from systems reviewers.
- Latency results are on **one GPU and one hardware type** (H100); no sensitivity to CPU offload, multi-GPU, memory pressure, or serving stack overhead.
- No real **serving/production story** (cacheability of soft tokens, concurrency, multi-tenant isolation, failure handling, monitoring).

#### 5. Detailed Critique

##### 5.1 Novelty Assessment

- Embedding-level injection (`inputs_embeds`) as a communication channel is meaningfully different from KV-cache fusion methods like Cache-to-Cache. ([arXiv][1])
- Still, novelty risk: some reviewers may see this as “a particular adapter + soft prompt transfer,” so you must sharpen the “systems protocol” novelty vs prior cross-model transfer and “LLM modules.” ([arXiv][2])

##### 5.2 Experimental Evaluation

- Baselines are directionally reasonable (Text, Token-budget, Prompt-tuning, etc.).
- But the small test size (200) plus high std (±4–6%) undermines claims like “exceeds prompt tuning on 2/3 tasks.”
- Latency: good that models are preloaded, but MLSys will ask about “p99 under concurrency,” and whether Text-Relay is optimized fairly (e.g., same batching effort, same decoding params).

##### 5.3 Ablation Studies

- Good hierarchy of ablations (adapter reg, calibration, first-token objective, anchor/BOS).
- Missing: ablate “Text-Relay” summarizer quality vs latency; if you shorten summaries to match accuracy, how does speedup move?

##### 5.4 Technical Soundness

- Method seems technically plausible; the calibration/scale mismatch discussion is credible.
- But from a systems perspective, you must explain why a 537M bridge is justified vs smaller projections for classification (tradeoffs already hinted).

##### 5.5 Writing Quality

- “Telepathically” is memorable but may read as “bloggy” for MLSys.
- Several speedup claims are scattered; consolidate into one “Measurement Protocol” box.

#### 6. Questions for Authors (3–5)

1. What is the **exact Text-Relay pipeline** (summary length, decoding params, prompt templates), and is it tuned for latency?
2. Can you report latency under **concurrency** (e.g., 16–64 in-flight requests) and p95/p99?
3. How do results change on the **full test sets**, not 200 samples?
4. Can soft tokens be **cached** per document/turn, and reused across multiple downstream agents?

#### 7. Missing References

- Add more direct comparison discussion to “cross-model prompt transfer” work beyond Su et al. (2022) and “modules” style approaches (you cite some, but the narrative could be stronger). ([arXiv][3])

#### 8. Suggestions for Improvement

- Add a **serving evaluation**: throughput vs concurrency, memory footprint, cache reuse, and a small “agent loop” benchmark.
- Replace “27× faster” headline with a more precise phrase: “27× vs autoregressive Text-Relay summarization baseline under X constraints.”

---

### Reviewer 2: Prompt Tuning / Soft Prompts Specialist

#### 1. Overall Assessment

- **Score:** 7/10
- **Confidence:** 4/5
- **Recommendation:** Accept

#### 2. Summary

The paper extends soft-prompt ideas to cross-family LLM communication by learning a PerceiverResampler that compresses sender hidden states into M soft tokens and injecting them into a frozen receiver. It reports strong classification accuracy with large compression and highlights a model-capacity threshold for decoding continuous prompts.

#### 3. Strengths

- Framed as **wire protocol**, not just “another prompt tuning trick.”
- Includes **prompt tuning baseline** and reports mean±std across seeds.
- Identifies and fixes a real optimization pitfall (diversity loss hurting binary tasks).
- Strong ablation breadth (calibration, BOS, anchor, M, source layer, model pairs).
- Good discussion of why single-model compressors (Gist/ICAE) don’t solve hetero tokenization/embedding mismatch.

#### 4. Weaknesses

- The “15–30× compression” claim is not consistently tied to one M (some tables use 16; main headline says 8).
- The 3B threshold is plausible, but the causal story is underdeveloped (is it parameters, `d_model`, training recipe, tokenizer, or positional encoding?).
- Comparisons to Gist Tokens and other compressors should be more “apples-to-apples” on **wall-time speedup**: gist reports modest wall-time gains even with large compression. ([OpenReview][4])
- Prompt-tuning baseline variance is huge (±9.6 on TREC), suggesting either unstable training or too few eval examples.
- Cross-model prompt transfer prior work exists; “first continuous embeddings as wire protocol” reads overstated unless carefully scoped. ([arXiv][3])

#### 5. Detailed Critique

##### 5.1 Novelty

- PerceiverResampler specifically for **LLM→LLM** transfer is a nice adaptation of Flamingo/BLIP-style resamplers, and the “inject into receiver embeddings” path is distinct from prefix-tuning-at-all-layers. ([arXiv][3])
- But the novelty must be positioned as “cross-family, constant-size runtime representation transfer,” not “we invented soft prompts.”

##### 5.2 Experimental Evaluation

- Classification results are strong, but because you beat prompt-tuning on AG/TREC, reviewers will demand **full test sets** and more seeds.
- Add reasoning tasks where compression typically hurts (GSM8K/MATH) to match the broader prompt-compression literature (LLMLingua reports reasoning retention). ([arXiv][5])

##### 5.3 Ablations

- Great to show M scaling and source-layer sensitivity.
- Missing: ablate “bridge parameterization”: full-dim (537M) vs bottleneck (≈5M) on classification.

##### 5.4 Technical Soundness

- The “calibration mismatch” point is important and believable (embedding amplitude mismatch).
- However, the training curriculum reference is brief; prompt people will want exact schedules and loss weights.

##### 5.5 Writing

- Ensure consistent notation: M, dz, dmodel, λ; keep a single “default config” table and point every result to it.

#### 6. Questions

1. What exact **prompt-tuning method** (Lester-style, P-tuning v2, prefix tuning) and how many prompt tokens?
2. Is the 3B threshold robust across **architectures** (MoE, different positional encodings)?
3. Can a bridge trained on AG News transfer to TREC zero-shot (task generalization), or is it always per-task?

#### 7. Missing References

- Stronger engagement with transferability literature (beyond Su et al.) and newer “prompt modules” ecosystems. ([arXiv][3])

#### 8. Suggestions

- Add a “prompt-compression comparison” table: Gist, ICAE, LLMLingua, 500xCompressor vs LatentWire—explicitly stating what’s **not comparable** (single-model vs hetero). ([arXiv][6])

---

### Reviewer 3: Multi-Agent Systems Researcher

#### 1. Overall Assessment

- **Score:** 5/10
- **Confidence:** 3/5
- **Recommendation:** Borderline

#### 2. Summary

LatentWire proposes continuous embeddings as a communication channel between heterogeneous LLM agents, replacing text relay with constant-size soft tokens to reduce overhead and enable batching. The evidence is primarily classification + some QA collaboration experiments, with strong latency gains reported vs Text-Relay.

#### 3. Strengths

- Directly targets a real multi-agent pain point: repeated retokenization + prefill.
- Shows bidirectional transfer exists in principle.
- Demonstrates a “two-model collaboration” (joint rescoring) improvement above single models.
- Identifies and documents failure modes (binary task reg, incompatible model pairs).
- The constant-size protocol is conceptually attractive for many-agent systems.

#### 4. Weaknesses

- No evaluation on _agentic_ benchmarks (tool use, planning, multi-turn coordination), so the “multi-agent systems” claim is mostly hypothetical.
- Per-task training requirement undermines multi-agent practicality (you’d want a general protocol).
- Model-pair compatibility failures are severe (Qwen→Mistral random chance), limiting “heterogeneous” promise.
- Asymmetry is large (AG News reverse transfer collapses), raising coordination concerns.
- Latency evaluation is not embedded into a realistic agent loop (no “N turns, K agents” end-to-end measurement).

#### 5. Detailed Critique

##### 5.1 Novelty

- Compared to LatentMAS (shared KV memory) the hetero aspect is novel, but LatentMAS is “training-free” and shown on multi-agent settings; you need a comparable agent setting. ([arXiv][7])
- C2C is also hetero but operates at KV-cache fusion; your contribution is different but the _agent impact_ is not demonstrated. ([arXiv][1])

##### 5.2 Experimental Evaluation

- Add an actual multi-agent benchmark (e.g., AutoGen-style multi-turn tool use) and measure total tokens, wall time, success rate. (You already cite agent frameworks—use them.)
- Current tasks (classification) are too far from “communication under uncertainty.”

##### 5.3 Ablations

- Need “N-turn conversation” experiments: does error accumulate? do soft tokens drift? can you refresh or re-anchor?

##### 5.4 Technical Soundness

- Sound as a representation transfer mechanism, but agent settings require robustness to distribution shift.

##### 5.5 Writing

- Replace agent-motivating story with a concrete “agent loop diagram” and explicit metrics.

#### 6. Questions

1. Can you show a 3–5 turn multi-agent interaction where LatentWire reduces cost while preserving task success?
2. What is the training cost per _model pair_ per _task_, and can the same bridge be reused across tasks?
3. How do you detect/handle incompatible pairs automatically?

#### 7. Missing References

- More multi-agent communication protocol work (beyond the frameworks) and any “semantic channel coding” analogies if you want that framing.

#### 8. Suggestions

- Add an “agent benchmark” section; without it, many MAS reviewers will vote reject even if the core idea is good.

---

### Reviewer 4: Representation Learning Theorist

#### 1. Overall Assessment

- **Score:** 6/10
- **Confidence:** 3/5
- **Recommendation:** Weak Accept

#### 2. Summary

LatentWire learns a bottleneck representation (M×dz soft tokens) from one model’s hidden states that can condition a different model, implying partial alignment between representation spaces. It also claims a “phase change” around 3B parameters for decoding continuous prompts into text.

#### 3. Strengths

- Empirical evidence of **cross-model latent compatibility** (Llama→Mistral strong; some bidirectionality).
- Identifies **capacity threshold** as a phenomenon worth studying.
- Good that you test model pair compatibility and don’t assume universality.
- Attempts a simple theoretical inequality tying dmodel, M, dz.
- Discusses calibration and amplitude mismatch (important for geometry).

#### 4. Weaknesses

- The theoretical discussion is thin: the inequality is not derived, and α is hand-wavy.
- “Phase change” language is strong given only a few checkpoints evaluated.
- Asymmetry (AG News reverse collapse) is not deeply analyzed; this is the most interesting representational result.
- Failure of Qwen→Mistral suggests alignment is not “Platonic” in practice; you need a clearer stance.
- Using only 200 samples makes it hard to separate “representation mismatch” from evaluation noise.

#### 5. Detailed Critique

##### 5.1 Novelty

- Representation-level transfer is not new broadly, but _runtime_ latent transfer between frozen LLMs with a learned resampler is a meaningful empirical contribution.

##### 5.2 Experimental Evaluation

- Add analyses: CKA/Procrustes similarity across layers for pairs that work vs fail; measure how similarity predicts Table 20 outcomes.

##### 5.3 Ablations

- Probe sensitivity to dz and whether increasing dz helps Qwen→Mistral.

##### 5.4 Technical Soundness

- Plausible; but prove the adapter isn’t just learning to “hallucinate” labels from biases—use counterfactual tests.

##### 5.5 Writing

- When invoking representation convergence ideas, be careful not to oversell; state as hypothesis.

#### 6. Questions

1. Do successful pairs show higher cross-layer similarity (CKA) than failing pairs?
2. For asymmetry, is it receiver decoding limitations or sender representation granularity?
3. How stable are soft tokens across paraphrases / augmentations?

#### 7. Missing References

- More empirical representation alignment work (e.g., cross-model steering transfer you listed) if not already in bib. ([arXiv][8])

#### 8. Suggestions

- Add a “representation diagnostics” section with 2–3 quantitative metrics explaining success/failure and asymmetry.

---

### Reviewer 5: Skeptical Senior Reviewer

#### 1. Overall Assessment

- **Score:** 4/10
- **Confidence:** 4/5
- **Recommendation:** Weak Reject

#### 2. Summary

The paper proposes learning a PerceiverResampler that compresses one model’s hidden states into a small set of soft tokens consumable by another model, aiming to replace text-based relay in multi-LLM systems. It reports strong classification accuracies and large speedups vs a Text-Relay baseline, plus a model-size threshold.

#### 3. Strengths

- Real problem: repeated prefill and tokenization overhead.
- Good ablation culture.
- Honest reporting of failures (Qwen→Mistral).
- Provides measurement protocol notes (better than many papers).
- Training cost surprisingly low in GPU-hours.

#### 4. Weaknesses

- The evaluation is **not convincing**: 200 samples per dataset is not acceptable to claim SOTA-ish improvements.
- Novelty is precarious: looks like “prompt transfer with a resampler,” adjacent to prior cross-model prompt transfer and module methods. ([arXiv][3])
- The 27× speedup is against a strawman-y Text-Relay that includes an expensive summarization generation step; compare to stronger systems baselines.
- Bridge is 537M params; calling it “small adapter” is misleading; also comparing to full fine-tuning via parameter ratio is not the right metric.
- The “3B threshold” is asserted as fundamental, but only 5 checkpoints are shown, with no error bars or architecture controls.

#### 5. Detailed Critique

##### 5.1 Novelty

- C2C and LatentMAS already demonstrate latent-ish inter-model communication; your embedding-space channel differs, but “separate contribution” must be proven via unique capabilities (e.g., batchability, constant-size invariants, lower engineering complexity). ([arXiv][1])

##### 5.2 Experimental Evaluation

- Replace 200-sample eval with full test sets and more seeds; report significance.
- Add stronger baselines: optimized text relay with caching, shorter summaries, or distillation.

##### 5.3 Ablations

- You need ablations that demonstrate the bridge is not simply learning dataset priors.

##### 5.4 Technical Soundness

- Likely works, but the paper must prove it generalizes beyond the three classification tasks.

##### 5.5 Writing

- Several marketing phrases (“telepathically,” “first wire protocol”) will irritate skeptical reviewers.

#### 6. Questions

1. Why only 200 samples? Where is the full-test evaluation?
2. How strong is an optimized Text-Relay baseline with strict summary-length limits?
3. Can you demonstrate value on tasks where sender knowledge matters (reasoning/tool use)?

#### 7. Missing References

- More explicit mapping to “transferability of prompt tuning” and module-based transfer comparisons. ([arXiv][3])

#### 8. Suggestions

- The fastest way to flip me: (a) full test sets + more seeds, (b) one real multi-agent benchmark, (c) stronger baselines, (d) precise novelty claims.

---

### Reviewer 6: Reproducibility & Methodology Expert

#### 1. Overall Assessment

- **Score:** 5/10
- **Confidence:** 5/5
- **Recommendation:** Borderline

#### 2. Summary

LatentWire learns a bridge between frozen LLMs by compressing sender hidden states into learned soft tokens that can be injected into a receiver, with reported latency/accuracy results and extensive ablations. The paper provides some hyperparameter tables and training cost estimates, but key methodological choices leave reproducibility risks.

#### 3. Strengths

- Many explicit configuration details (models, M, steps, seeds).
- Latency measurement methodology is described (H100, preloaded, runs, SD).
- Multi-seed reporting for main classification results.
- Hyperparameter search table in appendix.
- Failure modes and fixes are documented (SST-2).

#### 4. Weaknesses

- “Evaluation: 200 samples per dataset” undermines reproducibility and comparability to prior work that uses full test sets.
- Potential internal inconsistency: abstract says mean across 3 seeds, but setup mentions seed 42; clarify which experiments are multi-seed.
- Text-Relay baseline is underspecified (summary prompt, max tokens, decoding, truncation rules).
- Many ablations are single-seed; that’s fine, but you should label clearly and avoid mixing with multi-seed claims.
- No code/data release statement in the excerpted text I saw; MLSys expects strong artifacts.

#### 5. Detailed Critique

##### 5.1 Novelty

- Not my focus; but novelty claims depend on replicable evidence, which currently is too fragile due to small evaluation.

##### 5.2 Experimental Evaluation

- Run full test sets and report confidence intervals; for latency, report p50/p95 and throughput under multiple batch sizes and concurrent requests.

##### 5.3 Ablations

- Good coverage, but create a single “Ablation Protocol” with shared seeds and sample sizes.

##### 5.4 Technical Soundness

- Training description mentions curriculum/annealing (page 10 excerpt) but the actual schedule should be given as pseudocode.

##### 5.5 Writing

- Add a “Reproducibility Checklist” section: hardware, software versions, exact prompts, decoding params, dataset splits, random seeds.

#### 6. Questions

1. Provide exact Text-Relay prompts and decoding settings.
2. Why use 200 samples—compute constraint or design?
3. Do the reported ±std numbers include test-set sampling randomness or only seed randomness?

#### 7. Missing References

- Not references, but missing artifact pointers (repo, scripts) are a problem.

#### 8. Suggestions

- Provide a minimal open-source artifact: one script reproducing Table 13 + Table 14 end-to-end, including Text-Relay.

---

### Reviewer 7: NLP Applications Researcher

#### 1. Overall Assessment

- **Score:** 6/10
- **Confidence:** 4/5
- **Recommendation:** Weak Accept

#### 2. Summary

The paper shows that a small fixed number of continuous vectors can carry enough task-relevant information from one LLM to another to perform classification with high accuracy, sometimes exceeding prompt tuning. It argues this enables efficient cross-model “communication” without tokenization mismatch.

#### 3. Strengths

- Strong results on three standard-ish text classification datasets with multi-seed reporting.
- Clear baselines: random chance, zero-shot, 5-shot, prompt tuning.
- Insightful analysis about why binary classification needs different regularization.
- Shows that “Text-Relay summarization” can lose class signals (qualitative comment).
- Includes TREC-6 correctly (6 classes).

#### 4. Weaknesses

- Benchmarks are mostly **easy**; success here may not imply success on reasoning or compositional tasks.
- TREC/AG News results show high variance; 200-sample evaluation exacerbates this.
- No qualitative error analysis for misclassifications, especially for reverse transfer where AG News collapses.
- Prompt templates can dominate zero-shot baselines; ensure prompt engineering parity.
- Text-Relay baseline might be unfair if summarization prompt is not tuned to preserve label-relevant info.

#### 5. Detailed Critique

- Add tasks like **Natural Language Inference** variants, long-document classification, and more “instruction following with constraints.”
- If you keep classification focus, evaluate on full test sets and show calibration curves / confusion matrices.

#### 6. Questions

1. How does performance change on full test sets?
2. Can you show confusion matrices for AG News forward vs reverse?
3. For Text-Relay, what summary prompt is used and what length?

#### 7. Missing References

- More benchmark-focused prompt compression papers that measure end-task accuracy under compression (some are in your related list already).

#### 8. Suggestions

- Add one stronger “NLP task” beyond classification—e.g., QA with long contexts and exact-match constraints—using full sets.

---

### Reviewer 8: Efficiency / Compression Specialist

#### 1. Overall Assessment

- **Score:** 6/10
- **Confidence:** 3/5
- **Recommendation:** Weak Accept

#### 2. Summary

LatentWire compresses 300–500 tokens into a fixed number of learned soft vectors and reports large wall-time gains vs text relay. It argues constant-size latents enable batching and scale better with larger models, projecting further gains at 70B+.

#### 3. Strengths

- End-to-end latency improvements are large and measured with variance notes.
- Highlights why “compression does not equal speedup” for small models (overhead dominates).
- Shows batching advantage is central (uniform M vs variable text).
- Provides training compute costs and argues it’s practical.
- Positions against prompt compressors like Gist/ICAE as orthogonal due to hetero mismatch.

#### 4. Weaknesses

- Comparisons to Gist/ICAE/LLMLingua need a careful statement: those methods report modest wall-time speedups despite compression (gist). ([OpenReview][4])
- Your 27× speedup is vs Text-Relay that includes autoregressive summarization; that is closer to “agent summarization overhead” than to typical compressor baselines.
- The bridge uses a large parameterization for classification (537M); from a compression standpoint, the “compressor” cost matters.
- No analysis of memory bandwidth / KV-cache footprint at serving time (you claim it conceptually).
- The “projected 5–10× for 70B” is speculative; needs either a simulation or roofline-style argument.

#### 5. Detailed Critique

- Provide a cost model: total FLOPs for Bridge vs Text-Relay decomposed into (sender prefill, bridge attention, receiver prefill, decoding).
- Evaluate memory and throughput under realistic server settings; show where the 27× comes from and whether it persists with optimized relay.

#### 6. Questions

1. What is the end-to-end compute breakdown for Text-Relay vs Bridge?
2. How does bridge overhead scale with M and dz in practice?
3. Can you show a stronger 70B+ extrapolation argument?

#### 7. Missing References

- Include LLMLingua results discussion on reasoning retention and speedups for context. ([arXiv][5])

#### 8. Suggestions

- Add a “compressed-token caching” experiment: reuse soft tokens across multiple queries; show amortized gains.

---

### Reviewer 9: Writing & Presentation Critic

#### 1. Overall Assessment

- **Score:** 6/10
- **Confidence:** 5/5
- **Recommendation:** Weak Accept

#### 2. Summary

The paper is clearly motivated and the main idea is simple: replace cross-model text serialization with shared learned continuous vectors. The results sections contain useful tables and ablations, but the narrative sometimes overclaims and the terminology/claims need tightening for a top venue.

#### 3. Strengths

- Strong “hook” and clear motivation early (tokenizer mismatch, repeated prefill).
- Abstract is dense but informative and includes metrics.
- Tables are helpful and many key numbers are not buried (Table 13/14).
- You explicitly flag limitations and failure modes (rare, good).
- Related work section does attempt to differentiate from C2C/Cross-LoRA etc.

#### 4. Weaknesses

- Tone occasionally drifts into marketing (“telepathically,” “first wire protocol”).
- Claim precision issues: “15–30× compression” vs “8 tokens” vs tables showing 16 latents (readers will perceive inconsistency).
- The baseline “Text-Relay” needs a diagram—right now it’s described verbally and is easy to misinterpret.
- Some key methods (curriculum/annealing) are referenced but not cleanly summarized with pseudocode.
- Notation/term consistency: “Bridge,” “LatentWire,” “soft tokens,” “learned queries,” “dz” vs “full model dimension” needs one canonical set.

#### 5. Detailed Critique (incl. sentence-level edits)

##### 5.5 Sentence-level issues & concrete rewrites

- **“What if models could communicate telepathically…”** (Intro)

  - Issue: cute but risky for MLSys tone.
  - Suggest: “What if models could communicate _directly in representation space_, without serializing to text?”

- **“Our approach is the first to establish continuous embeddings as a wire protocol…”**

  - Issue: “first” claim is fragile given prompt-transfer and module literature.
  - Suggest: “We demonstrate a practical design for using continuous embeddings as a wire protocol _across heterogeneous LLM families_.”

- **Capacity claim language** (“phase change,” “cannot… regardless”)

  - Issue: too absolute without broader coverage.
  - Suggest: “In our experiments, models below 3B _failed to_ decode soft prompts into coherent text under our training setup…”

##### Figures & tables

- Add a single “System Overview” figure: sender → resampler → receiver; and a parallel “Text-Relay” pipeline figure.

#### 6. Questions

1. Can you reconcile the M=8 vs M=16 presentation so the paper reads internally consistent?
2. Provide an explicit Text-Relay diagram and pseudo-config.
3. Where do you define all symbols (M, dz, λ) in one place?

#### 7. Missing References

- If you keep “wire protocol” framing, cite any related “semantic communication” / representation-channel work (even outside LLMs), or explicitly say you’re borrowing the term.

#### 8. Suggestions

- Make one pass to remove absolute “first/always/cannot” claims unless strictly proven; these trigger adversarial reviewers.

---

### Reviewer 10: Industry Practitioner

#### 1. Overall Assessment

- **Score:** 5/10
- **Confidence:** 3/5
- **Recommendation:** Borderline

#### 2. Summary

LatentWire proposes training a bridge (PerceiverResampler + adapters) that allows two frozen LLMs to exchange a constant-size vector representation, giving major latency improvements over text relay. It appears inexpensive to train per task on a single H100, but it is not yet a drop-in production solution due to compatibility and generalization limitations.

#### 3. Strengths

- Real deployment pain point: repeated context passing between models.
- Per-task training cost is low enough for experimentation.
- Latency gains are huge vs Text-Relay and measured on relevant hardware (H100).
- Shows batching advantage (very relevant in serving).
- Explicitly calls out model compatibility and direction asymmetry.

#### 4. Weaknesses

- Requires **per-task training** and (likely) per model-pair training: hard to productize at scale.
- Some model pairs fail completely; in production you can’t discover that late.
- Bridge size (537M) adds operational complexity (versioning, serving, memory, security review).
- No discussion of **safety/security**: can soft tokens exfiltrate sensitive info across trust boundaries? can they be logged/audited?
- Evaluation doesn’t include real constraints: multi-tenant, p99 latency, autoscaling, failures.

#### 5. Detailed Critique

- From a production perspective, the big missing piece is a “deployment blueprint”: how you package, cache, and monitor soft-token messages; how you rollback; how you prevent misuse.

#### 6. Questions

1. Can you build a “compatibility test” that predicts whether a model pair will work before training?
2. Can the same bridge generalize across tasks, or do we need hundreds of bridges?
3. What is the security posture—can soft tokens be inspected, filtered, or constrained?

#### 7. Missing References

- Production systems papers about model serving and batching (even if not directly about LLM communication) would strengthen MLSys positioning.

#### 8. Suggestions

- Add a small “production simulation”: 3 agents, 10 turns, realistic prompt sizes, measure cost and success; include p95 latency.

---

## Meta-Review

### Consensus Summary

- Reviewers broadly agree the **core idea is interesting and plausible**: a constant-size latent “message” between frozen heterogeneous LLMs using embedding injection can work and can batch well.
- The committee agrees the **27× speedup** is real _for the defined Text-Relay baseline_ and is supported with a basic methodology description.
- Everyone agrees the paper helpfully documents **limitations**: per-task training, direction asymmetry, and incompatible model pairs.

### Key Controversies

1. **Novelty vs related work:**

   - Some see embedding-space injection as a distinct channel vs KV-cache fusion (C2C) and KV-sharing (LatentMAS). ([arXiv][1])
   - Others think it risks being categorized as “prompt transfer + adapter,” close to transferability and module work. ([arXiv][3])

2. **Strength of experimental evidence:**

   - Biggest dispute is evaluation size: the “200 samples per dataset” design makes performance claims feel fragile.

3. **Comparability of speedups:**

   - 27× vs Text-Relay is impressive, but not directly comparable to speedups in other papers because baselines differ (Text-Relay includes autoregressive generation).

### Critical Issues (must address)

1. **Full-test-set evaluation + more statistical rigor.** Replace 200-sample eval with full test sets, more seeds, and confidence intervals.
2. **Baseline clarity and fairness.** Fully specify Text-Relay (prompt, max tokens, decoding, batching attempts) and add at least one stronger relay baseline.
3. **Novelty scope statements.** Remove/soften “first” claims or narrowly scope them to “cross-family constant-size embedding wire protocol.”
4. **Practicality story.** Address per-task/per-pair training and incompatibility more proactively with mitigation strategies.

### Venue Recommendation

- **MLSys fit:** Conditional. The systems motivation is strong, and the latency/batching story is relevant.
- However, if you don’t add production-style evaluation (full test sets, stronger baselines, serving metrics), reviewers may view it as more **NLP/representation** than systems.
- Alternatives (if MLSys pushback persists): **EMNLP/ACL Findings** (as a communication/prompting mechanism), or **NeurIPS/ICLR** (representation + multi-model interaction), depending on added agent benchmarks.

### Overall Verdict

**Borderline → Weak Accept** if you (1) fix evaluation rigor and (2) tighten baseline definitions and novelty claims. Without those, a realistic outcome is **Weak Reject** due to perceived fragility and comparability concerns.

### Revision Roadmap

#### Critical (must fix for any venue)

1. **Evaluate on full test sets** for SST-2 / AG News / TREC (not 200 samples), report mean±CI over ≥5 seeds.
2. **Specify and strengthen Text-Relay baseline** (prompt, summary length cap, decoding params, batching attempt, caching).
3. **Make “first/novelty” claims defensible** (scope precisely; compare cleanly vs C2C/LatentMAS/LLM Modules). ([arXiv][1])

#### Major (significantly strengthens)

1. Add at least one **agentic / multi-turn benchmark** measuring end-to-end runtime and success.
2. Deepen analysis of **asymmetry** and **pair compatibility** with representation diagnostics and mitigation ideas.
3. Provide a clearer tradeoff story for **bridge size** (537M vs small bottleneck variants) on classification.

#### Minor (nice to have)

1. Tone polish (“telepathic”), unify notation, reconcile M=8 vs M=16 messaging.
2. Add a one-page “repro checklist” and a diagram for Text-Relay vs Bridge.

---

If you want, I can also generate a **reviewer-response checklist** that maps each “Critical/Major/Minor” item to the exact new experiment/figure/table you’d add (and where in the paper it should go) so you can execute revisions efficiently.

[1]: https://arxiv.org/abs/2510.03215?utm_source=chatgpt.com 'Cache-to-Cache: Direct Semantic Communication Between Large Language Models'
[2]: https://www.arxiv.org/abs/2502.08213?utm_source=chatgpt.com 'Knowledge Transfer from a Large to a Small Model using ...'
[3]: https://arxiv.org/pdf/2111.06719?utm_source=chatgpt.com 'arXiv:2111.06719v2 [cs.CL] 27 Apr 2022'
[4]: https://openreview.net/forum?id=2DtxPCL3T5&utm_source=chatgpt.com 'Learning to Compress Prompts with Gist Tokens'
[5]: https://arxiv.org/abs/2310.05736?utm_source=chatgpt.com 'LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models'
[6]: https://arxiv.org/pdf/2307.06945?utm_source=chatgpt.com 'In-context Autoencoder for Context Compression in a ...'
[7]: https://arxiv.org/abs/2511.20639?utm_source=chatgpt.com 'Latent Collaboration in Multi-Agent Systems'
[8]: https://arxiv.org/abs/2511.03945?utm_source=chatgpt.com 'Direct Semantic Communication Between Large Language Models via Vector Translation'

# Kimi

# LatentWire Peer Review Report

## Individual Reviews

### Reviewer 1: MLSys Systems Expert

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 5/5
- **Recommendation**: Weak Accept

**Summary**
LatentWire proposes a PerceiverResampler-based system to enable cross-architecture LLM communication via continuous soft tokens injected through `inputs_embeds`, claiming 27× speedup over text-relay baselines on classification tasks.

**Strengths**

1. **Clear systems motivation**: The prefill bottleneck analysis in Section 1.1 correctly identifies the O(n²) scaling issue in multi-agent LLM communication.
2. **Comprehensive efficiency metrics**: Reports compression ratios, wall-clock time, payload bytes, and throughput scaling (Figure 1).
3. **Practical implementation details**: Provides training stability fixes (first-token objective, exposure bias, calibration) that are valuable for reproducibility.
4. **Batching advantage demonstration**: Figure 1 clearly shows Bridge achieves 109 samples/sec vs. Text-Relay's ~1 sample/sec due to serialization bottlenecks.
5. **Capacity threshold discovery**: Establishes a clear 3B parameter minimum for viable deployment, which is crucial for practitioners.

**Weaknesses**

1. **Speedup measurement methodology is questionable**: The 27× claim compares Bridge (38.3ms) to Text-Relay (1055ms), but Text-Relay uses sequential generation without KV-cache reuse. A fairer comparison would use parallel text generation with proper batching and cache sharing within each model.
2. **Inference hardware details missing**: No specification of GPU count, memory bandwidth utilization, or whether speedup holds across different hardware configurations (e.g., CPU inference, edge devices).
3. **No end-to-end multi-agent evaluation**: Claims speedup for "multi-turn conversation" but evaluates only single-turn classification. Real multi-agent systems would have complex interaction patterns not captured.
4. **Bridge size is substantial**: 537M parameters is 6.7% of an 8B model, not "minimal" as claimed. No analysis of inference memory overhead or quantization potential.
5. **No distributed systems evaluation**: MLSys expects evaluation of distributed scenarios (multi-GPU, multi-node). The paper assumes colocated models without communication overhead.

**Detailed Critique**

**5.1 Novelty Assessment**
The distinction from Cache-to-Cache (C2C) is meaningful—embedding-space vs. attention-space communication—but not revolutionary. C2C's KV-cache fusion handles decoder-to-decoder flow naturally, while LatentWire requires a separate encoder and careful calibration. The novelty is incremental: it's applying Flamingo's PerceiverResampler to a new domain.

**5.2 Experimental Evaluation**
Latency measurements lack statistical rigor. Table 14 reports "Average 38.3ms" but no std deviation across runs. Speedup claims are based on batch size comparisons that unfairly penalize text-relay. A proper systems evaluation should:

- Measure latency distribution (p50, p99, p99.9)
- Evaluate under load/concurrent requests
- Report memory bandwidth saturation
- Compare against optimized text-relay with streaming

**5.3 Missing Systems Ablations**

- **No quantization evaluation**: Can the 537M bridge be quantized to int8/int4? This is critical for production deployment.
- **No multi-GPU communication cost**: What's the PCIe/NVLink overhead when models are on different GPUs?
- **No failure case analysis**: What happens if the bridge mispredicts? No error recovery mechanisms discussed.

**5.4 Technical Soundness**
The distributed systems assumptions are naive. The paper states "both models receive the same latent prefix" but doesn't address:

- Synchronization overhead
- Version mismatches between models
- Dynamic model loading/unloading in production
- Fault tolerance if one model crashes

**6. Questions for Authors**

1. What is the end-to-end latency including bridge encoding time? Table 14 only shows "Bridge" timing, but encoding adds overhead.
2. How does speedup scale with model size? Projections in Section 7.3 are theoretical; no empirical data for 70B+ models.
3. Can the bridge run on a separate device (CPU) while models run on GPU? This would be a practical deployment scenario.
4. What's the memory footprint increase at inference time? 537M parameters at fp16 = 1.07GB additional VRAM per model pair.

**7. Missing References**

- **SpecInfer**: Xia et al., OSDI 2024 - for batched speculative decoding comparisons
- **S-LoRA**: Zheng et al., MLSys 2024 - for multi-adapter serving systems
- **vLLM**: Kwon et al., MLSys 2023 - the standard for LLM serving throughput evaluation

**8. Suggestions for Improvement**

1. Add microbenchmarks: breakdown of encoding vs. adapter vs. generation time
2. Evaluate on multi-turn dialogue datasets (e.g., MultiWOZ) to justify multi-agent claims
3. Report memory bandwidth usage and quantization results
4. Compare against a properly optimized text-relay baseline using vLLM's continuous batching
5. Add a fault tolerance analysis section

---

### Reviewer 2: Prompt Tuning / Soft Prompts Specialist

**Overall Assessment**

- **Score**: 7/10
- **Confidence**: 5/5
- **Recommendation**: Accept

**Summary**
LatentWire adapts PerceiverResampler for cross-model soft prompt transfer, demonstrating strong classification performance with 8 soft tokens between Llama and Mistral.

**Strengths**

1. **Clear problem formulation**: Section 3.1 mathematically defines the cross-model communication problem precisely.
2. **Comprehensive ablation of design choices**: Table 16 shows detailed ablations (adapter regularization, calibration, first-token objective) that are essential for reproducing results.
3. **Training stability innovations**: The first-token objective (Section 3.4.1) and mixed warm-up alignment (3.4.2) address real exposure bias issues in soft-prompt training.
4. **Capacity threshold analysis**: Section 4 provides a thorough characterization of why small models fail, including control experiments and theoretical analysis.
5. **Bidirectional transfer validation**: Table 18 shows the bridge works in both directions, establishing universal properties.

**Weaknesses**

1. **Underdeveloped comparison to Gist tokens**: Claims Gist tokens only work within single families but doesn't evaluate Gist tokens on Llama→Mistral as a baseline. A direct comparison would strengthen claims.
2. **Prompt-tuning baseline is weak**: Table 13 compares to prompt-tuning on Mistral only, but doesn't compare to prompt-tuning on Llama (the sender). The sender's prompt-tuning performance is the relevant ceiling.
3. **No comparison to ICAE/500xCompressor**: These are single-model but achieve higher compression (26-500×). Their techniques (memorizing, reconstruction losses) might improve LatentWire.
4. **Insufficient evaluation of M scaling**: Table 17 shows M=8 optimal for AG News, but why does TREC drop at M=4 then recover? Non-monotonic behavior suggests hyperparameter instability.
5. **"Bridge" vs "Prompt-tuning" terminology is confusing**: The paper uses "Bridge" for the full system but "prompt-tuning" for a single-model baseline, creating inconsistent terminology.

**Detailed Critique**

**5.1 Novelty Assessment**
The cross-model extension is novel but incremental. The core architectural contribution is applying PerceiverResampler (from Flamingo) to LLM-to-LLM transfer. The key insight is that continuous embeddings preserve manifold structure better than discrete tokens, which is well-known from VQ-VAE literature. The 3B threshold is the most novel empirical finding, but it's unclear if this is specific to cross-model transfer or just reflects Lester et al.'s prompt-tuning threshold.

**5.2 Experimental Evaluation**
**Missing baselines:**

- **Gist tokens cross-model**: Train Gist tokens on Llama and evaluate on Mistral. Even if it fails, this directly justifies the need for PerceiverResampler.
- **Linear probe + generation**: Train a linear classifier on sender embeddings, then use that as initialization for soft tokens.
- **Full fine-tuning of receiver**: What if Mistral is fine-tuned on the tasks directly? This establishes the task ceiling.

**Benchmark concerns**: SST-2, AG News, and TREC are simple classification tasks. The paper claims general multi-agent communication, but doesn't evaluate on:

- Generation tasks (summarization, translation)
- Multi-hop reasoning (HotpotQA is mentioned but not in main tables)
- Code generation (different modality)
- Long-context tasks (beyond 500 tokens)

**5.3 Ablation Studies**
**Incomplete ablations:**

- **Source layer sweep is insufficient**: Table 19 shows only 6 layers. Why not test every 4th layer from 0-31?
- **No diversity loss schedule ablation**: Diversity loss is disabled for SST-2, but what about a curriculum that gradually reduces it?
- **No calibration method comparison**: Only tests "with/without" but not different calibration methods (z-normalization, percentile matching).
- **No adapter architecture comparison**: Only tests tanh+LayerNorm vs not. What about residual adapters, hypernetworks, or low-rank adaptations?

**5.4 Technical Soundness**
**Training procedure concerns**: The curriculum learning schedule (Section 3.4.2) is described qualitatively but not specified algorithmically. What triggers transitions between phases? Is it step-based, loss-based, or manual?

**Exposure bias analysis**: The first-token objective is motivated by exposure bias, but this is odd—soft prompts shouldn't have autoregressive teacher-forcing issues since they're static. The failure mode might be signal collapse rather than exposure bias. The control experiment in Table 10 ("Zero-prefix success") suggests the issue is decoding capability, not training dynamics.

**6. Questions for Authors**

1. Why not compare directly to Gist tokens? Gist tokens use a learned compressor within a single model; why can't you train on Llama and evaluate the compressor on Mistral?
2. What is the prompt-tuning performance on the _sender_ (Llama)? This is the true upper bound for the bridge.
3. Can you explain the non-monotonic behavior in Table 17 (TREC performance at M=4)?
4. How much of the 3B threshold is due to prompt tuning generally failing below 3B vs. cross-model transfer specifically?
5. Why does diversity loss harm binary classification but help multi-class? Is this a known phenomenon?

**7. Missing References**

- **Prefix-tuning for cross-task transfer**: Su et al. (2022) is cited but not thoroughly compared. Their cross-model prompt transfer within RoBERTa family shows different failure modes.
- **Memorizing Transformers**: Wu et al., 2022 - for comparison to external memory approaches
- **Hypernetworks for soft prompts**: Razdaibiedina et al., 2023 - could replace PerceiverResampler
- **Scaling laws for soft prompts**: Liu et al., 2024 - for predicting performance based on M and model size

**8. Suggestions for Improvement**

1. Add Gist token baseline: train gist compressor on Llama, evaluate on Mistral via adapter
2. Include sender prompt-tuning baseline to establish true ceiling
3. Add generation task evaluation (e.g., XSUM, CNN/DM) to demonstrate generalization
4. Provide algorithmic description of curriculum schedule in pseudocode
5. Test more model pairs (Llama→Gemini, GPT→Claude via API if possible)
6. Analyze why TREC shows non-monotonic M scaling
7. Add diversity loss visualization to show token orthogonality effects

---

### Reviewer 3: Multi-Agent Systems Researcher

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 4/5
- **Recommendation**: Weak Reject

**Summary**
LatentWire proposes a technical solution for cross-model LLM communication but lacks evaluation in realistic multi-agent scenarios, focusing instead on simple classification tasks.

**Strengths**

1. **Identifies real multi-agent bottleneck**: The prefill cost analysis is accurate for current systems.
2. **Provides concrete API**: The `inputs_embeds` injection approach is actionable for framework builders.
3. **Demonstrates bidirectional transfer**: Important for heterogeneous agent teams where any model might be the "sender."
4. **Training stability contributions**: The first-token objective and calibration fixes are valuable for agent system robustness.
5. **Constant-size communication**: Theoretical analysis in Section 7 shows O(M²) vs O(n²) scaling, which is compelling for long conversations.

**Weaknesses**

1. **No actual multi-agent evaluation**: Claims relevance to "multi-turn conversation between Llama and Qwen" (Section 1) but never evaluates dialogue tasks or agent coordination.
2. **Simplified turn-taking**: Classification is single-turn. Real agents have dynamic turn allocation, partial observability, and credit assignment.
3. **Lacks integration with agent frameworks**: No demonstration with AutoGen, LangChain, or CAMEL mentioned in Section 2.2.
4. **State management ignored**: Agents need to maintain conversation history, tool use traces, and intermediate reasoning. The paper doesn't show how soft tokens handle these.
5. **No failure recovery**: If the bridge produces corrupted tokens, there's no fallback mechanism (e.g., reverting to text).

**Detailed Critique**

**5.1 Novelty Assessment**
The "interlingua" framing is oversold. This is essentially a learned compression codec, not a general communication protocol. True multi-agent interlinguas (e.g., DroidSpeak, Anonymous 2024b) include:

- Protocol negotiation
- Error correction
- Semantic versioning
- Capability advertisement

LatentWire has none of these. It's a unidirectional compression pipeline, not a bidirectional communication protocol.

**5.2 Experimental Evaluation**
**Missing multi-agent benchmarks**:

- **Tool use**: Can soft tokens represent function calls and results?
- **Multi-turn dialogue**: Evaluate on MultiWOZ or Taskmaster
- **Debate/consensus**: Two agents discussing to reach agreement
- **Adversarial interaction**: Agent A tries to persuade Agent B

**Task mismatch**: SST-2 and AG News are static classification. Real agents perform dynamic reasoning with partial observability. The paper's evaluation doesn't match the problem statement.

**5.3 Ablation Studies**
Missing critical ablations for multi-agent scenarios:

- ** Partial conversation history**: What if only the last K turns are encoded?
- **Mixed modalities**: Can it handle text + images (if sender is vision-language)?
- **Agent identity encoding**: How does the bridge distinguish between "Agent A says" vs. "Agent B says"?

**5.4 Technical Soundness**
The `inputs_embeds` interface assumption is fragile. Changes to model implementations (e.g., RoPE, ALiBi modifications) could break compatibility. The paper doesn't discuss:

- **Version pinning**: Which exact model checkpoints are compatible?
- **Embedding space evolution**: If Llama-3.1 is updated, does the bridge need retraining?
- **Security**: Can malicious soft tokens cause adversarial behavior?

**6. Questions for Authors**

1. How would you integrate this into AutoGen or LangChain? Show pseudocode.
2. What happens if Agent A and Agent B disagree? Can soft tokens represent uncertainty or confidence?
3. How do you handle conversation history beyond what fits in the encoder's context window?
4. Can the bridge encode agent capabilities (e.g., "I can do math, I cannot browse")?
5. What's the cold-start problem? How much data is needed per new model pair?

**7. Missing References**

- **Agent communication languages**: FIPA ACL, KQML - for formal protocol comparison
- **DroidSpeak**: Anonymous 2024b - mentioned but not compared
- **Generative Agents**: Park et al., 2023 - for realistic multi-agent evaluation
- **ToolLLM**: Qin et al., 2023 - for tool-using agent benchmarks

**8. Suggestions for Improvement**

1. Add a multi-agent evaluation section: implement a two-agent dialogue system on MultiWOZ
2. Provide integration guide for existing frameworks (LangChain, AutoGen)
3. Evaluate stateful communication: encode conversation history of varying lengths
4. Add security analysis: can adversarial soft tokens cause harmful outputs?
5. Discuss fallback mechanisms: when to revert to text communication
6. Add protocol versioning discussion for model updates
7. Include capability advertisement mechanism

---

### Reviewer 4: Representation Learning Theorist

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 5/5
- **Recommendation**: Weak Accept

**Summary**
LatentWire explores cross-model representation alignment via PerceiverResampler, empirically establishing a capacity threshold for soft-prompt decoding.

**Strengths**

1. **Connects to Platonic Representation Hypothesis**: Section 2.5 correctly cites Huh et al. (2024) to motivate why cross-family transfer should work.
2. **Theoretical capacity analysis**: Section 4.4 provides a plausible inequality relating model dimension to latent capacity.
3. **Control experiments**: The zero-gain control (Section 4.3) cleanly isolates soft-prompt decoding failure from general generation failure.
4. **Embedding space visualization**: Figure 2 shows meaningful clusters, providing evidence that the bridge learns semantically structured representations.
5. **Rigorous failure mode analysis**: The progression from amplitude overwhelm → calibrated chaos → zero-prefix success is methodologically sound.

**Weaknesses**

1. **Theory is underdeveloped**: The capacity inequality (d_model ≥ α·M·d_z) is heuristic, not derived from first principles.
2. **No analysis of representation geometry**: No probing experiments to verify what information is preserved vs. discarded.
3. **Calibration is ad hoc**: RMS matching is justified empirically but lacks theoretical grounding. Why RMS? Why not mean/variance matching?
4. **No connection to relative representations**: Moschella et al. (2023) showed anchor transformations enable zero-shot stitching. Why not use this instead of learned adapters?
5. **Asymmetric transfer unexplained**: Table 18 shows significant directional bias (Llama→Mistral: 90.3% AG News vs. Mistral→Llama: 63.5%). This contradicts "universal interlingua" claims.

**Detailed Critique**

**5.1 Novelty Assessment**
The paper's core novelty is empirical: establishing that cross-model soft-prompt transfer works at scale. The architecture (PerceiverResampler + adapters) is standard. The theoretical contribution is weak—no new theorems, no convergence analysis, no generalization bounds.

The 3B threshold is interesting but confounded by two factors:

1. Prompt tuning generally fails below ~1B (Lester et al., 2021)
2. Cross-model transfer adds complexity

The paper doesn't disentangle these. A proper analysis would compare:

- Single-model prompt tuning at each scale
- Cross-model transfer at each scale
- The gap between them

**5.2 Experimental Evaluation**
**Missing representation analysis**:

- **Probing**: What linguistic properties (syntax, semantics, pragmatics) are preserved in soft tokens? Use probe classifiers.
- **Nearest neighbors**: For a given soft token, what are its nearest text tokens in the receiver's embedding space?
- **Interpolation**: Linearly interpolate between two soft tokens. Does generation smoothly interpolate semantically?
- **Dimensionality analysis**: How many dimensions of the 4096-dim latent actually carry information?

**Statistical concerns**: The paper uses fixed random seed 42 for all experiments. This is bad practice. Should average over multiple seeds for training dynamics, not just final evaluation.

**5.3 Ablation Studies**
**Missing theoretical ablations**:

- **No source layer analysis**: Table 19 is incomplete. The optimal layer should vary by task complexity. No explanation why layer 16 is generally good.
- **No adapter capacity scaling**: How does F1 change with adapter size? Is 537M parameters necessary or can smaller adapters suffice?
- **No orthogonality analysis**: What is the cosine similarity between learned soft tokens? Does diversity loss actually help?

**5.4 Technical Soundness**
**Mathematical notation issues**:

- Equation (1) uses n_i tokens but doesn't define n_i per model
- The capacity inequality uses α ≈ 0.5-1.0 "based on empirical observations"—this is vague
- The "working memory" calculation in Section 4.4 is incorrect: attention heads have dimension d_model/n_heads, not d_model

**Theoretical contradictions**: The paper claims continuous embeddings preserve "exact geometric relationships" (Section 3.2.1) but also calls it "lossy compression" (Section 8.4). This is contradictory. Clarify: Is it lossless for semantic structure but lossy for surface form?

**6. Questions for Authors**

1. Can you derive the capacity inequality from information-theoretic principles? What is the role of α?
2. Why does RMS calibration work better than mean-centering or whitening?
3. Why is transfer asymmetric (Table 18)? What geometric property of Llama's space makes it better for decoding?
4. What is the rank of the soft token representation matrix? Is it full rank or low-rank?
5. How do you know the bridge isn't just memorizing training data? Report validation vs. test generalization gap.

**7. Missing References**

- **Linear mode connectivity**: Entezari et al., 2022 - for analyzing adapter space geometry
- **Task arithmetic**: Ilharco et al., 2023 - for composing soft prompts
- **Representation engineering**: Zou et al., 2023 - for probing latent representations
- **Diffusion models as priors**: Issenhuth et al., 2024 - for alternative to deterministic compression

**8. Suggestions for Improvement**

1. Formalize the capacity inequality: relate to model's effective rank or VC dimension
2. Add representation probing experiments to verify what is preserved
3. Analyze the geometry of adapter mappings: are they approximately linear?
4. Explain asymmetric transfer theoretically (e.g., is Mistral's embedding space more isotropic?)
5. Add representation interpolation experiments
6. Compare to relative representations (Moschella et al.) which is training-free
7. Report effective dimensionality of soft tokens (e.g., via participation ratio)
8. Derive calibration method from maximum likelihood principle

---

### Reviewer 5: Skeptical Senior Reviewer

**Overall Assessment**

- **Score**: 4/10
- **Confidence**: 5/5
- **Recommendation**: Reject

**Summary**
The paper presents a well-engineered system for cross-model soft-prompt transfer but makes overblown claims about novelty and practical impact, with insufficient experimental rigor to support major assertions.

**Strengths**

1. **Thorough failure analysis**: Section 4's characterization of small model failures is methodologically exemplary.
2. **Training stability contributions**: The curriculum learning and first-token objective address real problems.
3. **Complete ablation study**: Table 16 is comprehensive and will be useful to future researchers.
4. **Honest limitations section**: Section 8 is unusually candid about failures (Qwen→Mistral, binary classification issues).
5. **Strong empirical validation of thresholds**: The 3B cutoff is well-supported across multiple model scales.

**Weaknesses**

1. **Core novelty is questionable**: The architecture is PerceiverResampler + adapters, a straightforward combination of existing components. No new algorithms.
2. **Overclaims on speedup**: 27× speedup is vs. a naive text-relay baseline. Properly optimized systems would be much closer.
3. **Benchmark selection is weak**: Three simple classification tasks are insufficient for MLSys. No generation, reasoning, or long-context evaluation.
4. **Statistical power is low**: 3 seeds, 200 samples per dataset. For classification with ~90% accuracy, this gives ±4% margin of error at 95% CI—too wide to claim superiority.
5. **No ablation of training cost**: 537M parameters is substantial. No comparison to training smaller bridges or using LoRA-style parameter sharing.

**Detailed Critique**

**5.1 Novelty Assessment**
**Not new**:

- **PerceiverResampler for compression**: Flamingo (Alayrac et al., 2022) did this for vision-language.
- **Cross-model adapters**: AdapterFusion (Pfeiffer et al., 2021) combined multiple model representations.
- **Soft prompt injection**: `inputs_embeds` is standard in HuggingFace.

**What's barely new**: Applying these to LLM-to-LLM communication. But the multi-agent motivation is superficial—the paper could be rewritten as "Cross-Model Prompt Compression" without losing technical content.

**Comparison to concurrent work**:

- **C2C**: Also cross-family, also uses learned components. LatentWire's only distinction is embedding vs. KV-cache injection.
- **LatentMAS**: Training-free, which is a bigger contribution. LatentWire's learned approach needs justification: why not use training-free KV-cache sharing?

**5.2 Experimental Evaluation**
**Statistical concerns**:

- **Sample size**: 200 samples per dataset is tiny. AG News has 7,600 test samples; using 200 introduces selection bias.
- **Seed variance**: Table 13 shows std dev ±5% on some results. With 3 seeds, this is unreliable. Need ≥10 seeds.
- **Multiple comparisons**: Table 16 has 12 configurations but no correction for multiple testing.

**Baseline problems**:

- **Prompt-tuning baseline**: Trained on receiver only. Should train on sender+receiver jointly to see if bridge adds value beyond independent tuning.
- **Text-relay baseline**: Unfair comparison. Uses greedy decoding (implicitly serialized) while Bridge uses parallel encoding. Should compare to batched text generation.

**Missing experiments**:

- **Cross-task generalization**: Train on AG News, test on TREC (same input format). Does bridge learn task-agnostic representations?
- **Longer contexts**: Test with 4K+ token inputs to validate compression benefits scale.
- **Out-of-distribution**: Test on adversarial examples or domain-shifted data.

**5.3 Ablation Studies**
**Incomplete**:

- **No training data scaling**: How does performance vary with 100, 1K, 10K training samples?
- **No convergence analysis**: How many steps are needed for loss vs. F1 correlation?
- **No hyperparameter robustness**: Table 22 shows optimal values but not sensitivity curves.

**5.4 Technical Soundness**
**Potential bugs**:

- **Section 3.3 loss equation**: The diversity loss λ(s_i-1)² might be incorrectly implemented. Usually diversity loss is between tokens, not scalar scale. The paper's description is ambiguous.
- **Checkpoint resume bug**: Section 3.4.3 describes bugs that caused retraining. Are these fully fixed? The paper doesn't provide unit test evidence.
- **Source layer bug**: Section 6.12.4 shows layer 8 performs poorly. Could this be due to residual connections not being properly normalized at early layers?

**6. Questions for Authors**

1. What is the exact p-value for Bridge vs. Prompt-Tuning on AG News? With ±5% std dev and 3 seeds, I doubt it's significant.
2. Why not compare to a joint sender+receiver prompt-tuning baseline?
3. How was the 200-sample test set selected? Random or cherry-picked?
4. What is the false positive rate for the 3B threshold? Did any >3B models fail?
5. Can you compress 10K token prompts? The evaluation stops at 500 tokens.

**7. Missing References**

- **AdapterFusion**: Pfeiffer et al., 2021 - for combining multiple model representations
- **Scaling Prompts**: Lester et al., 2021 - for proper baseline on 1B+ models
- **Multi-task prompt learning**: Vu et al., 2022 - for comparing to universal prompt methods
- **Cross-modal attention**: Tsai et al., 2019 - for theoretical analysis of cross-attention capacity

**8. Suggestions for Improvement**

1. Run statistical significance tests with proper multiple comparison correction
2. Increase sample size to at least 1K per dataset and seeds to 10
3. Add joint sender+receiver prompt-tuning baseline
4. Evaluate on generation tasks to justify "communication" framing
5. Test on 70B models to validate scaling projections
6. Provide training cost in GPU-hours per model pair
7. Add cross-task generalization experiments
8. Release training code and model weights for reproducibility

---

### Reviewer 6: Reproducibility & Methodology Expert

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 4/5
- **Recommendation**: Weak Reject

**Summary**
LatentWire presents an interesting approach but suffers from insufficient experimental rigor, missing details, and potential reproducibility issues.

**Strengths**

1. **Training debugging transparency**: Section 3.4's honest discussion of bugs (shuffling, resume, distribution alignment) is excellent for reproducibility.
2. **Comprehensive ablation table**: Table 16 provides clear ablation results that others can replicate.
3. **Detailed hyperparameters**: Appendix A.1 lists optimal values across scales.
4. **Validation of embedding interface**: Section 6.8's embedding baseline validation is methodologically sound.
5. **Control experiments**: Zero-prefix control and miscalibration controls properly isolate variables.

**Weaknesses**

1. **No code release**: No mention of code or model availability. Reproducibility is impossible without it.
2. **Incomplete training details**: Curriculum schedule is described qualitatively but not specified algorithmically. No learning rate schedules provided.
3. **Sample size issues**: 200 evaluation samples is too small for reliable metrics.
4. **Seed dependency**: Only 3 seeds reported, but Figure 1 and Table 13 suggest high variance. Need more seeds.
5. **Data processing unclear**: How are prompts tokenized for the encoder? MiniLM uses WordPiece but Llama uses BPE—mismatch not addressed.

**Detailed Critique**

**5.1 Novelty Assessment**
Not my focus, but the methodological novelty (curriculum learning, first-token objective) is more significant than architectural novelty.

**5.2 Experimental Evaluation**
**Reproducibility red flags**:

- **Checkpoint selection**: Which checkpoint is reported? Best validation loss? Final epoch? Early stopping?
- **Evaluation protocol**: SST-2 accuracy is extracted via string matching. What about ambiguous outputs like "positive sentiment, but with some negatives"? No error analysis.
- **Hyperparameter selection**: Table 22 shows "optimal" values but not how they were selected (grid search? random? manual?). No search space reported.

**Missing methodology details**:

- **Data splits**: How are train/val/test split? SST-2 has standard splits but AG News and TREC splits aren't specified.
- **Prompt formatting**: The "Answer:" anchor is used, but no ablation of other anchors ("Response:", "Classification:"). How sensitive is it?
- **Layer selection**: Source layer 16 is used, but no justification beyond Table 19. Why not optimize per-task?

**5.3 Ablation Studies**
**Incomplete ablation design**:

- **No interaction terms**: Table 16 ablates one factor at a time. What about interactions (e.g., calibration + first-token objective)?
- **No ablation of training schedule**: Does curriculum learning help beyond just more steps? Need to ablate warm-up length.
- **No encoder architecture comparison**: Only SimpleEncoder vs ByteEncoder is mentioned, but ByteEncoder results aren't shown.

**5.4 Technical Soundness**
**Potential methodological flaws**:

- **LayerNorm placement**: Section 3.2.2 puts LayerNorm before projection. Most transformers use post-norm. This could cause instability.
- **Calibration order**: Section 3.3 calibrates after adapter but before injection. Why not calibrate before adapter? The order matters.
- **Diversity loss implementation**: The description "λ(s_i-1)²" is ambiguous. Is s_i a scalar per adapter or per token? Implementation details missing.

**Data leakage risk**: The encoder uses MiniLM, which was trained on BookCorpus and Wikipedia. If evaluation data overlaps, this inflates results. No data decontamination analysis.

**6. Questions for Authors**

1. Will you release code, model weights, and evaluation scripts?
2. What is the exact algorithm for curriculum learning? Provide pseudocode.
3. How were hyperparameters selected? Report search spaces and costs.
4. What about LayerNorm pre- vs post-norm? Did you try both?
5. Can you provide failure case examples from the 200 evaluation samples?

**7. Missing References**

- **Best practices for LLM evaluation**: Liang et al., 2023 (HELM) - for standardized evaluation
- **Reproducibility in ML**: Henderson et al., 2018 - for reporting standards
- **LayerNorm variants**: Xu et al., 2019 - for stability analysis

**8. Suggestions for Improvement**

1. Release full code, training scripts, and model checkpoints
2. Provide detailed hyperparameter search procedure and costs
3. Increase evaluation sample size to 1K+ and seeds to 10
4. Add error bars to all tables and figures
5. Include failure case analysis with concrete examples
6. Report variance across training runs, not just seeds
7. Add data decontamination verification
8. Provide algorithmic specification of curriculum schedule
9. Test with different random seeds for curriculum phases
10. Report total compute cost in GPU-hours for full experimental suite

---

### Reviewer 7: NLP Applications Researcher

**Overall Assessment**

- **Score**: 6/10
- **Confidence**: 4/5
- **Recommendation**: Weak Accept

**Summary**
LatentWire evaluates cross-model soft prompt transfer on three classification benchmarks, showing competitive performance but limited task diversity.

**Strengths**

1. **Appropriate classification benchmarks**: SST-2, AG News, TREC cover binary, multi-class, and fine-grained classification.
2. **Multi-seed evaluation**: Tables report mean ± std across 3 seeds, showing awareness of variance.
3. **Task-specific tuning insights**: Section 6.12.1's analysis of binary classification failure is valuable.
4. **Comparison to strong baselines**: Includes few-shot and zero-shot baselines for both models.
5. **Generation quality analysis**: Table 9's pattern distribution shows careful manual inspection.

**Weaknesses**

1. **Limited task diversity**: Only classification. No generation, summarization, QA, or reasoning.
2. **Outdated benchmarks**: SST-2 and AG News are from 2011/2004. Modern benchmarks like SuperGLUE or MMLU would be more relevant.
3. **No error analysis**: No qualitative examples of failure cases or success cases.
4. **Simple classification may not stress compression**: 300-500 token prompts are short. Benefits may vanish with longer contexts.
5. **No comparison to in-context learning**: How does Bridge compare to simply providing 5-shot examples in text?

**Detailed Critique**

**5.1 Novelty Assessment**
**Task-specific novelty**: The binary classification fix (disabling diversity loss) is novel and useful. The cross-model transfer itself is incremental.

**Comparison to prompt-tuning**: Table 13 shows Bridge beats prompt-tuning on AG News (+6.1%) and TREC (+9.8%). However, prompt-tuning is trained on Mistral only, which is weaker than Llama on these tasks. A fair comparison would train prompt-tuning on Llama's representations (e.g., via distillation).

**5.2 Experimental Evaluation**
**Missing NLP tasks**:

- **Summarization**: CNN/DM, XSUM
- **QA**: SQuAD, Natural Questions
- **Reasoning**: GSM8K, MATH
- **Code**: HumanEval

**Benchmark quality**: AG News is trivial (94% accuracy with simple bag-of-words). Modern models get >95% zero-shot. The ceiling effect makes improvements hard to interpret.

**Evaluation metric issues**:

- **Classification accuracy**: Extracting labels from free-form text is brittle. Better to use constrained decoding or classification head.
- **No calibration metrics**: Report ECE (expected calibration error) to see if soft prompts affect model confidence.
- **No robustness evaluation**: Test on adversarial variants of these datasets.

**5.3 Ablation Studies**
**Missing task-specific ablations**:

- **Input length scaling**: Does compression ratio matter more for 100 vs 1000 token inputs?
- **Output format sensitivity**: Does performance depend on how the answer is formatted ("positive" vs. "sentiment: positive")?
- **Prompt template sensitivity**: How much does the anchor text matter?

**5.4 Technical Soundness**
**Task mismatch**: Section 6.1 describes SQuAD experiments but main results are classification. The paper seems to have omitted SQuAD results from main tables, suggesting poor performance.

**Evaluation protocol**: The "extract first predicted class label" method is not described in detail. What string matching is used? What about partial matches? This lack of specificity hurts reproducibility.

**6. Questions for Authors**

1. Why not evaluate on modern benchmarks like MMLU or SuperGLUE?
2. Can you show qualitative examples of soft token vs. text prompting?
3. What is the accuracy drop as input length increases from 100 to 1000 to 4000 tokens?
4. How does performance vary with different prompt templates?
5. Why are SQuAD results relegated to Phase 1? What were final SQuAD numbers?

**7. Missing References**

- **Modern classification benchmarks**: Wang et al., 2022 (SuperGLUE), Hendrycks et al., 2021 (MMLU)
- **Long-context evaluation**: Bai et al., 2023 (ZeroSCROLLS)
- **Prompt sensitivity**: Zhao et al., 2021 - for prompt formatting analysis
- **Constrained decoding**: Geng et al., 2023 - for robust classification evaluation

**8. Suggestions for Improvement**

1. Add generation task evaluation (summarization, QA) to justify "communication" framing
2. Evaluate on modern challenging benchmarks (MMLU, Big-Bench)
3. Provide detailed error analysis with qualitative examples
4. Test robustness to prompt template variations
5. Report performance vs. input length scaling
6. Compare to in-context learning with optimal prompting
7. Use constrained decoding for cleaner classification evaluation
8. Include SQuAD results in main paper with full analysis

---

### Reviewer 8: Efficiency/Compression Specialist

**Overall Assessment**

- **Score**: 7/10
- **Confidence**: 5/5
- **Recommendation**: Accept

**Summary**
LatentWire achieves impressive compression (18.6× on 7-8B models) with competitive accuracy, though compression-quality tradeoffs need deeper analysis.

**Strengths**

1. **Strong compression ratios**: 15-30× token reduction with minimal accuracy loss.
2. **Constant-size communication**: M=8 tokens regardless of input length is valuable for bandwidth-constrained scenarios.
3. **Speedup validation**: Table 6 shows 4.0× wall-clock speedup on 7-8B models, which is significant.
4. **Ablation of compression factors**: Table 17 shows M scaling, demonstrating capacity-accuracy tradeoffs.
5. **Theoretical scaling analysis**: Section 7.3 provides sound asymptotic analysis.

**Weaknesses**

1. **Compression is lossy**: The paper admits semantic-only compression (Section 8.4), making it unsuitable for precision tasks.
2. **No comparison to SOTA compressors**: LLMLingua-2, ICAE, and 500xCompressor achieve comparable/higher compression but aren't directly compared.
3. **Fixed M is suboptimal**: Simple tasks need fewer tokens; complex tasks need more. Adaptive M would improve efficiency.
4. **No analysis of compression artifacts**: What information is lost? No reconstruction error analysis.
5. **Payload size vs. KV cache**: Bridge's 8KB payload (fp16) is tiny, but KV cache for M tokens is still O(M·L·d), not constant as claimed.

**Detailed Critique**

**5.1 Novelty Assessment**
**Compression novelty**: The cross-model aspect is new for soft-prompt compression. However, the architecture (PerceiverResampler) is borrowed from vision-language, and the training objectives are standard.

**Comparison to ICAE**: ICAE achieves 4-15× compression with a single model. LatentWire's 18.6× is comparable but adds cross-model overhead. The paper should compare per-token reconstruction loss to ICAE.

**5.2 Experimental Evaluation**
**Missing compression analysis**:

- **Perplexity**: Report receiver's perplexity on gold text with vs. without bridge.
- **Reconstruction error**: For tasks where answer is extractable, measure token overlap between generated and gold text.
- **Entropy analysis**: What is the information content (bits) in M soft tokens vs. n text tokens?

**Efficiency metrics**:

- **FLOPs**: Report actual floating-point operations saved, not just token count.
- **Memory bandwidth**: Quantify memory traffic reduction.
- **Energy**: Estimate power consumption savings.

**5.3 Ablation Studies**
**Missing compression ablations**:

- **No quantization**: Test int8/int4 for bridge parameters. Does accuracy drop significantly?
- **No distillation analysis**: The bridge is trained with KL divergence? The loss is only cross-entropy on answers. This is indirect compression.

**5.4 Technical Soundness**
**False payload claim**: Section 8 claims "constant 8KB payload" but ignores that the receiver must still compute KV cache for M tokens, which is M·L·d = 8·32·4096 = 1M parameters = 2MB at fp16. This is not constant across model sizes.

**Compression-quality tradeoff**: Table 17 shows TREC performance drops from 97.5% (M=8) to 70.5% (M=4). This is a 27-point drop for 2× compression—very steep. The paper doesn't discuss this fragility.

**6. Questions for Authors**

1. What is the reconstruction perplexity on held-out text? How close to text baseline?
2. Can you compress arithmetic reasoning problems? ("What is 123+456?")
3. How does performance change if you quantize bridge to int8?
4. What is the information-theoretic compression ratio (entropy of soft tokens vs. text)?
5. Why is payload claimed as 8KB when KV cache scales with M?

**7. Missing References**

- **VQ-VAE for compression**: Razavi et al., 2019 - for discrete bottleneck comparisons
- **Model compression survey**: Cheng et al., 2018 - for broader context
- **Efficient transformers**: Tay et al., 2022 - for attention complexity analysis

**8. Suggestions for Improvement**

1. Add reconstruction perplexity evaluation
2. Evaluate on precision tasks (arithmetic, passkey retrieval)
3. Test quantization of bridge parameters
4. Analyze information content of soft tokens
5. Clarify payload vs. KV cache distinction
6. Add adaptive M selection based on input complexity
7. Compare token-by-token reconstruction to baselines
8. Report memory bandwidth savings explicitly

---

### Reviewer 9: Writing & Presentation Critic

**Overall Assessment**

- **Score**: 5/10
- **Confidence**: 5/5
- **Recommendation**: Weak Reject

**Summary**
The paper presents solid technical work but is hindered by poor organization, inconsistent notation, and unclear exposition that will confuse readers.

**Strengths**

1. **Clear problem statement**: Section 1 opens with a compelling narrative about multi-model communication costs.
2. **Good visualizations**: Figure 2's t-SNE and Figure 1's throughput plot are effective.
3. **Honest discussion of failures**: Section 3.4's training challenges and Section 8's limitations are transparent.
4. **Comprehensive tables**: Tables 6, 7, 13 provide extensive data.
5. **Use of appendices**: Additional details in Appendix A don't clutter the main paper.

**Weaknesses**

1. **Notation is inconsistent**: Uses M, m, dz, di, Li interchangeably. Sometimes bold for vectors, sometimes not. No notation table.
2. **Structural issues**: Section 3.4 (training challenges) interrupts the method flow. Should be in training subsection or appendix.
3. **Redundant content**: Section 6.1 (Phase 1) describes preliminary experiments that should be in appendix.
4. **Unclear writing**: Many sentences are convoluted, e.g., "The bridge is bidirectional:reverse transfer (Mistral→Llama)achieves 97.0%on SST-2" (missing space, unclear what "bidirectional" means).
5. **Missing definitions**: Key terms like "soft token," "interlingua," "wire protocol" are used before definition.

**Detailed Critique**

**5.1 Novelty Assessment**
Writing quality doesn't affect novelty, but unclear exposition makes it hard to assess. The paper mixes architectural novelty with engineering details.

**5.2 Experimental Evaluation**
**Unclear evaluation protocol**: Section 5.3.1 describes zero-shot prompts but doesn't specify extraction method. Section 6.11.1 mentions "extracting first predicted class label" but details are in unstated "supplementary material."

**Table inconsistencies**:

- Table 13 uses "mean ± std" but Table 14 reports "Average" without variance.
- Table 16 mixes absolute and relative changes in "%Change" column.
- Table 17 uses "Best per dataset in bold" but bolds entire rows, not cells.

**5.3 Ablation Studies**
**Ablations are scattered**: Table 16 (full system), Table 17 (M scaling), Table 18 (bidirectional), Table 19 (source layer) are in different sections. Consolidate into one place.

**5.4 Technical Soundness**
**Notation errors**:

- Equation (1) is malformed: "$E:\ mathrm{T e x t}\to\mathbb{R}^{M\times d_{z}}$" has syntax errors.
- Section 3.2.1: "Q ∈ RMXd （learnedqueries" is broken LaTeX.
- Section 4.4: "$\alpha$ ≈ 0$.5-1$ .0based" is garbled.

**Clarity issues**:

- **"Capacity threshold"**: Never formally defined. Is it parameter count? Attention dimension? FLOPs?
- **"Exposure bias"**: Misapplied term. Exposure bias typically refers to autoregressive training dynamics, not static prefix injection.
- **"Telepathically"**: Informal language in abstract (Section 1) is inappropriate for scientific paper.

**6. Questions for Authors**

1. Can you provide a notation table with consistent symbols?
2. Why is Section 3.4 placed before training algorithm description?
3. Can you move Phase 1 experiments to appendix and add a clear flowchart?
4. What is the exact string matching algorithm for classification extraction?
5. Can you unify notation for diversity loss (s_i vs. λ)?

**7. Missing References**

- **Style guide**: Strunk & White - for sentence clarity
- **Scientific writing**: Glasman-Deal, 2010 - for structure

**8. Suggestions for Improvement**

1. Create a notation table at the start of Section 3
2. Move Section 3.4 to a "Training Stability" subsection under 3.3
3. Move Section 6.1 to Appendix
4. Provide algorithm pseudocode for the full training procedure
5. Unify table formatting (consistent rounding, units, bolding)
6. Define all key terms before use
7. Have a native English speaker edit for grammar and flow
8. Add a "Summary of Contributions" bullet list after abstract
9. Provide a system diagram (Figure 1 is throughput, not architecture)
10. Standardize figure fonts and sizes

---

### Reviewer 10: Industry Practitioner

**Overall Assessment**

- **Score**: 4/10
- **Confidence**: 4/5
- **Recommendation**: Weak Reject

**Summary**
LatentWire is an interesting research prototype but faces significant barriers to production adoption: high training cost, model-specific bridges, and unclear advantages over simpler solutions.

**Strengths**

1. **Addresses real pain point**: Cross-model communication is indeed a bottleneck in multi-agent systems.
2. **Frozen model compatibility**: Works with any frozen checkpoint above 3B, avoiding fine-tuning costs.
3. **Clear performance improvements**: 27× speedup and competitive accuracy are compelling if they hold in production.
4. **Bidirectional capability**: Reduces deployment complexity (one bridge per pair vs. two).
5. **Training efficiency**: 30 min per task on H100 is reasonable for a one-time cost.

**Weaknesses**

1. **Training cost per model pair is prohibitive**: 537M parameters × 30 min × N tasks × M model pairs = huge cost. For 10 models × 10 tasks = 100 bridges.
2. **No incremental adaptation**: If Llama-3.1 is updated to Llama-3.2, the bridge needs full retraining. No warm-start or adaptation method.
3. **Unclear ROI**: Text-relay's main cost is developer complexity, not inference. Bridge adds training/maintaining custom adapters.
4. **Debugging is harder**: Soft token failures are opaque compared to readable text. No introspection tools provided.
5. **No production-grade features**: No monitoring, logging, A/B testing framework, or guardrails for soft token inputs.

**Detailed Critique**

**5.1 Novelty Assessment**
**From industry perspective**: Novelty is secondary to practicality. The paper's contributions are intellectually interesting but not clearly productizable.

**5.2 Experimental Evaluation**
**Missing production considerations**:

- **Cold start**: How many training samples needed? The paper uses 2K steps × 16 batch = 32K samples per task. This is expensive to collect.
- **Online learning**: Can the bridge adapt to user feedback without retraining?
- **Model versioning**: How do bridges handle model checkpoint updates?
- **Multi-tenancy**: Can one bridge serve multiple users with different tasks?

**Scalability concerns**:

- **Bridge storage**: 537M parameters × fp16 × 10 model pairs × 10 tasks = 107GB storage just for bridges.
- **GPU memory**: Two models + bridge + KV caches may not fit on consumer GPUs.
- **Latency variance**: No evaluation of p99 latency, which matters more than average for production.

**5.3 Ablation Studies**
**Missing deployment ablations**:

- **Quantization**: What is int8 accuracy? Required for production serving.
- **Pruning**: Can bridge parameters be reduced 50% with minimal loss?
- **Caching**: Can soft tokens be cached for repeated similar inputs?

**5.4 Technical Soundness**
**API design**: The paper assumes `inputs_embeds` interface, but production systems often use higher-level APIs (OpenAI, Anthropic) that don't expose embeddings. The approach is limited to self-hosted models.

**Error handling**: Section 4 shows small models produce garbage. In production, you need:

- **Detection**: How to detect bridge failure? No confidence score provided.
- **Fallback**: Automatically revert to text-relay when bridge fails.
- **Alerting**: Monitor bridge drift over time.

**6. Questions for Authors**

1. What is the total cost to deploy Bridge for 5 model pairs across 20 tasks? Include training and storage.
2. Can bridges be quantized to int8 for production serving?
3. How do you monitor bridge quality in production? What metrics?
4. What happens when a model is updated? Is retraining required?
5. Can you A/B test bridge vs. text-relay safely?

**7. Missing References**

- **ML Ops**: Kreuzberger et al., 2023 - for model deployment challenges
- **LLM serving systems**: Kwon et al., 2023 (vLLM) - for production serving
- **Cost analysis**: Patterson et al., 2021 - for carbon/energy costs

**8. Suggestions for Improvement**

1. Add production deployment cost analysis (training + inference)
2. Evaluate quantization (int8/int4) for bridge parameters
3. Provide monitoring and fallback mechanisms
4. Show A/B testing framework for safe rollout
5. Discuss model versioning and update strategies
6. Evaluate multi-tenancy scenarios
7. Add latency variance (p50, p99, p99.9) analysis
8. Compare to commercial solutions (e.g., routing via text)
9. Provide bridge introspection tools (visualize what soft tokens represent)
10. Discuss data privacy: does bridge training leak training data?

---

## Meta-Review

### Consensus Summary

The committee agrees on several points:

**Strengths**:

- **Clear problem identification**: The prefill bottleneck in multi-LLM systems is real and important.
- **Strong empirical validation**: The 3B parameter threshold is well-supported and represents a novel, practically useful finding.
- **Comprehensive ablations**: Table 16 is exemplary in its thoroughness.
- **Training stability contributions**: First-token objective and curriculum learning are valuable methodological improvements.
- **Honest presentation**: Authors openly discuss failures (Qwen→Mistral, binary classification issues).

**Weaknesses**:

- **Limited experimental scope**: Only three simple classification tasks, no generation or reasoning evaluation.
- **Questionable speedup claims**: 27× vs. naive text-relay is overstated; comparison to optimized systems is needed.
- **Statistical rigor**: 3 seeds and 200 samples per dataset is insufficient for strong claims.
- **Production practicality**: High per-pair training cost and lack of incremental adaptation hinder deployment.
- **Writing quality**: Inconsistent notation, unclear structure, and informal language.

### Key Controversies

1. **Novelty vs. C2C**: Reviewers disagree on whether embedding-space vs. KV-cache-space communication is a meaningful distinction. Systems experts (R1) think it's incremental; prompt specialists (R2) see methodological value.

2. **3B threshold validity**: Some reviewers (R5) believe this restates Lester et al.'s prompt-tuning threshold, while others (R4) see cross-model transfer as a new dimension.

3. **Speedup measurement**: R1 and R5 argue the 27× claim is inflated; R8 finds the compression ratio compelling. Need standardized comparison against optimized text-relay.

4. **Benchmark selection**: R7 and R5 argue classification tasks are insufficient; R2 and R8 think they're adequate for proving the concept.

5. **Production viability**: R10 is skeptical about deployment costs; R1 and R4 see research value regardless.

### Critical Issues

**Must address for acceptance**:

1. **Statistical rigor**: Increase seeds to ≥10 and evaluation samples to ≥1K per dataset. Report significance tests.
2. **Speedup validation**: Compare against properly optimized text-relay using vLLM or similar. Report p50/p99 latency.
3. **Novelty clarification**: Explicitly differentiate from C2C and LatentMAS in a dedicated "Comparison to Concurrent Work" section.
4. **Broader evaluation**: Add at least one generation task (e.g., summarization) to justify "communication" claims.

**Strongly recommended**: 5. **Code release**: Essential for reproducibility and should be a condition of acceptance. 6. **Scaling analysis**: Test on 70B+ models to validate projections. 7. **Production considerations**: Discuss quantization, monitoring, and update strategies. 8. **Writing overhaul**: Fix notation, add notation table, move Phase 1 to appendix.

### Venue Recommendation

**Primary: MLSys** - Appropriate if authors strengthen systems contributions (proper speedup evaluation, scalability analysis).

**Alternative: NeurIPS/ICLR** - If focus remains on representation learning and methodological contributions, these venues would be better fits.

**Not suitable for**: ACL/EMNLP (insufficient language-specific contributions) or pure ML venues without systems evaluation.

### Overall Verdict

**Weak Accept contingent on major revisions**. The paper addresses an important problem and has solid technical contributions, particularly the capacity threshold analysis. However, overstated speedup claims, limited evaluation, and insufficient statistical rigor prevent strong acceptance. The authors must address critical issues #1-4 for the paper to be acceptable at MLSys.

### Revision Roadmap

#### Critical (Must Fix)

1. **Increase statistical power**: 10+ seeds, 1000+ evaluation samples, report significance tests with multiple comparison correction.
2. **Validate speedup fairly**: Compare against optimized text-relay with vLLM batching; report latency distributions (p50, p99, p99.9).
3. **Clarify novelty**: Add section explicitly comparing to C2C, LatentMAS, LLM Modules with architectural diagrams.
4. **Add generation task**: Evaluate on XSUM or CNN/DM to demonstrate general communication ability.
5. **Release code**: Commit to releasing code, models, and evaluation scripts (required for final acceptance).

#### Major (Significantly Strengthen)

6. **Test 70B models**: Validate scaling projections empirically.
7. **Quantization analysis**: Evaluate int8/int4 bridge compression.
8. **Model pair diversity**: Test at least 2-3 more model pairs (e.g., Llama→Gemma, Qwen→Mistral).
9. **Cross-task generalization**: Train on one task, evaluate on another to test universality.
10. **Hyperparameter sensitivity**: Report curves, not just optimal points.
11. **Error analysis**: Provide concrete examples of success/failure cases.
12. **Production cost analysis**: Compute total cost of ownership for deployment.

#### Minor (Nice to Have)

13. **Notation table**: Create unified notation at start of methods.
14. **Algorithm pseudocode**: Provide formal training algorithm.
15. **Move Phase 1**: Relocate to appendix with clearer motivation.
16. **Writing polish**: Professional editing for grammar and flow.
17. **System diagram**: Add figure showing full inference pipeline.
18. **Interactive visualization**: Provide tool to explore soft token space.
19. **License discussion**: Address model license compatibility (Llama vs. Mistral).
20. **Security analysis**: Discuss adversarial soft token risks.

---

# Gemini

# LatentWire Peer Review Report

## Individual Reviews

### Reviewer 1: MLSys Systems Expert

**Score:** 7
**Confidence:** 4
**Recommendation:** Weak Accept

**Summary:**
The paper introduces LatentWire, a mechanism to compress context from a sender LLM into soft tokens for a receiver LLM using a PerceiverResampler adapter. This bypasses text generation, achieving significant speedups and data compression for model-to-model communication.

**Strengths:**

- **Throughput Analysis:** Figure 1 provides a compelling demonstration of batching efficiency. Soft tokens allow uniform tensor shapes (), enabling effective batching that variable-length text generation prohibits.
- **Latency Improvements:** The 27x speedup claim, while baseline-dependent, is mechanically sound due to the removal of the autoregressive decoding step on the sender side.
- **Frozen Model constraint:** Keeping LLMs frozen is a strong systems decision, minimizing deployment overhead compared to C2C or methods requiring full fine-tuning.

**Weaknesses:**

- **Baseline Straw-manning:** The "Text-Relay" baseline assumes the sender must generate a full summary. In many systems, the baseline is simply passing the original prompt to the receiver (if bandwidth allows) or a simple extraction. The speedup is valid for _summarization-based_ workflows but not all communication.
- **Memory Overhead:** The adapter is 537M parameters. For a 7B model, this is an ~8% memory overhead. This impact on vRAM and serving capacity is not sufficiently analyzed.
- **Hardware Specificity:** Results are H100-specific. Does this bandwidth saving hold on consumer hardware or older datacenter cards (A100/V100)?

**Detailed Critique:**
The distinction from **Cache-to-Cache (Fu et al., 2025)** is important. C2C fuses KV-caches, which is heavier on memory bandwidth during transfer but potentially richer. LatentWire operates at the embedding interface. The paper should explicitly compare inference memory bandwidth consumption against C2C. The 27x speedup is impressive, but it compares to a slow autoregressive process. A "speedup vs. raw text transfer" (assuming infinite bandwidth vs limited bandwidth) trade-off curve would be more rigorous for MLSys.

**Questions for Authors:**

1. How does the 537M parameter adapter affect the maximum batch size on a single GPU compared to standard inference?
2. Can you provide a roofline model analysis of where the speedup comes from (compute bound vs memory bound)?

---

### Reviewer 2: Prompt Tuning Specialist

**Score:** 5
**Confidence:** 5
**Recommendation:** Borderline

**Summary:**
The authors propose a learned interlingua using soft tokens to transfer information between heterogeneous LLMs. It extends concepts from soft prompting and prompt compression to a cross-model setting.

**Strengths:**

- **Cross-Model Transfer:** Successfully demonstrating transfer from Llama to Mistral via embeddings is a non-trivial extension of **Su et al. (2022)**.
- **Bidirectional Analysis:** The Mistral Llama reverse transfer result (97% SST-2) is a strong validator that the method isn't just overfitting to one architecture.

**Weaknesses:**

- **Incremental Novelty:** The technique is essentially **Prompt Tuning (Lester et al., 2021)** + **PerceiverResampler (Alayrac et al., 2022)** applied to the output of one model.
- **Gist Token Comparison:** The paper dismisses **Gist Tokens (Mu et al., 2023)** too quickly. Gist tokens also compress context. The distinction that LatentWire is "cross-family" is valid, but the mechanism is very similar.
- **Limited "Language":** Calling it an "Interlingua" suggests a universal protocol. The failure of Qwen Mistral suggests this is a "Pair-Specific Link," not a shared language.

**Detailed Critique:**
The paper claims the 3B threshold is a key discovery. However, **Lester et al. (2021)** already established that prompt tuning only becomes competitive with fine-tuning around the 1B-10B parameter scale. This paper confirms that finding applies to _decoding_ compressed representations, which is useful but not ground-breaking. The comparison to **500xCompressor** and **ICAE** needs to be more rigorous—specifically, can those methods be adapted for cross-model use simply by changing the tokenizer?

**Questions for Authors:**

1. Is the PerceiverResampler strictly necessary, or would a simple linear projection (like in **LLM Modules**) work if trained for longer?
2. How does the "Interlingua" claim hold up given that you must retrain the adapter for every specific task and model pair?

---

### Reviewer 3: Multi-Agent Systems Researcher

**Score:** 3
**Confidence:** 5
**Recommendation:** Reject

**Summary:**
LatentWire proposes a communication protocol for multi-LLM systems using soft tokens. It is evaluated on classification benchmarks.

**Strengths:**

- **Efficiency:** For high-frequency agent communication, the low bandwidth is appealing.

**Weaknesses:**

- **Task Selection:** SST-2, AG News, and TREC are not representative of modern multi-agent workloads (e.g., coding, math, complex planning). **LatentMAS** uses GSM8K and MATH. The omission of reasoning tasks is critical.
- **Precision Loss:** The paper admits in Section 8.4 that "Precision tasks fail" (arithmetic, passkeys). This renders the system unusable for the vast majority of agentic tool-use scenarios where exact parameters (JSON) must be passed.
- **Training Overhead:** Agents need to be generalists. If I need to train a specific adapter for every task my agent might encounter, the system is not scalable.

**Detailed Critique:**
The paper frames itself as a solution for "Heterogeneous LLM Communication" (Title), but it is actually a "Heterogeneous Classification Transfer" system. A true agent communication protocol needs to handle open-ended generation. The **LatentMAS (Zou et al., 2025)** and **Anonymous (DroidSpeak)** papers tackle broader collaboration. If LatentWire cannot pass a phone number or a SQL query reliably (due to being a "lossy semantic compressor"), its utility in the multi-agent space is severely limited to just "gisting" sentiment or topics.

**Questions for Authors:**

1. Can this system transfer instructions? E.g., if Llama generates a plan, can Mistral execute it based on the soft tokens?
2. Why were HotpotQA/SQUAD results excluded from the main cross-model table (Table 13)?

---

### Reviewer 4: Representation Learning Theorist

**Score:** 6
**Confidence:** 4
**Recommendation:** Weak Accept

**Summary:**
The paper investigates the learnability of a mapping between the embedding spaces of different LLMs using a PerceiverResampler. It identifies capacity thresholds and compatibility issues between model families.

**Strengths:**

- **3B Parameter Threshold:** The empirical analysis of the "Phase Change" between 1B and 3B models (Section 4.5) is scientifically valuable. It connects well to the "emergent abilities" literature.
- **Latent Space Visualization:** Figure 2 (t-SNE) provides good evidence that the soft tokens are learning a semantic topology, not just memorizing.
- **Negative Results:** Reporting the Qwen Mistral failure is intellectually honest and important for the community.

**Weaknesses:**

- **Platonic Hypothesis Weakness:** The authors cite the **Platonic Representation Hypothesis (Huh et al., 2024)**. However, if representations were truly converging to a platonic ideal, Qwen and Mistral (both strong models) should be compatible. The failure suggests the hypothesis does not hold for these architectures, or the method is flawed. The paper should discuss this contradiction more deeply.
- **Embedding vs. Attention:** The paper argues that embedding injection is superior to KV-cache fusion (**C2C**). Theoretically, why? KV-cache contains the processed contextual information. Embeddings are just the input. The PerceiverResampler has to do a lot of heavy lifting to simulate the "processing" that C2C gets for free.

**Detailed Critique:**
The paper needs to better disentangle _model capacity_ from _architectural compatibility_. Does Qwen fail because it is 7B (capacity) or because of its specific attention rotation/vocabulary (architecture)? The paper conflates these slightly. The control experiments with zero-gain prefixes are excellent for isolating the decoding failure.

**Questions for Authors:**

1. Does the Qwen Mistral failure persist if you use a larger Qwen model (e.g., Qwen-72B Mistral)? This would isolate size vs. architecture.

---

### Reviewer 5: Skeptical Senior Reviewer

**Score:** 4
**Confidence:** 5
**Recommendation:** Weak Reject

**Summary:**
A method to compress prompts into vectors for cross-model transfer. Claims high speedups and "telepathic" communication.

**Strengths:**

- Extensive ablation studies (Section 6.12).
- Clear identification of failure modes (1B models, binary classification).

**Weaknesses:**

- **Overblown Speedup Claims:** The 27x speedup is against a "Text-Relay" that generates a summary. **Cache-to-Cache (Fu et al.)** achieves 2x, **LatentMAS** 4x. These are realistic numbers. 27x is achieved by comparing to a very slow strawman. If the baseline was "send the input ID pointers" (if on same node) or just "send text" (without summarization), the speedup vanishes or becomes a slowdown due to the adapter.
- **Fragility:** The fact that SST-2 required completely different hyperparameters (disabling diversity loss, changing , changing LR) to work (Section 6.11) suggests the method is extremely brittle. A "wire protocol" shouldn't need retuning for the number of classes.
- **Task Specificity:** Table 13 shows results for specific trained tasks. Does this generalize? If I train on AG News, does it work on minimal variations? The lack of Zero-Shot Transfer evaluation (Section 8.1 admits this) is a major flaw.

**Detailed Critique:**
The paper sells "General Communication" but delivers "Task-Specific Compression." **LLM Modules (Kolomeitsev et al.)** and **Cross-LoRA** attempt more general transfer. The paper acknowledges **C2C** and **LatentMAS** but dismisses them too easily. C2C's KV-fusion is a more robust way to transfer state than compressing everything into 8 input tokens. The "Interlingua" title is misleading; it's a "Learned Task-Specific Compressor."

**Questions for Authors:**

1. Why is the method so sensitive to the number of classes (SST-2 vs AG News)?
2. Did you try training a single universal adapter on a mixture of tasks?

---

### Reviewer 6: Reproducibility & Methodology Expert

**Score:** 8
**Confidence:** 5
**Recommendation:** Accept

**Summary:**
Detailed experimental paper on cross-model soft prompting. The authors devote significant space to "Training Challenges and Solutions" which is excellent for reproducibility.

**Strengths:**

- **Section 3.4 (Training Challenges):** Documenting the "shuffling bug," "resume bug," and "exposure bias" is fantastic. This adds real value to the community.
- **Detailed Hyperparameters:** Appendix A and Section 5.5 provide exact learning rates, batch sizes, and hardware details.
- **Code/Data Availability:** (Assuming code is submitted/promised).

**Weaknesses:**

- **Missing SQUAD/HotpotQA in Main Results:** Table 7 shows SQUAD results for "Same-family" (Llama->Llama). Table 12 shows Joint Rescoring. But Table 13 (Cross-model) _only_ has classification. The paper implies cross-model works for QA, but doesn't explicitly show Llama Mistral SQUAD F1 scores. This omission is suspicious.
- **Error Bars:** Table 13 has error bars (std dev across 3 seeds). This is good practice. However, Table 17 (Ablation on ) is single-seed. Given the fragility mentioned in SST-2, ablations should also be multi-seed.

**Detailed Critique:**
The methodology is sound, but the selective reporting of tasks in the cross-model section is a red flag. If Llama Mistral failed on SQUAD/HotpotQA, it should be stated. The "First-Token Objective" (Eq 18) is a clever fix for the exposure bias problem and is well-justified.

**Questions for Authors:**

1. What is the Llama Mistral performance on SQUAD and HotpotQA?
2. Please confirm if the code will be open-sourced.

---

### Reviewer 7: NLP Applications Researcher

**Score:** 3
**Confidence:** 4
**Recommendation:** Reject

**Summary:**
Evaluates LatentWire on SST-2, AG News, and TREC.

**Strengths:**

- Beat prompt-tuning baselines on 2/3 tasks.

**Weaknesses:**

- **Inappropriate Benchmarks:** SST-2 and AG News are solved problems (accuracy > 90% zero-shot). Improving them via a complex 537M parameter adapter is not practically useful.
- **No Generative Evaluation:** The paper talks about "Communication" but only tests "Classification." Communication implies dialogue, instruction following, or information passing. Evaluating on ROUGE (Summarization) or BLEU (Translation) is standard for such claims.
- **Limited Scope:** The paper admits it cannot handle precision tasks. This excludes 90% of high-value NLP applications (RAG, Code, Math).

**Detailed Critique:**
The paper demonstrates that you can compress a document into 8 vectors such that a classifier can still guess the topic. This is "Feature Extraction," not "Communication." **LLMLingua** and **AutoCompressors** evaluate on summarization and QA retention. This paper needs to show that the _content_ of the message is preserved, not just the _class label_.

**Questions for Authors:**

1. Please provide ROUGE scores for a summarization task (e.g., CNN/DailyMail) transferred from Llama to Mistral.

---

### Reviewer 8: Efficiency/Compression Specialist

**Score:** 7
**Confidence:** 5
**Recommendation:** Accept

**Summary:**
Focuses on the compression ratio (15-30x) and the vector quantization aspects.

**Strengths:**

- **Constant Overhead:** The communication cost is regardless of input length . This is asymptotically superior to **LLMLingua** (which is just with a smaller constant).
- **Ablation of VQ-VAE:** Section A.4.1 details why discrete codes failed (Codebook collapse). This justifies the continuous approach.

**Weaknesses:**

- **Compression vs. Capacity:** The paper notes that works for classification. For longer contexts (e.g., 4k tokens), is likely insufficient. The paper does not explore the scaling law of vs. Input Length.
- **Parameter Efficiency:** The adapter is 537M parameters. **Prompt Tuning** is usually <1M params. The storage cost of LatentWire is significantly higher.

**Detailed Critique:**
The continuous compression is the right approach for maintaining semantics. The comparison to PCA (Section 6.9) proves that non-linear compression is required. The paper successfully argues that for large-scale systems (70B+), the bandwidth savings outweigh the adapter cost.

**Questions for Authors:**

1. Did you experiment with scaling for longer input contexts (e.g., >1000 tokens)?

---

### Reviewer 9: Writing & Presentation Critic

**Score:** 8
**Confidence:** 3
**Recommendation:** Accept

**Summary:**
Evaluates the clarity and structure of the paper.

**Strengths:**

- **Clear Visuals:** Figure 1 (Throughput) and Figure 2 (t-SNE) are high quality and informative.
- **Structure:** The paper follows a logical flow: Intro -> Method -> Capacity Discovery -> Experiments.
- **Writing:** Prose is generally high quality and engaging.

**Weaknesses:**

- **Hype:** Terms like "Telepathically" (Abstract) and "Telepathy" (Section A.4.1) are unscientific. "Latent communication" is sufficient.
- **Structure Issues:** SQUAD results appear in ablations but not main results. This is confusing for the reader.
- **Notation:** The notation is used for adapted latents, but also probability. Slightly overloading.

**Detailed Critique:**
The paper is well-written. The "Token Soup" tables (Tables 1 & 2) are very effective at illustrating the failure modes. The citation style is consistent.

**Suggestions:**

- Replace "Telepathically" with "via latent representations."
- Move the definition of earlier in the Method section.

---

### Reviewer 10: Industry Practitioner

**Score:** 4
**Confidence:** 5
**Recommendation:** Weak Reject

**Summary:**
Evaluates feasibility for real-world deployment.

**Strengths:**

- **Latency:** Low latency is always good.
- **Frozen Models:** Essential for industry adoption (we can't retrain base models).

**Weaknesses:**

- **Deployment Complexity:** I need to manage a matrix of adapters (). This is "Dependency Hell."
- **Reliability:** The "Precision tasks fail" limitation is a dealbreaker for most business logic (e.g., extracting dates, prices).
- **Qwen Failure:** If it doesn't work with Qwen (a major open model), it's risky to adopt.

**Detailed Critique:**
While the speedup is nice, the operational complexity of training and serving specific adapters for every task/pair outweighs the compute savings for most applications, unless the scale is massive (Google/Meta scale). For a generic "MLSys" contribution, it needs to be more "plug-and-play."

**Questions for Authors:**

1. What is the dollar cost of training one adapter?

---

## Meta-Review

### Consensus Summary

Reviewers agree that the **3B parameter capacity threshold** (Reviewers 1, 2, 4) is the paper's most robust and scientifically interesting finding, empirically validating that smaller models lack the machinery to decode compressed continuous representations. There is also consensus that the **throughput/batching benefits** (Reviewer 1, 8) are real and significant compared to text-based methods. However, almost all reviewers criticized the **baseline comparison** (Text-Relay via summarization) as inflating the speedup numbers.

### Key Controversies

- **Novelty vs. Prior Work:** Reviewers 2 and 5 feel the work is too derivative of Soft Prompting and C2C, while Reviewer 1 and 4 value the specific system architecture (PerceiverResampler on inputs_embeds) as a distinct contribution.
- **Utility:** Reviewer 7 and 3 (Applications/Agents) strongly reject based on the lack of reasoning/generative tasks and precision failure, whereas Reviewer 8 (Efficiency) values the compression regardless of task limitation.
- **"Interlingua" Claim:** Reviewers 2 and 4 dispute the term "Interlingua" given the Qwen failure and task-specificity, suggesting it's merely a "Learned Compressor."

### Critical Issues

1. **Missing Generative/Reasoning Benchmarks:** The paper claims communication but tests classification. The absence of SQUAD/HotpotQA in the _cross-model_ results (Table 13) is a major red flag that implies failure on complex tasks.
2. **Fragility:** The extreme hyperparameter sensitivity for SST-2 (binary classification) undermines the "robust wire protocol" narrative.
3. **Baseline Fairness:** The 27x speedup must be contextualized. It is 27x faster than _summarization_, but likely not faster than raw text transfer.

### Venue Recommendation

**MLSys:** Borderline. The paper has strong systems elements (throughput, memory analysis), but the core contribution is arguably an ML technique (soft prompting).
**Alternate:** EMNLP or ACL (Industry Track) might be better fits if the systems analysis isn't deepened, though the NLP reviewers were harsher on the benchmarks.

### Overall Verdict

**Weak Reject / Borderline.**
While the 3B threshold and efficiency analysis are valuable, the paper overclaims "Communication" while only delivering "Classification Transfer." The omission of reasoning/QA tasks in the main cross-model evaluation, combined with the fragility of the training process and the failure on Qwen, suggests the "LatentWire" is not yet a mature protocol. The novelty over C2C is present (embeddings vs KV) but the utility comparison is missing.

### Revision Roadmap

**Critical (Must Fix):**

1. **Add Generative Evaluation:** Include ROUGE/BLEU scores for a summarization or translation task Llama Mistral. If it fails, admit it.
2. **Clarify Cross-Model QA:** Explicitly report SQUAD/HotpotQA results for Llama Mistral in Table 13.
3. **Refine Speedup Claim:** Explicitly state "27x faster than summarization-based text relay" and add a comparison to "Raw Text Transfer" (bandwidth permitting).

**Major:**

1. **Compare to C2C/LatentMAS:** Add a qualitative or quantitative comparison of memory bandwidth usage vs. Cache-to-Cache.
2. **Address Qwen:** Discuss _why_ Qwen failed with more depth. Is it the vocabulary size? The attention mechanism?
3. **Tone Down Hype:** Remove "Telepathic" and "Interlingua" (unless generalization improves). Use "Learned Semantic Compression."

**Minor:**

1. **Fix Table consistency:** Ensure SQUAD results are consistently reported.
2. **Fix Notation:** Disambiguate .
3. **Typos:** Fix "inputs.embe" (page 3) and "inputs embeds" (page 4).

# Grok

# LatentWire Peer Review Report

## Individual Reviews

### Reviewer 1: MLSys Systems Expert

**Overall Assessment**  
Score: 6/10  
Confidence: 4/5  
Recommendation: Borderline

**Summary**  
The paper proposes LatentWire, a system that replaces text-based communication between heterogeneous LLMs with a compact sequence of learned soft tokens (typically M=8) injected directly into the input embedding space of the receiver model via a PerceiverResampler-based bridge. It claims 15–30× prefill reduction and 27× overall inference speedup for cross-model classification tasks while maintaining strong accuracy, and demonstrates constant-size overhead for long conversations.

**Strengths**

- Clearly identifies and quantifies the prefill bottleneck in multi-LLM systems, with clean theoretical analysis (O(n²) vs O(M²)).
- The constant-size payload for arbitrary conversation length is a genuine systems advantage over text serialization.
- Provides concrete training stabilizations (first-token loss, warm-up alignment, adapter regularization) that are valuable for practitioners attempting similar systems.
- Bidirectional transfer without retraining the bridge is impressive from a systems perspective.
- Detailed discussion of failure modes in small models adds practical insight.

**Weaknesses**

- The 27× speedup claim is not supported by end-to-end wall-clock measurements; it appears to be a theoretical prefill-operation count ratio that ignores encoder cost, generation phase, and GPU kernel launch overhead.
- No multi-turn experiment is presented, so the “constant overhead” claim remains theoretical.
- The bridge itself has ~537M parameters and requires a forward pass on the sender side—its latency is never reported or ablated.
- Only prefill phase is optimized; total inference time (prefill + decode) is not measured for realistic sequence lengths.
- No comparison of actual measured latency against C2C or other baselines.

**Detailed Critique**

**5.1 Novelty Assessment**  
The embedding-level channel is meaningfully different from KV-cache fusion (C2C), as it avoids layer-by-layer alignment complexity and enables constant-size messages. This is a valid systems contribution, though the speedup magnitude is overstated without full measurements.

**5.2 Experimental Evaluation**  
Baselines are reasonable but incomplete—no measured latency for any method. The 27× claim is derived from operation counts assuming M=8–16, but real systems are memory-bound, not compute-bound, for short prefixes. Missing: GPU timeline profiles, multi-turn wall-clock times, comparison to KV-cache reuse within a single model.

**5.3 Ablation Studies**  
Missing ablation of encoder cost and total inference time. The calibration and regularization ablations are present but systems-relevant ablations (e.g., different M, bridge size vs latency) are absent.

**5.4 Technical Soundness**  
Method is sound, but speedup methodology is not—theoretical O(n²) vs O(M²) ignores constant factors and memory bandwidth.

**5.5 Writing Quality**  
Clear overall, but speedup claims are presented too aggressively in the abstract and introduction without caveats.

**Questions for Authors**

1. Can you provide end-to-end wall-clock latency numbers (including encoder forward pass) on the same hardware for LatentWire vs text-relay?
2. What is the measured latency of the ~537M-parameter PerceiverResampler on a single A100/H100?
3. Why no multi-turn conversation experiment to validate constant overhead?
4. How does total inference time (prefill + decode 50 tokens) compare?

**Missing References**

- Recent KV-cache compression papers (e.g., FastGen, 2024; InfiniGen, 2025) for context on alternative prefill reductions.
- Systems papers on multi-model serving (e.g., Sarathi-Serve, 2024).

**Suggestions for Improvement**

- Add full-system latency measurements on standard hardware.
- Include a multi-turn dialogue experiment with cumulative timing.
- Ablate bridge size vs latency/accuracy tradeoff.

### Reviewer 2: Prompt Tuning / Soft Prompts Specialist

**Overall Assessment**  
Score: 7/10  
Confidence: 5/5  
Recommendation: Weak Accept

**Summary**  
LatentWire extends soft-prompt compression to heterogeneous LLMs by learning a shared continuous interlingua encoded via PerceiverResampler and decoded via lightweight adapters. It achieves strong classification performance with only 8 soft tokens and demonstrates bidirectional transfer across Llama and Mistral families.

**Strengths**

- First convincing demonstration of soft-prompt transfer across truly heterogeneous model families (different vocabularies, embedding scales).
- Careful engineering to stabilize training (first-token loss, warm-up alignment) that goes beyond prior soft-prompt work.
- Clear superiority over prompt-tuning baselines on 2/3 tasks despite cross-model handicap.
- Thorough analysis of why discrete bottlenecks (VQ-VAE) fail.
- Identifies and documents a sharp capacity threshold for soft-prompt decoding.

**Weaknesses**

- Compression ratio (37–75×) is good but not state-of-the-art compared to single-model compressors like 500xCompressor.
- Relies on classification tasks where short soft prompts suffice; no evidence it scales to complex reasoning.
- PerceiverResampler architecture is directly borrowed from vision-language models (Flamingo/BLIP-2) with minimal modification.
- No comparison to ICAE or Gist tokens even in the same-model setting.
- 3B threshold feels like a restatement of Lester et al. (2021) findings in a cross-model context.

**Detailed Critique**

**5.1 Novelty Assessment**  
The cross-model aspect is novel relative to Gist/ICAE/500xCompressor (all single-model). The use of PerceiverResampler is not particularly novel—Q-Former in BLIP-2 uses nearly identical query cross-attention. The main novelty is making it work reliably across model families. The 3B threshold is incremental over Lester et al.’s ~1B finding for single-model prompt tuning.

**5.2 Experimental Evaluation**  
Baselines lack recent single-model compressors. Benchmarks are appropriate for prompt compression but too simple to stress semantic preservation.

**5.3 Ablation Studies**  
Good ablations on training tricks, but missing: different query counts M, different source layers, discrete vs continuous.

**5.4 Technical Soundness**  
Training procedure is now reproducible and well-justified. Choice of continuous over discrete is convincingly argued.

**5.5 Writing Quality**  
Related work section fairly cites single-model compressors but could better acknowledge PerceiverResampler origins.

**Questions for Authors**

1. Why no same-model baseline using Gist tokens or ICAE on Llama-8B?
2. Have you tried deeper prompt tuning (P-Tuning v2) as a stronger baseline?
3. What happens with M=32 or M=4 on more complex tasks?

**Missing References**

- BLIP-2 (Li et al., 2023) and Flamingo (2022) for PerceiverResampler/Q-Former.
- Su et al. (2022) “On Transferability of Prompt Tuning” (already cited but needs more discussion).

**Suggestions for Improvement**

- Add same-model compression baselines (Gist, ICAE).
- Test on a reasoning task (e.g., GSM8K few-shot).

### Reviewer 3: Multi-Agent Systems Researcher

**Overall Assessment**  
Score: 5/10  
Confidence: 4/5  
Recommendation: Borderline

**Summary**  
The paper introduces a latent communication protocol for heterogeneous LLM agents using soft tokens, promising efficient multi-turn collaboration without growing text payloads.

**Strengths**

- Addresses a real pain point in current multi-agent frameworks (text serialization overhead).
- Constant payload size is ideal for long-horizon agent interactions.
- Bidirectional transfer is a step toward general multi-model collaboration.
- Practical training recipe provided.
- Capacity threshold finding has implications for agent system design.

**Weaknesses**

- No actual multi-agent experiment—only single-transfer classification.
- Only one cross-model pair thoroughly tested; Qwen→Mistral mentioned as failing in smoke runs.
- Training cost per model pair is non-trivial (~537M bridge).
- No demonstration in a real agent loop (e.g., debate, tool use).
- Asymmetry (Mistral→Llama better) is reported but not analyzed.

**Detailed Critique**

**5.1 Novelty Assessment**  
Distinct from C2C (KV-level) and LatentMAS (same-family). Meaningful for agent systems, but practical utility unproven.

**5.2 Experimental Evaluation**  
Classification tasks do not stress multi-turn reasoning or tool use typical in agents. Missing: any agent benchmark (e.g., GAIA, AgentBench).

**5.3 Ablation Studies**  
Missing multi-turn ablation.

**5.4 Technical Soundness**  
Sound, but scalability to >2 agents unclear.

**5.5 Writing Quality**  
Motivation for multi-agent use is strong in intro but never validated experimentally.

**Questions for Authors**

1. Why no multi-turn agent experiment?
2. How would this scale to 3+ agents with different architectures?
3. What was the training cost (GPU-hours) for the bridge?

**Missing References**

- Multi-agent frameworks using text protocols (AutoGen, MetaGPT).

**Suggestions for Improvement**

- Add a simple multi-turn agent task (e.g., collaborative QA).

### Reviewer 4: Representation Learning Theorist

**Overall Assessment**  
Score: 7/10  
Confidence: 4/5  
Recommendation: Weak Accept

**Summary**  
The paper provides empirical evidence that sufficiently large LLMs can condition on shared continuous embeddings across families, supporting aspects of the Platonic Representation Hypothesis.

**Strengths**

- Clean demonstration of cross-family latent alignment.
- Sharp capacity threshold is interesting theoretically.
- Bidirectional transfer with asymmetry is intriguing.
- Careful rejection of discrete bottlenecks.
- Connects to representation convergence literature.

**Weaknesses**

- Theoretical grounding is light—mostly citations to Platonic hypothesis without new analysis.
- No probing of latent space (e.g., linear CCA across models).
- Asymmetry on SST-2 (97% vs 91.5%) reported but not explained.
- No comparison to relative representations (Moschella et al.).
- Capacity threshold explanation (working memory hypothesis) is post-hoc.

**Detailed Critique**

**5.1 Novelty Assessment**  
Embedding-level transfer is distinct from C2C. Threshold is new in cross-model generation context.

**5.2 Experimental Evaluation**  
Benchmarks fine for transfer but not for probing representations.

**5.3 Ablation Studies**  
Missing representation probes.

**5.4 Technical Soundness**  
Correct.

**5.5 Writing Quality**  
Theoretical connections are mentioned but shallow.

**Questions for Authors**

1. Can you compute linear probe accuracy from shared latents to task labels?
2. Why is Mistral→Llama stronger?

**Missing References**

- More on representation convergence (Huh et al. deeper analysis).

**Suggestions for Improvement**

- Add latent space probes (CCA, centering).

### Reviewer 5: Skeptical Senior Reviewer

**Overall Assessment**  
Score: 4/10  
Confidence: 5/5  
Recommendation: Reject

**Summary**  
The paper claims a new interlingua for heterogeneous LLMs using soft tokens, with large speedups and strong accuracy.

**Strengths**

- Engineering effort to stabilize training is commendable.
- Capacity threshold finding is solid.
- Bidirectional transfer works.
- Clear writing in methods.
- Identifies real problems in multi-LLM systems.

**Weaknesses**

- Novelty marginal—essentially PerceiverResampler from VLMs applied to LLM transfer.
- 27× speedup claim is misleading (theoretical, ignores encoder).
- Experiments limited to simple classification; no reasoning tasks.
- Only one main model pair; asymmetry not explained.
- Overstates implications for multi-agent systems without any multi-turn demo.

**Detailed Critique**

**5.1 Novelty Assessment**  
Not sufficiently distinct from C2C (different level but same goal). PerceiverResampler is off-the-shelf.

**5.2 Experimental Evaluation**  
Benchmarks too easy; missing reasoning tasks used in LatentMAS/C2C.

**5.3 Ablation Studies**  
Insufficient for key claims.

**5.4 Technical Soundness**  
Some claims overstated.

**5.5 Writing Quality**  
Abstract overclaims.

**Questions for Authors**

1. Why no reasoning benchmarks?
2. Actual measured speedup?

**Missing References**

- Cross-LoRA, LLM Modules.

**Suggestions for Improvement**  
Major new experiments needed.

### Reviewer 6: Reproducibility & Methodology Expert

**Overall Assessment**  
Score: 6/10  
Confidence: 4/5  
Recommendation: Borderline

**Summary**  
The paper presents a reproducible training recipe for cross-model soft-token transfer.

**Strengths**

- Excellent documentation of training bugs and fixes.
- Hyperparameters and seeds reported.
- Multiple tasks and directions.
- Control experiments for capacity threshold.
- Code likely releasable (training details clear).

**Weaknesses**

- Only 3 seeds for main results.
- Small evaluation sets (200 samples).
- No public code yet.
- Hyperparameter sensitivity not ablated.
- No confidence intervals reported.

**Detailed Critique**

**5.1 Novelty Assessment**  
N/A

**5.2 Experimental Evaluation**  
Small eval sets; need full test sets.

**5.3 Ablation Studies**  
Good on training tricks, missing on architecture choices.

**5.4 Technical Soundness**  
Reproducible now.

**5.5 Writing Quality**  
Good.

**Questions for Authors**

1. Will code and checkpoints be released?
2. Full test set results?

**Missing References**  
None major.

**Suggestions for Improvement**

- Report ±std over seeds.
- Release code.

### Reviewer 7: NLP Applications Researcher

**Overall Assessment**  
Score: 5/10  
Confidence: 4/5  
Recommendation: Borderline

**Summary**  
Evaluates on standard classification benchmarks.

**Strengths**

- Strong accuracy on SST-2, AG News, TREC.
- Beats prompt-tuning on harder tasks.

**Weaknesses**

- Benchmarks are outdated and too simple for 2025 LLMs (zero-shot baselines already near ceiling).
- No modern NLP tasks (e.g., MMLU, BIG-Bench).
- No long-context or retrieval tasks.

**Detailed Critique**

**5.1 Novelty Assessment**  
Fine.

**5.2 Experimental Evaluation**  
Benchmarks inappropriate for strong claims about semantic preservation.

**5.3 Ablation Studies**  
OK.

**5.4 Technical Soundness**  
OK.

**5.5 Writing Quality**  
OK.

**Questions for Authors**

1. Why these old benchmarks?

**Missing References**  
None.

**Suggestions for Improvement**

- Add modern reasoning/QA benchmarks.

### Reviewer 8: Efficiency/Compression Specialist

**Overall Assessment**  
Score: 6/10  
Confidence: 4/5  
Recommendation: Borderline

**Summary**  
Achieves 15–30× token reduction with minor quality loss.

**Strengths**

- Better quality than high-compression single-model methods.
- Constant size payload.

**Weaknesses**

- Lower compression than 500xCompressor.
- Encoder cost ignored.
- No Pareto curve vs other methods.

**Detailed Critique**

**5.1 Novelty Assessment**  
Distinct channel.

**5.2 Experimental Evaluation**  
Missing compression baselines.

**5.3 Ablation Studies**  
Missing M ablation.

**5.4 Technical Soundness**  
OK.

**5.5 Writing Quality**  
OK.

**Questions for Authors**

1. Pareto front vs M?

**Missing References**  
None.

**Suggestions for Improvement**

- Add compression-quality curve.

### Reviewer 9: Writing & Presentation Critic

**Overall Assessment**  
Score: 7/10  
Confidence: 5/5  
Recommendation: Weak Accept

**Summary**  
Well-structured paper with clear contribution.

**Strengths**

- Logical flow.
- Detailed method section.
- Good tables.

**Weaknesses**

- Abstract overclaims 27× speedup.
- Some sentences repetitive (e.g., prefill bottleneck).
- Figures missing (no latency plots).
- Notation minor inconsistencies (dz vs d_z).

**Detailed Critique**

**5.1–5.4**  
N/A

**5.5 Writing Quality**  
Sentence-level: Page 1 “pay the full prefill cost repeatedly—a process that is both slow and lossy.” → good, but repeated. Page 4 calibration equation clear.

**Questions for Authors**  
None.

**Missing References**  
None.

**Suggestions for Improvement**

- Tone down speedup claims.
- Add figures for training curves, latency.

### Reviewer 10: Industry Practitioner

**Overall Assessment**  
Score: 5/10  
Confidence: 3/5  
Recommendation: Borderline

**Summary**  
Proposes efficient communication for multi-LLM systems.

**Strengths**

- Addresses real deployment pain.
- Constant overhead attractive.

**Weaknesses**

- Training 537M bridge per pair is expensive.
- No deployment numbers.
- Limited model pairs.
- No black-box API compatibility.

**Detailed Critique**

**5.1 Novelty Assessment**  
Useful if scaled.

**5.2 Experimental Evaluation**  
Not production-relevant tasks.

**5.3 Ablation Studies**  
OK.

**5.4 Technical Soundness**  
OK.

**5.5 Writing Quality**  
OK.

**Questions for Authors**

1. Cost to train bridge?
2. Inference latency on production hardware?

**Missing References**  
None.

**Suggestions for Improvement**

- Add deployment cost analysis.

## Meta-Review

**Consensus Summary**  
Reviewers agree the core idea—embedding-level latent communication—is technically sound and addresses a real problem. The training stabilizations and capacity threshold are solid contributions. Bidirectional transfer is impressive.

**Key Controversies**

- Magnitude and validity of 27× speedup (theoretical vs measured).
- Degree of novelty (distinct from C2C vs incremental).
- Sufficiency of classification benchmarks vs need for reasoning/multi-turn tasks.

**Critical Issues**

1. Provide measured end-to-end latency (including encoder).
2. Add at least one reasoning or multi-turn experiment.
3. Tone down speedup claims and clarify measurement methodology.
4. Better compare/differentiate from C2C and PerceiverResampler precedents.

**Venue Recommendation**  
MLSys is reasonable for the systems angle (prefill bottleneck, constant overhead), but the contribution is currently more ML than systems. If latency measurements and multi-turn demos are added, MLSys fits; otherwise ICLR or NeurIPS may be better.

**Overall Verdict**  
Borderline paper with clear potential. Major revision needed to strengthen experimental validation and claims.

**Revision Roadmap**  
**Critical**

- End-to-end latency measurements including bridge cost.
- Clarify/abstract qualify 27× claim.
- Add reasoning benchmark (e.g., GSM8K) or multi-turn demo.

**Major**

- Multi-turn experiment.
- Stronger baselines (Gist/ICAE same-model, C2C if possible).
- Ablate M and bridge cost.

**Minor**

- Release code.
- Add figures/tables for training stability and latency.
- More representation probes.
