# Telepathy Paper Review Notes

This document tracks issues, concerns, and improvement ideas for the paper.

---

## Open Issues

### 1. Text-Relay Baseline May Be a Strawman

**Status**: Unresolved

**Problem**: The text-relay baseline performs suspiciously poorly:
- SST-2: 41.3% (below random 50%)
- AG News: 1.0% (random is 25%)
- TREC: 4.0% (random is 16.7%)

**How it works**:
1. Llama generates a summary: `"Summarize this review while preserving its sentiment: {text}"`
2. Mistral classifies from summary with awkward prompt: `"Sentiment:\n\nSummary: {summary}\n\nClassify as one of [...]. Answer:"`

**Concerns**:
- The classification prompt structure is confusing for Mistral
- No practitioner would design a cross-model system this way
- Results being worse than random suggests implementation issues, not just information loss

**More fair alternatives would be**:
1. Direct text pass-through (Llama receives → passes full text → Mistral classifies)
2. Llama-only classification (baseline without Mistral)
3. Chain-of-thought relay (Llama reasons → passes reasoning to Mistral)

**Recommendation**: Either improve text-relay to be fairer, or acknowledge it represents a naive/worst-case approach. The zero-shot Mistral comparison (89.5% SST-2) is more meaningful.

---

### 2. Linear Probe Uses Different Layer Than Bridge

**Status**: Unresolved

**Problem**: The linear probe baseline uses a different layer than the bridge:
- Linear probe: Layer 16
- Bridge: Layer 31

**Results comparison**:
| Dataset | Linear Probe (L16) | Bridge (L31) |
|---------|-------------------|--------------|
| SST-2 | 92.1% | 93.7% |
| AG News | 86.8% | 90.7% |
| TREC | **95.0%** | 87.9% |

**Concerns**:
1. **Layer mismatch undermines "upper bound" argument** - The paper claims linear probe shows information IS extractable, but it's measuring different representations
2. **Unexplained 7pp gap on TREC** - If layer 16 has 95% accuracy with a linear classifier, why does the bridge (with Perceiver + cross-attention) only get 87.9% from layer 31?
3. **Apples-to-oranges** - Linear probe is single-model (Llama→classifier), bridge is cross-model (Llama→Mistral)
4. **No cross-model alignment challenge** - The linear probe doesn't have to deal with cross-model vocabulary/representation alignment, which is a core difficulty the bridge faces

**Verdict**: Linear probe is appropriate as a concept but problematic in execution.

**Recommendation**:
- Run linear probe on layer 31 (same as bridge) for fair comparison
- Add "Mistral linear probe" baseline to isolate cross-model transfer difficulty
- Explain the TREC gap in the paper

---

### 3. Related Work Section is Outdated

**Status**: Unresolved

**Problem**: The Related Work section primarily cites papers from 2021-2023. Most foundational citations are old:
- Vaswani 2017 (Transformers)
- Lester 2021 (Prompt Tuning)
- Li 2021 (Prefix Tuning)
- Jaegle 2021 (Perceiver)
- Alayrac 2022 (Flamingo)
- Bansal 2021 (Model Stitching)

**The bib file HAS newer papers that aren't cited in Related Work**:
- `promptbridge2025` - Prompt Bridge (2025)
- `stitchllm2025` - Stitch LLMs (ACL 2025)
- `modelstitching2025` - Model Stitching analysis (2025)
- `xu2024soft` - Soft token methods (ICML 2024)
- `hiddentransfer2024` - Hidden transfer for acceleration (2024)
- `hao2024coconut` - Coconut continuous thought (2024)

**Missing recent work (2024-2025) that should be discussed**:
- Cross-model communication methods
- Latent space transfer between LLMs
- Soft prompt/token sharing across models
- Multi-agent LLM systems with latent communication
- Speculative decoding and draft model approaches
- Model merging and stitching for LLMs specifically

**Why this matters**:
- Reviewers will immediately notice outdated related work
- Makes paper look like it ignores recent developments
- Misses opportunity to position contribution relative to latest work
- 2024-2025 has seen explosion of work in this space

**Recommendation**:
1. Do a thorough literature search for 2024-2025 papers on:
   - "cross-model communication LLM"
   - "soft token transfer"
   - "latent space LLM collaboration"
   - "model stitching language models"
2. Add a paragraph on recent concurrent work
3. Differentiate from newer methods explicitly

---

### 4. Bridge Training Details Need Clarification

**Status**: Unresolved

**Problem**: Key questions about bridge training are not clearly answered:

**Q1: Is training task-specific or general?**
- Paper says (line 465): "Bridges must be trained per-task. We did not observe meaningful zero-shot transfer between tasks (e.g., SST-2→AG News)"
- But this is buried in Limitations section, not prominently stated
- Readers need to know upfront: one bridge per task

**Q2: How much compute relative to LLM training?**
- Paper says: "3.5-5 minutes per bridge, ~1 GPU-hour total"
- But doesn't contextualize: LLM training takes **thousands of GPU-hours**
- Bridge training is ~0.001% of LLM training compute - this is a major selling point!

**Q3: Do bridges generalize within similar tasks?**
- Paper only tested SST-2→AG News (both classification, different domains) - no transfer
- But what about:
  - SST-2 → other sentiment datasets (Yelp, IMDB)?
  - AG News → other news classification?
  - Same domain, different distribution?
- This is important for practical deployment

**Q4: Training data requirements?**
- How many examples needed? (Currently uses 2000 steps with batch size 8 = 16K examples seen)
- Is this from the task's training set?
- Can it work with fewer examples?

**What the paper currently says**:
- Training time: 3.5-5 minutes per bridge (lines 554-556)
- Total: ~42 minutes for all experiments (line 558)
- Steps: 2000-3000 (line 222)
- Task-specific: yes, no cross-task transfer observed (line 465)

**What's missing**:
1. Prominent statement that bridges are task-specific (not hidden in Limitations)
2. Compute comparison to LLM training (huge efficiency win!)
3. Within-domain generalization experiments
4. Data efficiency analysis (how few examples can work?)

**Recommendation**:
1. Add a "Bridge Training" subsection in Methods that clearly states:
   - Bridges are trained per-task
   - Training takes minutes, not hours (vs. LLM training)
   - Uses standard task training data
2. Add experiment: train on SST-2, test on IMDB/Yelp (same task type)
3. Add data efficiency curve: accuracy vs. training examples
4. Emphasize the compute efficiency as a contribution

---

### 5. Paper Doesn't Explain How Bridge Solves Identified Challenges

**Status**: Unresolved (Critical)

**Problem**: The paper identifies four specific architectural mismatches between Llama and Mistral (lines 128-131):

1. **Vocabulary**: Llama (128K tokens) vs. Mistral (32K tokens)
2. **Positional encoding**: Different RoPE base frequencies
3. **Attention**: Grouped-query (Llama) vs. sliding window (Mistral)
4. **Statistics**: Hidden state magnitude differs by ~5×

Then says "A naive linear projection fails because it assumes isomorphic spaces. The bridge must learn a semantic translation."

**But the Bridge Architecture section (lines 143-163) doesn't explain how each challenge is addressed:**

| Challenge | How Bridge Addresses It | Explained in Paper? |
|-----------|------------------------|---------------------|
| Vocabulary mismatch | ??? | NO |
| RoPE differences | ??? | NO |
| Attention differences | ??? | NO |
| Magnitude differences | RMS normalization + α calibration | Partially (line 158-162) |

**What the paper actually says about the bridge:**
1. Input Projection: Linear projection to bridge dim
2. Learned Latent Queries: M learnable vectors
3. Cross-Attention Layers: N transformer blocks
4. Output Projection: Linear + RMS normalization with calibrated α

**The disconnect:**
- The paper raises specific technical challenges but then describes a generic Perceiver architecture
- Only the magnitude difference (#4) is explicitly addressed via RMS normalization
- The other three challenges are implicitly handled by "learning a semantic translation" but this is hand-wavy

**How the bridge ACTUALLY addresses these (what should be explained):**

1. **Vocabulary mismatch**: The bridge operates in hidden state space, NOT token space. Soft tokens are continuous embeddings that bypass vocabulary entirely. The receiver interprets them as "pseudo-embeddings" without tokenization.

2. **RoPE differences**: The bridge extracts hidden states AFTER positional encoding is applied in the sender. The soft tokens are position-agnostic when fed to the receiver (they're prepended, not interleaved). The receiver applies its own positional encoding to subsequent text tokens.

3. **Attention differences**: The cross-attention in the bridge learns to extract information regardless of how the sender computed it. The receiver then processes soft tokens with its own attention mechanism - the soft tokens just need to be in a compatible embedding space.

4. **Magnitude differences**: RMS normalization + learned scale α calibrates soft tokens to match receiver's expected embedding statistics.

**Recommendation**:
Add a paragraph after the architecture description explicitly connecting each challenge to its solution:

```latex
\paragraph{Addressing Representation Mismatch}
The bridge architecture specifically addresses each identified challenge:
(1) Vocabulary differences are bypassed entirely---soft tokens are continuous
embeddings that never pass through either tokenizer.
(2) Positional encoding differences are handled by extracting post-RoPE hidden
states from the sender; soft tokens are prepended to the receiver's input
without positional encoding, allowing the receiver to apply its own scheme.
(3) Attention mechanism differences are abstracted away by the cross-attention
layers, which learn to extract task-relevant information regardless of how
the sender computed it.
(4) Magnitude differences are explicitly calibrated via RMS normalization
with a learned scale factor α.
```

---

## Resolved Issues

### Fixed in commit abfac2f (2025-01-12)

1. **Internal dimension inconsistency**: Paper said d=512, code uses d=4096 → Fixed to d=d_R
2. **Learning rate mismatch**: Paper said 1e-4, code uses 2e-4 → Fixed
3. **Soft token size calculation**: Paper said ~16KB, actual is 256KB → Fixed with formula
4. **Missing citations**: Added BoolQ, PIQA, CommonsenseQA, LoRA references
5. **Unreferenced table**: Added ref to tab:size_threshold
6. **Bimodal TREC reporting**: Changed misleading mean±std to "84/38" format
7. **Informal language**: "Interestingly" → "Notably", removed "actually"

---

---

### 6. Method Novelty Concerns (Critical)

**Status**: Unresolved

**Problem**: The core technical contribution is applying existing Perceiver Resampler architecture (Flamingo 2022, BLIP-2 2023) to LLM-to-LLM communication.

**Specific concerns**:
1. Paper admits architecture is "based on Perceiver Resampler" (line 75) - not architecturally novel
2. The differentiation from model stitching (line 93-94) is weak - the bridge IS a stitching layer
3. "Soft token communication" is essentially the same as ICAE, Gisting, AutoCompressor approaches

**Missing comparisons**:
- No comparison to simple linear stitching layer (Bansal et al. 2021)
- No comparison to ICAE for cross-model compression
- Linear probe achieves 91.5% vs Perceiver's 92.0% on SST-2 - is complexity justified?

**Recommendation**: Either add genuine technical novelty OR reframe as systems/empirical contribution.

---

### 7. Missing Critical Baselines

**Status**: Unresolved

**Problem**: Several obvious baselines are missing:

1. **Few-shot baseline (5-shot)**: Mentioned in paper but NOT in results tables
2. **Simple ensemble**: Average Llama + Mistral logits - tests "super-additive" claim
3. **LLMLingua prompt compression**: SOTA compression, direct competitor
4. **Same-model bridge (Llama→Llama)**: Control for "cross-model" claim
5. **Small fine-tuned models**: BERT-base achieves 94%+ on SST-2 with 110M params (vs 15B)

**Why ensemble is critical**: Paper claims "super-additive performance" but never compares to simply averaging both models' predictions.

---

### 8. Datasets Are Outdated

**Status**: Unresolved

**Problem**: Using datasets from 1999-2015:
- SST-2 (2013) - Movie reviews
- AG News (2015) - News classification
- TREC (1999) - Question classification

**Modern alternatives (2020-2024)**:
- TweetEval Sentiment (2020) - Social media, 3-class
- Financial PhraseBank (2020) - Domain-specific sentiment
- CLINC150 (2019) - 150-class intent classification
- GoEmotions (Google 2020) - 27 fine-grained emotions
- ToxiGen (2022) - Safety-critical classification

**Why this matters**: Reviewers expect contemporary benchmarks; old datasets suggest cherry-picking.

---

### 9. Statistical Significance Missing

**Status**: Unresolved

**Problem**: Paper reports mean±std but no significance tests:
- Is 93.7% vs 92.1% (Bridge vs Linear Probe on SST-2) significant with n=5?
- No p-values, no confidence intervals, no effect sizes (Cohen's d)
- No multiple comparison correction for the many comparisons made

**Infrastructure exists**: `scripts/statistical_testing.py` has full testing suite but results not in paper.

---

### 10. Reasoning Failure Underexplored

**Status**: Unresolved

**Problem**: CommonsenseQA result (17% vs 20% random) is dismissed without analysis.

**Key questions unanswered**:
1. Does the sender even encode reasoning in hidden states? (Run linear probes)
2. Is information lost in compression or cross-model transfer?
3. Why does PIQA work (60.4%) but CommonsenseQA fail (17%)?
4. Could iterative refinement help? (Coconut-style multi-hop)

**Existing diagnostic script** (`analyze_reasoning_failure.py`) tests 5 hypotheses but results not in paper.

---

### 11. Practical Value Questionable

**Status**: Unresolved

**Problem**: Why use 15B parameters (8B+7B) for classification when:
- Fine-tuned BERT-base (110M params) achieves 94%+ on SST-2
- SetFit achieves 92%+ with 8 examples and 110M params
- A single fine-tuned Mistral would likely beat the bridge

**The paper's implicit use case is never articulated**:
- When would you NEED two frozen models?
- What scenario justifies 15B params for classification?

**Potential angles**:
- "Frozen model composition" - can't fine-tune API models
- "Capability preservation" - fine-tuning hurts other abilities
- Neither is demonstrated in the paper.

---

## Prioritized Experiment Recommendations

### HIGH PRIORITY (Must-Run)

1. **Fix Text-Relay Baseline** (~2h)
   - Implement fair direct pass-through baseline
   - Expected: Text-relay matches Mistral zero-shot (~90%)
   - Makes speedup claim valid

2. **Run Layer-31 Linear Probe** (~1h)
   - Fair comparison with bridge
   - Explain TREC gap (95% probe vs 87.9% bridge)

3. **Add Modern Datasets** (~4h)
   - TweetEval, Financial PhraseBank, GoEmotions
   - Shows method works on contemporary data

4. **Simple Reasoning Datasets** (~3h)
   - ARC-Easy (elementary science questions)
   - WinoGrande (common sense)
   - HellaSwag (situational reasoning)
   - Quick to run, properly characterizes reasoning boundary

5. **Ensemble Baseline** (~2h)
   - Average Llama + Mistral logits
   - Tests "super-additive" claim

6. **Statistical Significance** (~2h)
   - Run paired t-tests with correction
   - Add p-values to key comparisons

### MEDIUM PRIORITY (Strengthen Paper)

7. **Same-Model Control (Llama→Llama)** (~3h)
   - Establishes ceiling for cross-model transfer
   - Shows what "translation cost" actually is

8. **LLMLingua Comparison** (~4h)
   - SOTA prompt compression baseline
   - Match token budgets (8, 16, 32)

9. **Data Efficiency Curve** (~6h)
   - Train with 100, 500, 1K, 5K, 16K examples
   - Shows practical deployment requirements

10. **Cross-Task Transfer Test** (~4h)
    - SST-2 bridge → IMDB, Rotten Tomatoes
    - Tests within-domain generalization

### LOWER PRIORITY (Nice to Have)

11. **Soft Token Interpretability** (~3h)
    - Vocabulary projection analysis
    - t-SNE visualization by class

12. **Attention Visualization** (~4h)
    - What does bridge attend to?
    - What does Mistral attend to in soft tokens?

13. **Multi-Task Bridge** (~6h)
    - Single bridge for SST-2 + AG News + TREC
    - Would address "per-task bridge" limitation

14. **TREC Bimodality Analysis** (~5h)
    - Why do some seeds fail (38%) and others succeed (84%)?
    - Intervention experiments to escape bad mode

---

## Citations to Add

### Must-Add (2024-2025 Papers)

```bibtex
@article{hao2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and others},
  journal={arXiv:2412.06769},
  year={2024}
}

@article{jiang2023llmlingua,
  title={LLMLingua: Compressing Prompts for Accelerated Inference},
  author={Jiang, Huiqiang and others},
  journal={arXiv:2310.05736},
  year={2023}
}

@article{ge2024icae,
  title={In-context Autoencoder for Context Compression},
  author={Ge, Tao and others},
  journal={ICLR 2024},
  year={2024}
}

@article{mu2023gisting,
  title={Learning to Compress Prompts with Gist Tokens},
  author={Mu, Jesse and others},
  journal={NeurIPS 2023},
  year={2023}
}

@article{leviathan2023speculative,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and others},
  journal={ICML 2023},
  year={2023}
}

@article{tunstall2022setfit,
  title={Efficient Few-Shot Learning Without Prompts},
  author={Tunstall, Lewis and others},
  journal={arXiv:2209.11055},
  year={2022}
}
```

---

## Key Paper Weaknesses Summary

| Issue | Severity | Fix Difficulty | Impact |
|-------|----------|----------------|--------|
| Text-relay strawman | Critical | Easy (2h) | Makes speedup meaningful |
| No ensemble baseline | Critical | Easy (2h) | Validates super-additive |
| Linear probe layer mismatch | Major | Easy (1h) | Fair comparison |
| No statistical significance | Major | Easy (2h) | Rigorous claims |
| Outdated datasets | Major | Medium (4h) | Contemporary relevance |
| Novelty concerns | Critical | Hard | Core contribution |
| No reasoning analysis | Major | Medium (3h) | Explains limitations |
| Missing few-shot baseline | Major | Easy (1h) | Complete baselines |
| 15B params not justified | Critical | Hard | Practical value |
| Related work outdated | Major | Medium (4h) | Positioning |

---

## Resolved Issues

### Fixed in commit abfac2f (2025-01-12)

1. **Internal dimension inconsistency**: Paper said d=512, code uses d=4096 → Fixed to d=d_R
2. **Learning rate mismatch**: Paper said 1e-4, code uses 2e-4 → Fixed
3. **Soft token size calculation**: Paper said ~16KB, actual is 256KB → Fixed with formula
4. **Missing citations**: Added BoolQ, PIQA, CommonsenseQA, LoRA references
5. **Unreferenced table**: Added ref to tab:size_threshold
6. **Bimodal TREC reporting**: Changed misleading mean±std to "84/38" format
7. **Informal language**: "Interestingly" → "Notably", removed "actually"

---

*Last updated: 2025-01-12*
*Issues: 11 open, 7 resolved*
*Based on: 10 critique subagents + 20 experiment subagents*
