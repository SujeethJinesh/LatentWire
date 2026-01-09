# Telepathy: Neural Bridges for Cross-Model LLM Communication
## Slide Deck Outline for Hour-Long Presentation

---

## SLIDE 1: Title Slide
**Telepathy: Neural Bridges for Cross-Model LLM Communication**
- Presenter: [Your Name]
- Advisor: [PI Name]
- Date: January 2025
- *Visual: Architecture diagram showing Llama → Bridge → Mistral*

---

## SLIDE 2: The Problem - Tower of Babel for LLMs
**Current State: LLMs Communicate Through Text**
- Text generation is slow (835ms for simple classification)
- Information loss through discretization
- Each model must tokenize/detokenize repeatedly
- *Visual: Diagram showing inefficient text relay between models*
- **Key stat: 92% of multi-agent latency is text generation**

---

## SLIDE 3: Research Question
**Can heterogeneous LLMs communicate directly through learned representations?**
- Hypothesis: Soft tokens can transfer information more efficiently than text
- Challenge: Different architectures (Llama vs Mistral)
- Goal: Faster, more accurate cross-model communication
- *Visual: Direct neural connection vs text bottleneck*

---

## SLIDE 4: Related Work - Standing on Giants' Shoulders
**Three Pillars of Our Approach**
1. **Soft Prompts** (Lester et al., 2021): Continuous prompts in embedding space
2. **Vision-Language Bridges** (BLIP-2, Flamingo): Cross-modal alignment
3. **Model Stitching** (Bansal et al., 2024): Layer-wise connections
- **Our Innovation**: Runtime cross-model bridge for heterogeneous LLMs
- *Visual: Timeline of related work leading to Telepathy*

---

## SLIDE 5: The Architecture - Perceiver Resampler Bridge
**Three-Stage Pipeline**
1. **Sender (Llama 3.1 8B)**: Extract layer-31 hidden states
2. **Bridge**: Perceiver cross-attention → 8-32 soft tokens
3. **Receiver (Mistral 7B)**: Process soft tokens as input_embeds
- *Visual: Detailed architecture diagram from paper*
- **Key insight: 2.3% additional parameters (350M) enable cross-model transfer**

---

## SLIDE 6: The Four Boss Battles
**Architectural Challenges Solved**
1. **Magnitude Mismatch**: RMSNorm(Z) × target_rms
2. **Vocabulary Size**: 128K vs 32K → project to shared space
3. **Position Encoding**: RoPE vs ALiBi → learned alignment
4. **Attention Patterns**: GQA vs MHA → cross-attention bridge
- *Visual: Before/after gradient flow diagrams*

---

## SLIDE 7: Training Objective
**Multi-Component Loss Function**
```
L = L_ce + λ_div × L_diversity + λ_kl × L_kl
```
- Cross-entropy on classification labels
- Diversity regularization (prevent collapse)
- KL divergence from teacher (Mistral with text)
- **1,500 steps, batch size 16, 15 minutes on H100**

---

## SLIDE 8: Main Results - Classification Success
| Dataset | Bridge | Text-Relay | Prompt-Tuning | Speedup |
|---------|--------|------------|---------------|---------|
| AG News | **89.5%** | 70.0% | 82.5% | 22× |
| TREC-6 | **96.0%** | 47.0% | 90.0% | 22× |
| SST-2* | 49.5% | 95.0% | 97.5% | N/A |

*Binary classification failure mode
- *Visual: Bar chart comparing methods*

---

## SLIDE 9: Super-Additive Performance
**The 1+1 > 2 Phenomenon on TREC-6**
- Llama alone: 67.5%
- Mistral alone: 67.5%
- Bridge (Llama→Mistral): **96.0%** (+28.5pp!)
- **Why?** Complementary representations + regularization
- *Visual: Venn diagram showing capability overlap*

---

## SLIDE 10: Inverse Token Scaling
**Fewer Tokens = Better Performance**
- 8 tokens: 21.5% on Banking77
- 16 tokens: 19.0%
- 32 tokens: 15.5%
- 128 tokens: 12.0%
- **Information bottleneck acts as regularization**
- *Visual: Line graph showing inverse scaling*

---

## SLIDE 11: Latency Analysis
**22.4× Speedup Breakdown**
- Text-Relay: 834.5ms total
  - Llama generation: 756ms (91%)
  - Mistral processing: 78.5ms
- Bridge: 37.3ms total
  - Llama encoding: 12.1ms
  - Bridge forward: 8.7ms
  - Mistral processing: 16.5ms
- *Visual: Stacked bar chart of latency components*

---

## SLIDE 12: Failure Analysis - SST-2
**Why Binary Classification Fails**
- Random performance: 49.5% (chance = 50%)
- Hypothesis: Binary signal too weak for cross-model transfer
- Text-relay succeeds: 95% (text preserves nuance)
- **Lesson: Not all tasks suitable for compression**
- *Visual: Confusion matrix showing random predictions*

---

## SLIDE 13: Failure Analysis - Reasoning
**GSM8K Math Reasoning: Complete Failure**
| Method | Accuracy |
|--------|----------|
| Llama (direct) | 72.6% |
| Mistral (direct) | 41.7% |
| Text-Relay | 38.0% |
| **Bridge** | **2.0%** |

- **Cannot preserve multi-step reasoning in 8-32 tokens**
- *Visual: Example showing reasoning chain destruction*

---

## SLIDE 14: The Research Journey
**19 Phases Over 6 Months**
1. **Phase 1-5**: LatentWire attempts (failed)
2. **Phase 6-10**: Cross-model discovery
3. **Phase 11-15**: Architecture battles
4. **Phase 16-19**: Telepathy success
5. **Phase 20+**: Paper evaluation
- *Visual: Timeline with key breakthroughs and failures*

---

## SLIDE 15: Statistical Validation
**Rigorous Testing with Multiple Baselines**
- 51 experiments total (2 seeds × 3 datasets × multiple methods)
- Bootstrap CI (1000 samples)
- McNemar's test for paired predictions
- Bonferroni correction for multiple comparisons
- **All results p < 0.05 after correction**
- *Visual: Statistical significance table*

---

## SLIDE 16: Ablation Studies
| Component | Impact on AG News |
|-----------|-------------------|
| Full Model | 89.5% |
| No diversity loss | 84.2% (-5.3pp) |
| No KL regularization | 81.7% (-7.8pp) |
| No cross-attention | 73.4% (-16.1pp) |
| Random init | 52.1% (-37.4pp) |
- **Every component matters**
- *Visual: Ablation waterfall chart*

---

## SLIDE 17: Why This Matters
**Three Levels of Impact**

**Immediate (Practical)**:
- 22× latency reduction for multi-agent systems
- 4-8× compression for bandwidth-limited scenarios
- $0.02 per 1M tokens vs $0.30 (15× cost reduction)

**Scientific**:
- First neural bridge between heterogeneous LLMs
- Discovery of super-additive phenomenon
- Inverse scaling in representation learning

**Future (Foundational)**:
- Enables new multi-agent architectures
- Foundation for learned protocols
- Step toward neural internet for AI

---

## SLIDE 18: Limitations (Honest Assessment)
**What Doesn't Work**:
1. **Reasoning tasks**: Fundamental architectural limitation
2. **Binary classification**: Signal too weak
3. **Zero-shot transfer**: Requires task-specific training
4. **Scale**: Only tested on 7-8B models

**What We Don't Know**:
- Scaling to 70B+ models
- More than 2-model chains
- Non-English languages
- *Visual: Limitation matrix*

---

## SLIDE 19: Future Directions
**Three Research Threads**

1. **Reasoning Bridge** (High Risk, High Reward)
   - Chain-of-thought compression
   - Iterative refinement
   - 64+ token capacity

2. **Multi-Model Networks** (Natural Extension)
   - 3+ model chains
   - Branching/merging topologies
   - Learned routing

3. **Universal Bridge** (Long Term)
   - Single bridge for all model pairs
   - Meta-learning approach
   - Zero-shot task transfer

---

## SLIDE 20: Conclusions
**What We Achieved**:
✅ 22× faster cross-model communication
✅ 96% accuracy on multi-class classification
✅ Super-additive performance (1+1 > 2)
✅ Inverse token scaling discovery

**What We Learned**:
📚 Classification ≠ Reasoning for compression
📚 Heterogeneity provides beneficial regularization
📚 Less can be more (8 tokens > 128 tokens)

**Next Steps**:
🚀 Submit to MLSys 2025
🚀 Open-source release
🚀 Extend to reasoning tasks

---

## SLIDE 21: Questions?
**Thank you!**

**Resources**:
- Paper: [Link to ArXiv when available]
- Code: github.com/SujeethJinesh/LatentWire
- Contact: [Your email]

*Visual: QR code to repository*

---

## BACKUP SLIDES

---

## BACKUP 1: Detailed Architecture
*Full architecture diagram with dimensions*

---

## BACKUP 2: Training Curves
*Loss curves showing convergence*

---

## BACKUP 3: Hyperparameter Sensitivity
*Grid search results table*

---

## BACKUP 4: Additional Datasets
*Results on IMDB, MNLI, 20NewsGroups*

---

## BACKUP 5: Compute Requirements
- Training: 15 min on single H100
- Inference: 37ms on V100
- Memory: 32GB for both models
- Storage: 350MB for bridge weights

---

## BACKUP 6: Prompt Templates
*Exact prompts used for baselines*

---

## BACKUP 7: Error Analysis
*Detailed failure cases with examples*

---

## BACKUP 8: Mathematical Formulation
*Full equations for loss functions*

---

## BACKUP 9: Related Work Comparison Table
*Detailed comparison with 10+ related papers*

---

## BACKUP 10: Implementation Details
*PyTorch code snippets for key components*