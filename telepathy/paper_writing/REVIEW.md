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

## Future Considerations

(Add notes here as review continues)

---

*Last updated: 2025-01-12*
*Issues: 4 open, 7 resolved*
