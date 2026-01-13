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
