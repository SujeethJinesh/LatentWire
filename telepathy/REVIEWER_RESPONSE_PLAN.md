# Reviewer Response Plan: Addressing Committee Concerns

**Date**: 2024-12-14
**Status**: PLANNED
**Goal**: Strengthen paper by addressing top critiques from simulated review committee

---

## Executive Summary

We simulated reviews from 10 professors (Percy Liang, Sara Hooker, Tri Dao, Chris Ré, Yann LeCun, Jason Wei, Chelsea Finn, Denny Zhou, Colin Raffel, Ludwig Schmidt). This document outlines experiments to address their concerns.

---

## Critique 1: Only Classification, No Generation/Reasoning
**Severity**: HIGH | **Reviewers**: 6/10 (LeCun, Zhou, Wei, Liang, Ré, Hooker)

### The Concern
> "The claim 'cross-model communication' is overstated when you only show classification. Real multi-agent systems need generation."

### Our Response: GSM8K Math Reasoning Experiment

**Why GSM8K?**
- Tests if bridge can transfer reasoning signals, not just classification
- Well-established benchmark with clear metrics (exact match accuracy)
- Chain-of-thought is the standard approach - we can compare latency

**Experiment Design:**
1. **Bridge approach**: Llama encodes problem → Bridge → Mistral generates answer
2. **Training objective**: Next-token prediction on solution steps
3. **Evaluation**: Exact match on final numerical answer

**Expected Outcomes:**
- If works: Major strengthening - proves method extends beyond classification
- If fails partially: Still valuable - shows where method struggles
- If fails completely: Honest limitation to acknowledge

**Compute**: ~2-4 hours on H100

---

## Critique 2: Missing Fine-tuning (LoRA) Comparison
**Severity**: HIGH | **Reviewers**: 4/10 (Liang, Ré, Raffel, Finn)

### The Concern
> "What if you just fine-tuned Mistral on the same data? The 188K bridge parameters might be accomplishing the same thing as a small LoRA adapter."

### Our Response: LoRA Baseline on Mistral

**Why this matters:**
- If LoRA achieves similar accuracy, the bridge's value is just latency (still valid!)
- If bridge beats LoRA, proves cross-model communication adds real value
- Either way, this is a fair comparison reviewers will ask for

**Experiment Design:**
1. Train LoRA (rank 8, ~200K params to match bridge) on Mistral
2. Same training data, same training budget as bridge
3. Compare accuracy AND latency

**Key insight**: Even if LoRA matches accuracy, it still requires:
- The sender model's INPUT (we don't need sender at inference)
- Wait, actually we DO need sender for bridge too...
- The real comparison: LoRA on Mistral alone vs Bridge (Llama→Mistral)

**Expected Outcomes:**
- Bridge beats LoRA: Cross-model transfer adds value
- LoRA matches bridge: Paper still valid (sender hidden states = valuable signal source)
- LoRA beats bridge: Need to investigate why

**Compute**: ~1 hour per dataset on H100

---

## Critique 3: Weak Individual Model Baselines (Zero-shot Only)
**Severity**: HIGH | **Reviewers**: 4/10 (Wei, Zhou, Ré, Liang)

### The Concern
> "The 'super-additive' claim falls apart if few-shot baselines close the gap. TREC 53.5% for Llama is suspiciously bad."

### Our Response: Few-shot Baselines (5-shot)

**Why this matters:**
- Zero-shot baselines may be artificially weak
- Few-shot is standard practice and fair comparison
- If few-shot closes gap, "super-additive" claim needs revision
- If bridge still wins, claim is strengthened

**Experiment Design:**
1. 5-shot prompting for Llama and Mistral on all 4 datasets
2. Randomly sample 5 examples per class from training set
3. Report mean over 3 different random samples

**Critical for TREC:**
- Zero-shot Llama: 53.5% (suspicious)
- If 5-shot Llama: 85%+, our narrative changes
- Need to be prepared either way

**Expected Outcomes:**
- Few-shot << Bridge: Super-additive claim validated
- Few-shot ≈ Bridge: Need to reframe (bridge = implicit few-shot via training?)
- Few-shot > Bridge: Serious concern, but unlikely

**Compute**: ~30 min (inference only)

---

## Critique 4: Text-Relay Baseline Artificially Weak
**Severity**: MEDIUM | **Reviewers**: 3/10 (Hooker, Ré, Wei)

### The Concern
> "You're comparing to summarization, but what about chain-of-thought where Llama explains its reasoning?"

### Our Response: Chain-of-Thought Text-Relay

**Why this matters:**
- Summarization is indeed a weak baseline for communication
- CoT is how real multi-agent systems communicate
- Fair comparison: information-rich text vs soft tokens

**Experiment Design:**
1. Llama generates detailed reasoning/explanation (not summary)
2. Mistral receives full CoT and makes prediction
3. Measure both accuracy AND latency

**Prompt for CoT:**
```
Analyze this text step by step, then provide your classification:
[text]
Think through: What is the sentiment/topic/question type?
```

**Expected Outcomes:**
- CoT accuracy < Bridge: Bridge wins on both speed AND accuracy
- CoT accuracy ≈ Bridge: Bridge wins on speed (22×), CoT has interpretability
- CoT accuracy > Bridge: Concerning, but unlikely given latency penalty

**Compute**: ~1 hour (generation is slow)

---

## Critique 5: No Cross-Task Transfer (Zero-shot Bridge)
**Severity**: MEDIUM | **Reviewers**: 2/10 (Finn, Raffel)

### The Concern
> "Can a bridge trained on SST-2 work on AG News? If not, this is just task-specific adapter training."

### Our Response: Zero-shot Bridge Transfer

**Why this matters:**
- Tests if bridge learns general representations or task-specific patterns
- If it transfers: Major finding about representation universality
- If it fails: Honest limitation (we already acknowledge task-specific training)

**Experiment Design:**
1. Take SST-2 trained bridge (no modification)
2. Evaluate on AG News, TREC (zero-shot, no training)
3. Compare to random chance

**Expected Outcomes:**
- Transfer works (>random): Add as finding about representation learning
- Transfer fails: Confirm task-specific limitation, note in paper

**Compute**: ~10 min (inference only)

---

## Critique 6: Missing Batched Latency/Throughput
**Severity**: MEDIUM | **Reviewers**: 2/10 (Dao, Ré)

### The Concern
> "Latency comparison is single-sample. What about batched inference? Throughput matters for production."

### Our Response: Batched Latency Benchmark

**Why this matters:**
- Single-sample latency doesn't tell the full story
- Production systems batch requests
- Need to show the method scales

**Experiment Design:**
1. Measure latency at batch sizes: 1, 2, 4, 8, 16, 32
2. Calculate throughput (samples/second)
3. Compare Bridge vs Text-Relay vs Direct

**Expected Outcomes:**
- Bridge maintains speedup advantage at all batch sizes: Great
- Speedup decreases with batch: Still faster, note the trend

**Compute**: ~30 min

---

## Implementation Plan

### Phase 1: Quick Wins (Day 1)
| Experiment | Script | Est. Time | Priority |
|------------|--------|-----------|----------|
| Few-shot baselines | `run_fewshot_baselines.py` | 30 min | HIGH |
| Cross-task transfer | `run_transfer_test.py` | 10 min | MEDIUM |
| Batched latency | `run_batched_latency.py` | 30 min | MEDIUM |

### Phase 2: Moderate Effort (Day 1-2)
| Experiment | Script | Est. Time | Priority |
|------------|--------|-----------|----------|
| LoRA comparison | `run_lora_baseline.py` | 2 hours | HIGH |
| CoT text-relay | `run_cot_relay.py` | 1 hour | MEDIUM |

### Phase 3: Major Experiment (Day 2-3)
| Experiment | Script | Est. Time | Priority |
|------------|--------|-----------|----------|
| GSM8K reasoning | `train_gsm8k_bridge.py` | 4 hours | HIGH |

---

## Success Criteria

### Must Achieve (for paper to be strengthened):
- [ ] Few-shot baselines don't completely close the gap
- [ ] Bridge competitive with or beats LoRA
- [ ] Latency advantage holds with batching

### Nice to Have:
- [ ] GSM8K shows some positive signal
- [ ] CoT relay slower but comparable accuracy
- [ ] Cross-task transfer shows partial success

### Prepared to Handle:
- [ ] Few-shot closes gap → Reframe super-additivity as "implicit few-shot via training"
- [ ] LoRA beats bridge → Emphasize latency advantage as main contribution
- [ ] GSM8K fails → Acknowledge classification focus in limitations

---

## Paper Updates After Experiments

### If experiments go well:
1. Add "Reasoning Tasks" subsection with GSM8K results
2. Add Table: Bridge vs LoRA vs Few-shot comparison
3. Update latency table with batched results
4. Add CoT-relay to text-relay comparison
5. Add transfer learning analysis to appendix

### If experiments reveal weaknesses:
1. Strengthen limitations section with honest assessment
2. Adjust "super-additive" claims if few-shot is competitive
3. Position paper as "latency-optimized cross-model communication for classification"

---

## Files to Create

```
telepathy/
├── run_reviewer_experiments.sh      # Master script
├── eval_fewshot_baselines.py        # Few-shot evaluation
├── train_lora_baseline.py           # LoRA training
├── eval_cot_relay.py                # Chain-of-thought relay
├── eval_transfer.py                 # Cross-task transfer
├── benchmark_batched_latency.py     # Batched latency
└── train_gsm8k_bridge.py            # GSM8K reasoning
```
