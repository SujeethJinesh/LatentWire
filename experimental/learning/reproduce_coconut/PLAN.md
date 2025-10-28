# COCONUT Reproduction Plan

## Paper Overview

**Title**: "Training Large Language Models to Reason in a Continuous Latent Space" (Dec 2024, Meta AI)
**ArXiv**: https://arxiv.org/html/2412.06769v1
**Key Idea**: Replace discrete chain-of-thought (CoT) reasoning steps with continuous hidden states that are fed back as input embeddings, enabling breadth-first search reasoning patterns.

## Design Choices & Justifications

### Why 2 stages instead of 3-4?

- Paper uses 3-4 stages to **gradually** introduce continuous thoughts (curriculum learning)
- For minimal reproduction: **Stage 0 (CoT baseline) + Stage 1 (first continuous thought)** is sufficient to prove the concept works
- We can add more stages later if Stage 1 shows promise
- Testing hypothesis: "Can continuous thoughts help at all?" doesn't require full curriculum

### Why c=1 instead of c=2?

- Paper says c=3 showed **instability**
- c=2 might be close to instability boundary
- **Start conservative** with c=1 to ensure training stability
- Paper showed "performance steadily improved" from c=0 to c=2, so c=1 should show improvement over c=0
- Can increase to c=2 after c=1 works

### Why GSM8k?

- Standard benchmark for mathematical reasoning (8.5K problems)
- Paper used this dataset extensively
- Simple format: question → reasoning → answer
- Well-supported by HuggingFace

---

## Technical Background from Paper

### Architecture

1. **Special Tokens**: Add `<bot>` (beginning of thought) and `<eot>` (end of thought) to tokenizer
2. **Latent Reasoning Mode**: During generation, feed `hidden_state[t-1]` as `input_embedding[t]` directly (skip language model head)
3. **No transformer changes**: Standard Llama 3.1 8B architecture unchanged

### Training Approach from Paper

1. **Stage 0 (Baseline)**: Fine-tune on GSM8k with standard CoT format
   - Format: `Question: {q}\nAnswer: Let's think step by step. {reasoning} The answer is {answer}`

2. **Stage 1**: Replace first reasoning step with continuous thoughts
   - Format: `Question: {q}\nAnswer: <bot>{c continuous thoughts}<eot> {remaining reasoning} The answer is {answer}`
   - Use `c=2` continuous thoughts (paper's setting for math)

3. **Stage 2+**: Replace first 2+ reasoning steps with 2×c continuous thoughts

### Training Details from Paper

- **Dataset**: GSM8k (7,473 training examples)
- **Hyperparameters**:
  - Learning rate: 1e-4
  - Batch size: 128 (effective)
  - Epochs: 6 (stage 0), 3 (stage 1+)
  - c=2 (continuous thoughts per reasoning step)
- **Loss**: Cross-entropy on answer tokens only (mask question + continuous thoughts)
- **<eot> strategy**: Fixed-length padding (simpler than training classifier)

---

## Implementation Strategy

### Key Principle: Use Latest HuggingFace Best Practices

**IMPORTANT**: Before implementing each milestone, use web search to verify:
- Latest HuggingFace APIs for the task
- Best practices for Llama 3.1 specifically
- Current recommendations for tokenizer modifications
- Proper training techniques (2024-2025 standards)

This ensures we're not using outdated patterns and leverage the latest improvements.

---

## Milestone-Based Implementation Plan

### **Milestone 1: Data Loading & Exploration** (20 min)

**Goal**: Load GSM8k, understand format, verify we can use it

**Tasks**:
1. Web search: Latest way to load GSM8k from HuggingFace
2. Load GSM8k dataset using recommended API
3. Inspect data format (question, answer structure)
4. Print 3 examples to understand CoT format
5. Verify train/test split sizes

**File**: `step1_data.py`

**Command to run**:
```bash
cd experimental/learning/reproduce_coconut
python step1_data.py
```

**Expected output**:
- Dataset statistics
- 3 sample problems with answers
- CoT reasoning format

**Exit criteria**: Can load and parse GSM8k successfully

---

### **Milestone 2: Add Special Tokens to Llama 3.1** (15 min)

**Goal**: Add `<bot>` and `<eot>` tokens without breaking model

**Tasks**:
1. Web search: Latest best practice for adding special tokens to Llama 3.1
2. Load Llama 3.1 8B tokenizer
3. Add special tokens using recommended method
4. Resize model embeddings properly
5. Test tokenization with new tokens
6. Verify model can still generate normal text

**File**: `step2_tokens.py`

**Command to run**:
```bash
python step2_tokens.py
```

**Expected output**:
- Original vocab size
- New vocab size after adding tokens
- Token IDs for `<bot>` and `<eot>`
- Test generation to verify model still works

**Exit criteria**: Model generates coherent text with new tokens in vocabulary

---

### **Milestone 3: Stage 0 - CoT Baseline Training** (30 min + training time)

**Goal**: Fine-tune Llama 3.1 8B on GSM8k with standard CoT (no continuous thoughts yet)

**Tasks**:
1. Web search: Latest HuggingFace Trainer best practices for Llama 3.1
2. Format GSM8k in CoT format: `Question: {q}\nAnswer: Let's think step by step. {reasoning}\nThe answer is {answer}`
3. Use HuggingFace Trainer for simplicity
4. Train on subset (500 examples) for fast iteration
5. Save checkpoint to `runs/stage0/`

**File**: `step3_stage0.py`

**Command to run**:
```bash
python step3_stage0.py --samples 500 --epochs 3 --output_dir runs/stage0
```

**Expected output**:
- Training progress (loss decreasing)
- Checkpoint saved to `runs/stage0/`
- Sample predictions on 5 test examples

**Exit criteria**: Loss decreases, model can answer some GSM8k questions

---

### **Milestone 4: Implement Continuous Thought Mechanism** (45 min)

**Goal**: Create custom forward pass that feeds hidden states as next input embeddings

**Tasks**:
1. Web search: Latest techniques for custom training loops in HuggingFace
2. Implement function to replace first CoT step with `<bot>{hidden states}<eot>`
3. Create custom training loop (can't use Trainer for this)
4. Test forward pass with c=1 continuous thought
5. Verify gradients flow correctly

**File**: `step4_continuous.py`

**Command to run**:
```bash
python step4_continuous.py --test_only
```

**Expected output**:
- Forward pass diagram showing hidden state feedback
- Gradient flow verification
- Loss computation on continuous thoughts (should be masked)

**Exit criteria**: Forward pass works, gradients flow, losses computed correctly

---

### **Milestone 5: Stage 1 - Train with Continuous Thoughts** (30 min + training time)

**Goal**: Train Stage 1 where first reasoning step is replaced with c=1 continuous thoughts

**Tasks**:
1. Format data: replace first CoT step with `<bot>{1 continuous thought}<eot>`
2. Train with custom loop from Milestone 4
3. Train on same 500 examples
4. Save checkpoint to `runs/stage1/`

**File**: `step5_stage1.py`

**Command to run**:
```bash
python step5_stage1.py --samples 500 --epochs 3 --c 1 --output_dir runs/stage1
```

**Expected output**:
- Training progress with continuous thoughts
- Checkpoint saved to `runs/stage1/`
- Sample predictions

**Exit criteria**: Training completes without errors

---

### **Milestone 6: Evaluation & Comparison** (20 min)

**Goal**: Compare Stage 0 (CoT) vs Stage 1 (COCONUT) on test set

**Tasks**:
1. Web search: Latest evaluation best practices for LLMs
2. Evaluate both checkpoints on same 100 test examples
3. Compute accuracy (exact match on final answer)
4. Show example outputs side-by-side
5. Measure reasoning efficiency (tokens used)

**File**: `step6_eval.py`

**Command to run**:
```bash
python step6_eval.py \
  --stage0_ckpt runs/stage0 \
  --stage1_ckpt runs/stage1 \
  --test_samples 100
```

**Expected output**:
```
Stage 0 (CoT):      Accuracy = X.XX%  Avg reasoning tokens = YYY
Stage 1 (COCONUT):  Accuracy = X.XX%  Avg reasoning tokens = ZZZ

Example 1:
Question: ...
Stage 0 answer: ...
Stage 1 answer: ...
Correct answer: ...
```

**Exit criteria**: Can compare both approaches quantitatively

---

## File Structure

```
experimental/learning/reproduce_coconut/
├── PLAN.md                    # This file
├── step1_data.py             # Milestone 1: Data loading
├── step2_tokens.py           # Milestone 2: Add special tokens
├── step3_stage0.py           # Milestone 3: CoT baseline training
├── step4_continuous.py       # Milestone 4: Continuous thought mechanism
├── step5_stage1.py           # Milestone 5: Stage 1 training
├── step6_eval.py             # Milestone 6: Evaluation
└── runs/                     # All checkpoints and logs
    ├── stage0/               # Stage 0 checkpoint + logs
    └── stage1/               # Stage 1 checkpoint + logs
```

---

## Timeline & Resources

**Total estimated time**: ~3 hours coding + ~2 hours training = **5 hours**

**Hardware requirements**:
- MacBook with 64GB RAM (your setup)
- MPS backend for GPU acceleration
- Sufficient disk space for Llama 3.1 8B (~16GB)

---

## Next Steps After Milestone 6

### If COCONUT helps (Stage 1 > Stage 0):
- Increase c from 1 to 2
- Train on full dataset (7.5K examples)
- Add Stage 2 (replace first 2 reasoning steps)
- Implement learned `<eot>` classifier instead of fixed padding

### If COCONUT doesn't help (Stage 1 ≈ Stage 0):
- Debug: Check if hidden states are actually being used
- Try different c values
- Inspect what the continuous thoughts encode
- Compare to paper's results and identify gaps

### If COCONUT hurts (Stage 1 < Stage 0):
- Check for implementation bugs
- Verify loss masking is correct
- Ensure gradients flow properly
- Review paper methodology for missed details

---

## Key Implementation Notes

1. **Simplicity First**: Use HuggingFace APIs whenever possible (Trainer, datasets, etc.)
2. **Test Incrementally**: Each milestone must run successfully before moving to next
3. **Small Batches**: Start with 500 samples, not full dataset
4. **Web Search**: Verify latest techniques before each milestone
5. **No Mock Code**: All code must be functional and tested
6. **Logs Everything**: Save all outputs to runs/ directory with timestamps

---

## References

- Paper: https://arxiv.org/html/2412.06769v1
- GSM8k Dataset: https://huggingface.co/datasets/openai/gsm8k
- Llama 3.1 8B: https://huggingface.co/meta-llama/Llama-3.1-8B
