# COCONUT Reproduction Plan

**⚠️ IMPORTANT**: This plan has been updated after analyzing the official COCONUT implementation. See `COCONUT_ANALYSIS.md` for detailed comparison.

## Paper Overview

**Title**: "Training Large Language Models to Reason in a Continuous Latent Space" (Dec 2024, Meta AI)
**ArXiv**: https://arxiv.org/html/2412.06769v1
**Official Repo**: https://github.com/facebookresearch/coconut
**Key Idea**: Replace discrete chain-of-thought (CoT) reasoning steps with continuous hidden states that are fed back as input embeddings, enabling breadth-first search reasoning patterns.

## Critical Updates After Analyzing Official Implementation

### Key Findings:

1. **Data**: Uses **385K synthetic examples** from "Internalize CoT Step by Step" paper, NOT original GSM8k (7.5K)
2. **Format**: Steps are **calculator-style** (`<<600*30/100=180>>`), NOT natural language reasoning
3. **Structure**: `{"question": str, "steps": List[str], "answer": str}` where each step is one calculation
4. **Training format**: `question\nstep1\nstep2\n### answer` (answer prefixed with `###`)
5. **Special tokens**: `<|start-latent|>`, `<|latent|>`, `<|end-latent|>` (not `<bot>`, `<eot>`)
6. **Base model**: GPT-2 (124M params), though Llama3 also tested
7. **Stages**: 4 total (Stage 0: CoT baseline, Stages 1-3: progressive latent replacement)
8. **c_thought**: 2 for math tasks (2 latent tokens per reasoning step)

### Milestone 1 Status:

❌ **Milestone 1 needs major revision**:
- Used wrong data source (original GSM8k vs Internalize CoT)
- Wrong format (natural language vs calculator-style)
- Wrong structure (single paragraph vs list of steps)
- 50x less data (7.5K vs 385K examples)

See `COCONUT_ANALYSIS.md` for full comparison.

---

## Implementation Options

### Option A: Exact Official Reproduction (RECOMMENDED for minimal reproduction)

**Approach**: Follow official COCONUT implementation exactly
- ✅ Base model: GPT-2 (124M params) OR Llama 3.1 8B
- ✅ Data: Internalize CoT dataset (385K synthetic examples with calculator steps)
- ✅ Format: `question\n<<calc1>>\n<<calc2>>\n### answer`
- ✅ Special tokens: `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`
- ✅ Stages: 0 (CoT baseline) + 1 (first step → latent)
- ✅ c_thought: 2 (as in paper)

**Pros**:
- Directly comparable with paper results
- Uses proven data and format
- Lower risk - we know it works

**Cons**:
- Calculator-style less interesting than natural language
- Need to download 385K examples (~100MB)

**Time**: ~5 hours (same as original estimate)

### Option B: Adapted for Natural Language Reasoning

**Approach**: Adapt COCONUT to natural language CoT reasoning
- ⚠️ Base model: Llama 3.1 8B
- ⚠️ Data: Original GSM8k (7.5K) with natural language reasoning
- ⚠️ Format: Parse reasoning into sentences/steps
- ⚠️ Challenge: How to split "Let's think step by step..." into discrete steps?

**Pros**:
- More interesting: natural language reasoning
- Smaller dataset (easier to iterate)
- Could be novel contribution

**Cons**:
- Won't match paper results
- Risky - may not work as well
- Need to figure out step splitting

**Time**: ~8 hours (extra time for figuring out step splitting)

### Option C: Hybrid (Staged Approach)

**Approach**: Start with Option A, then explore Option B
1. Phase 1: Reproduce with calculator-style (validate approach)
2. Phase 2: Adapt to natural language (explore)

**Pros**:
- Best of both worlds
- Validates implementation before exploring

**Cons**:
- Most work (double implementation)

**Time**: ~12 hours total

---

## Recommended Path: Option A with Llama 3.1 8B

**Rationale**:
1. User wants to use Llama 3.1 8B (stated in initial request)
2. But use official data/format for validity
3. Minimal changes from official implementation
4. Can verify approach works before exploring alternatives

**Modifications from official**:
- ✅ Base model: Llama 3.1 8B (instead of GPT-2)
- ✅ Data: Internalize CoT (same as official)
- ✅ Format: Calculator-style (same as official)
- ✅ Stages: 0 + 1 (simplified from 0 + 1 + 2 + 3)
- ✅ c_thought: Start with 1, increase to 2 if stable

---

## Design Choices & Justifications (Updated)

### Why 2 stages instead of 4?

- **Official**: Trains 4 stages (0: CoT, 1-3: progressive latent replacement)
- **Our plan**: Train 2 stages (0: CoT, 1: first step replaced)
- **Rationale**: Testing hypothesis "Can continuous thoughts help?" doesn't require full curriculum
- **Official takes**: 25 (Stage 0) + 3 + 3 + 3 (Stages 1-3) = 34 epochs
- **We'll do**: Stage 0 + Stage 1 for minimal reproduction
- Can add Stages 2-3 later if Stage 1 works

### Why start with c=1 instead of c=2?

- **Official uses**: c=2 for math tasks (GSM8k)
- **Our plan**: Start with c=1, increase to c=2 if stable
- **Rationale**: Paper showed c=3 was unstable; c=1 is safest starting point
- **Trade-off**: May get slightly worse results than paper, but lower risk
- Paper showed "performance steadily improved" from c=0 to c=2

### Why Llama 3.1 8B instead of GPT-2?

- **Official uses**: GPT-2 (124M params), though paper mentions Llama3 tested too
- **Our plan**: Llama 3.1 8B (8B params) - 64x larger!
- **Rationale**: User explicitly requested Llama 3.1 8B in initial request
- **Risk**: Larger model may behave differently (better or worse)
- **Mitigation**: If issues arise, can fall back to GPT-2

### Data: Internalize CoT vs Original GSM8k

- **Official uses**: "Internalize CoT Step by Step" dataset (385K synthetic examples)
- **Why**: Augmented data with cleaner calculator-style steps
- **Format**: `{"question": str, "steps": ["<<calc>>", ...], "answer": str}`
- **Our plan**: Use same data source for validity

---

## Technical Background (From Official Implementation)

### Data Format (Internalize CoT)

**Structure**:
```json
{
  "question": "Out of 600 employees in a company, 30% got promoted...",
  "steps": ["<<600*30/100=180>>", "<<600*10/100=60>>", "<<180+60=240>>", "<<600-240=360>>"],
  "answer": "360"
}
```

**Training Format**:
```
Out of 600 employees in a company, 30% got promoted...
<<600*30/100=180>>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

- Question on first line
- Each step on separate line (calculator-style)
- Answer prefixed with `###`
- 385,620 training examples

### Architecture (From coconut.py)

1. **Special Tokens**: Add 3 tokens to tokenizer:
   - `<|start-latent|>`: Beginning of latent thought region
   - `<|latent|>`: Placeholder for continuous thought
   - `<|end-latent|>`: End of latent thought region

2. **Continuous Thought Mechanism**:
   ```python
   # For each <|latent|> token:
   #   1. Forward pass up to position BEFORE <|latent|>
   #   2. Extract hidden state at that position
   #   3. Replace <|latent|> token embedding with the hidden state
   #   4. Continue forward pass with replaced embedding
   # Key: Each latent uses hidden state from PREVIOUS position
   ```

3. **No transformer changes**: Base LLM (GPT-2 or Llama) unchanged

### Training Process (From Official Configs)

**Stage 0 - CoT Baseline** (`args/gsm_cot.yaml`):
- Format: Full calculator-style reasoning (no latent tokens)
- Model: GPT-2 or Llama 3.1 8B
- Epochs: 25
- Batch size: 32 × 4 GPUs = 128 effective
- LR: 1e-4
- Target: ~40% validation accuracy
- c_thought: 0

**Stage 1 - First Latent** (`args/gsm_coconut.yaml`):
- Format: Replace first 1 step with c×1 = 2 latent tokens
```
question
<|start-latent|><|latent|><|latent|><|end-latent|>
<<step2>>
<<step3>>
### answer
```
- Load Stage 0 checkpoint as initialization
- Epochs: 3
- c_thought: 2
- Reset optimizer: True

**Stage 2-3** (Official uses, we'll skip for minimal reproduction):
- Stage 2: Replace first 2 steps with c×2 = 4 latent tokens (3 epochs)
- Stage 3: Replace first 3 steps with c×3 = 6 latent tokens (3 epochs)

### Training Details

- **Learning rate**: 1e-4
- **Batch size**: 32 per GPU × 4 GPUs = 128 effective
- **Optimizer**: AdamW with weight_decay=0.01
- **Loss masking**: Mask question + latent tokens + special tokens (compute loss only on remaining steps + answer)
- **Precision**: Float32 (bf16=False in official configs)

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
