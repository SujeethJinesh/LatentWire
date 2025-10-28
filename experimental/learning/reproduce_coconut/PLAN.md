# COCONUT Reproduction Plan (Option A: Official Implementation with Llama 3.1 8B)

**Based on**: Official COCONUT implementation analysis (https://github.com/facebookresearch/coconut)
**Goal**: Reproduce COCONUT with Llama 3.1 8B on GSM8k using official data format
**Approach**: Minimal reproduction (Stage 0 + Stage 1) following official implementation

---

## Paper & Implementation

- **Paper**: "Training Large Language Models to Reason in a Continuous Latent Space" (Dec 2024, Meta AI)
- **ArXiv**: https://arxiv.org/abs/2412.06769
- **Official Repo**: https://github.com/facebookresearch/coconut
- **Key Idea**: Replace discrete reasoning steps with continuous hidden states fed back as input embeddings

---

## Data Format (Internalize CoT Dataset)

### Source
- **Dataset**: "Internalize CoT Step by Step" (Deng et al., 2023)
- **URL**: https://github.com/da03/Internalize_CoT_Step_by_Step/tree/main/data/gsm8k
- **Size**: 385,620 train examples (vs 7,473 in original GSM8k)
- **Why**: Synthetic augmented data with clean calculator-style reasoning steps

### Structure
```json
{
  "question": "Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?",
  "steps": [
    "<<600*30/100=180>>",
    "<<600*10/100=60>>",
    "<<180+60=240>>",
    "<<600-240=360>>"
  ],
  "answer": "360"
}
```

**Key characteristics**:
- `question`: String (the math word problem)
- `steps`: List of strings (calculator-style calculations in `<<...>>` format)
- `answer`: String (final numerical answer)
- Each step is ONE calculation
- Number of steps varies per problem (1-8 typically)

### Training Format (How Model Sees It)

**Stage 0 (CoT baseline - no latent)**:
```
Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?
<<600*30/100=180>>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

**Stage 1 (First step → latent, c=1)**:
```
Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?
<|start-latent|><|latent|><|end-latent|>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

**Stage 1 (First step → latent, c=2)**:
```
Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?
<|start-latent|><|latent|><|latent|><|end-latent|>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

**Format rules**:
- Question on first line, followed by newline
- Each step on separate line, followed by newline
- Answer prefixed with `### ` (three hashes and space)
- No "Let's think step by step" prompt
- No "The answer is" wrapper

---

## Architecture Details

### Special Tokens
Add 3 tokens to Llama 3.1 8B tokenizer:
- `<|start-latent|>`: Marks beginning of latent thought region
- `<|latent|>`: Placeholder for continuous thought (will be replaced by hidden state)
- `<|end-latent|>`: Marks end of latent thought region

### Continuous Thought Mechanism

**Key algorithm** (from `coconut.py`):
```python
# For each <|latent|> token in the sequence:
#   1. Forward pass up to position BEFORE the <|latent|> token
#   2. Extract hidden state h[t-1] at that position
#   3. Replace embedding of <|latent|> token with h[t-1]
#   4. Continue forward pass with replaced embedding
#   5. Use KV cache for efficiency (don't recompute earlier positions)
```

**Example**:
```
Input IDs:  [question_tokens] [start] [latent] [latent] [end] [step2_tokens]
                    ↓             ↓       ↓        ↓      ↓         ↓
Embeddings: [E(q)]           [E(start)] [h[q[-1]]] [h[start]] [E(end)] [E(step2)]
                                           ↑           ↑
                                    Replace with    Replace with
                                    hidden state    hidden state
                                    from prev pos   from prev pos
```

**No transformer changes**: Base Llama 3.1 8B architecture unchanged

### Loss Masking

**Mask (ignore) during training**:
- All question tokens
- All latent tokens (`<|latent|>`)
- Special tokens (`<|start-latent|>`, `<|end-latent|>`)

**Compute loss on**:
- Remaining reasoning steps (after latent region)
- Answer tokens

Example labels:
```
Tokens:  [question...] [<|start-latent|>] [<|latent|>] [<|end-latent|>] [step2...] [###] [answer]
Labels:  [-100 ...]    [-100]              [-100]       [-100]           [step2...] [###] [answer]
```

---

## Training Process

### Stage 0: CoT Baseline

**Goal**: Train Llama 3.1 8B on full calculator-style CoT (no latent tokens)

**Configuration**:
- Model: Llama 3.1 8B (or GPT-2 for faster testing)
- Data: Internalize CoT train set (385K examples)
- Format: Full reasoning (`question\nstep1\nstep2\n...\n### answer`)
- Epochs: 3-6 (reduced from official 25 for faster iteration)
- Batch size: 8 per GPU (adjust for MacBook MPS)
- Learning rate: 1e-4
- Optimizer: AdamW (weight_decay=0.01)
- c_thought: 0 (no latent tokens)

**Target**: Model should learn to solve GSM8k problems with calculator-style reasoning

**Training subset**: Start with 10K examples for fast iteration, scale to full 385K later

**Validation**: Compute accuracy on validation set every epoch

### Stage 1: First Continuous Thought

**Goal**: Replace first reasoning step with c continuous thought tokens

**Configuration**:
- Model: **Load Stage 0 checkpoint** as initialization
- Data: Same Internalize CoT train set
- Format: First step replaced with `<|start-latent|><|latent|>...<|end-latent|>`
- Epochs: 3
- Batch size: 8 per GPU
- Learning rate: 1e-4
- Optimizer: **Reset** (don't use Stage 0 optimizer state)
- c_thought: 1 (start conservative, increase to 2 if stable)

**Training subset**: Same 10K examples as Stage 0

**Validation**: Compare accuracy with Stage 0 baseline

---

## Milestones (Revised)

### Milestone 1: Data Loading & Preprocessing ✅ TO REIMPLEMENT

**Goal**: Download Internalize CoT dataset and understand format

**Tasks**:
1. Download data from Internalize CoT repo
2. Load and parse JSON format
3. Verify structure: `{"question", "steps": [...], "answer"}`
4. Show examples in both raw and training format
5. Compute statistics (step counts, lengths)

**File**: `step1_data.py`

**Command**:
```bash
cd experimental/learning/reproduce_coconut
python step1_data.py
```

**Expected output**:
- Dataset statistics (385K train examples)
- Example in raw JSON format
- Example in Stage 0 training format
- Example in Stage 1 training format (with latent tokens)
- Step count distribution

---

### Milestone 2: Add Special Tokens to Llama 3.1 8B

**Goal**: Extend tokenizer with 3 special tokens without breaking model

**Tasks**:
1. Load Llama 3.1 8B tokenizer
2. Add tokens: `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`
3. Resize model embeddings (add 3 new rows, initialize randomly)
4. Test tokenization with new tokens
5. Verify model still generates coherent text

**File**: `step2_tokens.py`

**Command**:
```bash
python step2_tokens.py
```

**Expected output**:
- Original vocab size: 128256
- New vocab size: 128259 (+3)
- Token IDs for special tokens
- Test generation showing model still works

---

### Milestone 3: Implement COCONUT Forward Pass

**Goal**: Create custom model class that replaces latent tokens with hidden states

**Tasks**:
1. Create `Coconut` wrapper class (based on `coconut.py`)
2. Implement iterative forward pass with hidden state feedback
3. Implement KV cache for efficiency
4. Test forward pass with dummy data
5. Verify gradients flow correctly

**File**: `step3_coconut_model.py`

**Command**:
```bash
python step3_coconut_model.py --test_only
```

**Expected output**:
- Forward pass diagram
- Loss computation (with proper masking)
- Gradient flow verification
- Comparison with base model forward pass

---

### Milestone 4: Stage 0 Training (CoT Baseline)

**Goal**: Train Llama 3.1 8B on calculator-style CoT (no latent tokens)

**Tasks**:
1. Create dataset with Stage 0 format
2. Implement training loop with HuggingFace Trainer
3. Train on 10K examples for 3 epochs
4. Save checkpoint
5. Evaluate on validation set

**File**: `step4_stage0_train.py`

**Command**:
```bash
python step4_stage0_train.py --samples 10000 --epochs 3 --output_dir runs/stage0
```

**Expected output**:
- Training progress (loss decreasing)
- Checkpoint saved to `runs/stage0/`
- Validation accuracy
- Sample predictions (5-10 examples)

---

### Milestone 5: Stage 1 Training (Continuous Thoughts)

**Goal**: Train with first step replaced by c=1 continuous thoughts

**Tasks**:
1. Create dataset with Stage 1 format (latent tokens)
2. Use COCONUT model from Milestone 3
3. Load Stage 0 checkpoint as initialization
4. Train on same 10K examples for 3 epochs
5. Save checkpoint

**File**: `step5_stage1_train.py`

**Command**:
```bash
python step5_stage1_train.py --samples 10000 --epochs 3 --c_thought 1 \
  --load_ckpt runs/stage0 --output_dir runs/stage1
```

**Expected output**:
- Training progress with continuous thoughts
- Checkpoint saved to `runs/stage1/`
- Validation accuracy
- Sample predictions

---

### Milestone 6: Evaluation & Comparison

**Goal**: Compare Stage 0 (CoT) vs Stage 1 (COCONUT) performance

**Tasks**:
1. Load both checkpoints
2. Evaluate on same validation set (1000 examples)
3. Compute accuracy (exact match on final answer)
4. Measure efficiency (tokens used, inference time)
5. Show side-by-side examples

**File**: `step6_eval.py`

**Command**:
```bash
python step6_eval.py \
  --stage0_ckpt runs/stage0 \
  --stage1_ckpt runs/stage1 \
  --eval_samples 1000
```

**Expected output**:
```
Stage 0 (CoT):      Accuracy = X.XX%  Avg tokens = YYY  Time = Z.Zs
Stage 1 (COCONUT):  Accuracy = X.XX%  Avg tokens = YYY  Time = Z.Zs

Efficiency gain: X% fewer tokens
Quality: ±X% accuracy change

Example 1:
Question: ...
Stage 0 answer: ... (correct/wrong)
Stage 1 answer: ... (correct/wrong)
Ground truth: ...
```

---

## Implementation Notes

### Hardware Requirements

**MacBook (your setup)**:
- 64GB RAM ✅ Sufficient for Llama 3.1 8B
- MPS backend ✅ GPU acceleration available
- Disk: ~20GB for model + data + checkpoints

**Training time estimates** (on MacBook MPS):
- Milestone 4 (Stage 0, 10K examples, 3 epochs): ~2-3 hours
- Milestone 5 (Stage 1, 10K examples, 3 epochs): ~2-3 hours
- Total: ~5-6 hours for minimal reproduction

### Key Simplifications from Official

| Aspect | Official | Our Implementation | Justification |
|--------|----------|-------------------|---------------|
| Training examples | 385K | 10K (then scale) | Faster iteration |
| Stage 0 epochs | 25 | 3-6 | Faster to validate approach |
| Stages | 0, 1, 2, 3 | 0, 1 | Minimal reproduction |
| c_thought | 2 | 1 (then 2) | Start conservative |
| Base model | GPT-2 | Llama 3.1 8B | User requirement |
| GPUs | 4× A100 | 1× MPS | Different hardware |
| Batch size | 128 | 8 | Memory constraints |

### Success Criteria

**Milestone 1-3**: Technical validation
- ✅ Data loads correctly
- ✅ Special tokens work
- ✅ Forward pass works with latent tokens

**Milestone 4 (Stage 0)**: Baseline established
- ✅ Loss decreases during training
- ✅ Model can solve some GSM8k problems
- ✅ Validation accuracy > 10% (sanity check)

**Milestone 5-6 (Stage 1)**: COCONUT validation
- ✅ Training completes without errors
- ✅ Stage 1 accuracy ≥ 80% of Stage 0 accuracy (proving continuous thoughts don't hurt much)
- 🎯 Stage 1 uses fewer tokens than Stage 0 (efficiency gain)
- 🎯 Stage 1 accuracy ≥ Stage 0 (ideal outcome)

---

## File Structure

```
experimental/learning/reproduce_coconut/
├── PLAN.md                      # This file
├── README.md                    # Quick start guide
├── COCONUT_ANALYSIS.md          # Detailed comparison with official
├── coconut/                     # Official repo (for reference)
│
├── step1_data.py               # Milestone 1: Data loading
├── step2_tokens.py             # Milestone 2: Special tokens
├── step3_coconut_model.py      # Milestone 3: COCONUT forward pass
├── step4_stage0_train.py       # Milestone 4: Stage 0 training
├── step5_stage1_train.py       # Milestone 5: Stage 1 training
├── step6_eval.py               # Milestone 6: Evaluation
│
└── runs/                        # Training outputs
    ├── stage0/                  # Stage 0 checkpoint + logs
    └── stage1/                  # Stage 1 checkpoint + logs
```

---

## Next Steps After Milestone 6

### If COCONUT Works (Stage 1 ≥ Stage 0):
1. Increase c_thought from 1 to 2 (match paper)
2. Train on full 385K dataset (not just 10K)
3. Add Stage 2 (replace first 2 steps)
4. Increase Stage 0 to more epochs for better baseline

### If COCONUT Doesn't Work (Stage 1 < Stage 0):
1. Debug continuous thought mechanism
2. Check loss masking is correct
3. Verify hidden states are actually being used
4. Try c_thought=2 instead of c=1
5. Compare with official GPT-2 implementation

### Future Explorations:
1. Adapt to natural language reasoning (instead of calculator-style)
2. Try with original GSM8k (7.5K examples)
3. Test on other reasoning datasets (ProntoQA, ProsQA)
4. Explore inference-time scaling (generate multiple continuous thoughts)

---

## References

- **Paper**: https://arxiv.org/abs/2412.06769
- **Official Code**: https://github.com/facebookresearch/coconut
- **Internalize CoT**: https://github.com/da03/Internalize_CoT_Step_by_Step
- **GSM8k**: https://huggingface.co/datasets/openai/gsm8k
- **Llama 3.1**: https://huggingface.co/meta-llama/Llama-3.1-8B
