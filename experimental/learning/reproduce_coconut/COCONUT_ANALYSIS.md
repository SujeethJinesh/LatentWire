# COCONUT Official Implementation Analysis

## Summary

After cloning and analyzing the official COCONUT repository, I've identified major differences from our initial plan. This document compares the official implementation with our Milestone 1.

## Key Findings

### 1. Data Format

**Official COCONUT**:
```json
{
  "question": "Out of 600 employees...",
  "steps": ["<<600*30/100=180>>", "<<600*10/100=60>>", "<<180+60=240>>", "<<600-240=360>>"],
  "answer": "360"
}
```

- Steps are **calculator-style calculations** (`<<...>>`)
- NOT natural language reasoning
- Each step is one calculation
- Dataset from "Internalize CoT Step by Step" paper (Deng et al., 2023)
- **385,620 training examples** (synthetic/augmented data)

**Our Milestone 1**:
```python
{
  "question": "Natalia sold clips...",
  "reasoning": "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips...",
  "final_answer": "72"
}
```

- Single paragraph of natural language reasoning
- **7,473 training examples** (original GSM8k)
- Steps not separated

**Difference**: ❌ **MAJOR** - Wrong data source, wrong format, 50x less data

---

### 2. Training Format

**Official COCONUT**:
```
question
<<calc1>>
<<calc2>>
<<calc3>>
### answer
```

- Format: `question\n + step1\n + step2\n + ... + \n### answer`
- Answer prefixed with `###`
- Each step on separate line
- Steps are pure calculations

**Our Milestone 1**:
```
Question: {q}
Answer: Let's think step by step. {reasoning}
The answer is {answer}
```

- Added "Let's think step by step" (not in paper)
- "The answer is" instead of "###"
- Combined reasoning into paragraph

**Difference**: ❌ **MAJOR** - Wrong format, wrong structure

---

### 3. Special Tokens

**Official COCONUT**:
- `<|start-latent|>`: Beginning of latent thoughts
- `<|latent|>`: Placeholder for continuous thought
- `<|end-latent|>`: End of latent thoughts

**Our Plan**:
- `<bot>`: Beginning of thought
- `<eot>`: End of thought

**Difference**: ⚠️ **MINOR** - Different names, but concept is same

---

### 4. Model Architecture

**Official COCONUT**:
```python
class Coconut(nn.Module):
    def forward(self, input_ids, ...):
        # For each latent token:
        #   1. Forward pass up to latent position
        #   2. Extract hidden state at position BEFORE latent
        #   3. Replace latent token embedding with that hidden state
        #   4. Continue forward pass
        # Use KV cache for efficiency
```

Key insight from `coconut.py:148-150`:
```python
# Replace latent token with hidden state from PREVIOUS position
tensor_list[batch_idx][token_idx] = hidden_states[
    batch_idx, token_idx - 1 - hidden_states_offset, :
]
```

**Our Plan**:
- Mentioned feeding hidden states back as embeddings
- Didn't specify the exact mechanism
- Didn't detail the iterative forward pass

**Difference**: ⚠️ **MEDIUM** - Concept correct, but missing implementation details

---

### 5. Training Process

**Official COCONUT**:

**Stage 0 (CoT baseline)**:
- Train GPT-2 on full CoT (no latent tokens)
- 25 epochs
- Until ~40% validation accuracy
- Batch size: 128 (32 × 4 GPUs)
- LR: 1e-4

**Stage 1-3 (Coconut training)**:
- Load Stage 0 checkpoint
- Stage 1: Replace first 1 step with c×1 = 2 latent tokens (3 epochs)
- Stage 2: Replace first 2 steps with c×2 = 4 latent tokens (3 epochs)
- Stage 3: Replace first 3 steps with c×3 = 6 latent tokens (3 epochs)
- c_thought = 2 (for math tasks)
- Reset optimizer between stages
- Total: 25 + 3 + 3 + 3 = 34 epochs

**Format at Stage 1**:
```
question
<|start-latent|><|latent|><|latent|><|end-latent|>
<<calc2>>
<<calc3>>
### answer
```

**Our Plan**:
- Stage 0: CoT baseline (correct)
- Stage 1: Continuous thoughts (correct concept)
- But c=1 (not c=2)
- Missing Stage 2, 3
- Missing optimizer reset

**Difference**: ⚠️ **MEDIUM** - Fewer stages, lower c value, but approach is similar

---

### 6. Base Model

**Official COCONUT**:
- GPT-2 (`openai-community/gpt2`)
- 124M parameters
- Tested with GPT-2 and Llama3

**Our Plan**:
- Llama 3.1 8B
- 8B parameters (64x larger!)

**Difference**: ⚠️ **MAJOR** - Much larger model, may have different behavior

---

### 7. Loss Masking

**Official COCONUT** (from `dataset.py:288-298`):
```python
"labels": [-100] * (
    len(sample["question_tokenized"])
    + n_latent_tokens
    + n_additional_tokens  # for <|start-latent|> and <|end-latent|>
) + tokens[...]
```

- Mask question tokens
- Mask latent tokens
- Mask special tokens (<|start-latent|>, <|end-latent|>)
- Compute loss ONLY on remaining steps + answer

**Our Plan**:
- Mentioned masking continuous thoughts
- Didn't specify question masking

**Difference**: ⚠️ **MINOR** - We got the concept but missed question masking

---

## Critical Differences Summary

| Aspect | Official COCONUT | Our Milestone 1 | Severity |
|--------|------------------|-----------------|----------|
| **Data source** | Internalize CoT (385K synthetic examples) | Original GSM8k (7.5K) | ❌ CRITICAL |
| **Step format** | Calculator-style `<<...>>` | Natural language reasoning | ❌ CRITICAL |
| **Data structure** | `{"question", "steps": [...], "answer"}` | `{"question", "reasoning", "final_answer"}` | ❌ CRITICAL |
| **Training format** | `question\nstep1\nstep2\n### answer` | `Question: ... Answer: Let's think step by step. ...` | ❌ CRITICAL |
| **c_thought** | 2 for math | 1 in our plan | ⚠️ MEDIUM |
| **Stages** | 0, 1, 2, 3 (4 total) | 0, 1 (2 total) | ⚠️ MEDIUM |
| **Base model** | GPT-2 (124M) | Llama 3.1 8B (8B) | ⚠️ MEDIUM |
| **Special tokens** | `<\|start-latent\|>`, `<\|latent\|>`, `<\|end-latent\|>` | `<bot>`, `<eot>` | ✅ MINOR |
| **Architecture** | Iterative forward + hidden state feedback | Concept correct | ✅ MINOR |

---

## Recommendations

### Option A: Reproduce Official COCONUT Exactly
**Pros**:
- Can directly compare with paper results
- Uses proven data and format
- Smaller model (GPT-2) trains faster

**Cons**:
- Calculator-style steps are less interesting than natural language
- Can't use Llama 3.1 8B (user's stated goal)
- Need to download 385K synthetic examples

### Option B: Adapt COCONUT to Llama 3.1 + Natural Language
**Pros**:
- Uses Llama 3.1 8B (user's goal)
- More interesting: natural language reasoning instead of calculations
- Could be novel contribution

**Cons**:
- Won't match paper's results exactly
- Need to split natural language reasoning into steps
- Riskier - may not work as well

### Option C: Hybrid Approach
**Pros**:
- Start with GPT-2 + calculator format (reproduce paper)
- Then adapt to Llama 3.1 + natural language (explore)
- Best of both worlds

**Cons**:
- More work
- Need both datasets

---

## Recommended Path Forward

**My recommendation**: **Option A** for minimal reproduction

1. Use official COCONUT data (download Internalize CoT dataset)
2. Use GPT-2 as base model
3. Use calculator-style format
4. Train Stage 0 (CoT) + Stage 1 (c=2, first step replaced)
5. Evaluate on test set

**If user wants Llama 3.1**: Modify to use Llama 3.1 8B but keep everything else the same.

**If user wants natural language**: Need to figure out how to split GSM8k natural language reasoning into discrete steps.

---

## Next Steps

1. **Discuss with user**: Which option to pursue?
2. **Update PLAN.md**: Based on chosen option
3. **Redo Milestone 1**: Load correct data format
4. **Continue implementation**: Following official structure
