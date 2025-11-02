# Gist Tokens Reproduction Plan for Llama 3.1 8B

## Goal
Reproduce or beat "Learning to Compress Prompts with Gist Tokens" (Mu et al., NeurIPS 2023) using Llama 3.1 8B.

## Their Setup (LLaMA-7B)
- **Model:** LLaMA-7B (similar to our Llama 3.1 8B ✓)
- **Dataset:** Alpaca+ instruction tuning dataset
- **Gist tokens:** 1-10 tokens (they test multiple values)
- **Hardware:** 4× A100 80GB GPUs (we have 4× H100s ✓)
- **Batch size:** 1 (critical - required for position embeddings)
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L on seen/unseen/human splits

## Our Advantages
✅ **Similar model:** Llama 3.1 8B vs their LLaMA-7B (same architecture family)
✅ **Better hardware:** 4× H100 vs their 4× A100
✅ **Same framework:** PyTorch + HuggingFace

## Core Implementation Requirements

### 1. Attention Masking (CRITICAL - The Paper's Key Innovation)

The magic of Gist is in the **attention mask pattern**:

```
Sequence structure: [GIST_1, GIST_2, ..., GIST_K, PROMPT_1, ..., PROMPT_N, GEN_1, ..., GEN_M]

Attention pattern:
- Gist tokens: Can attend to EVERYTHING (bidirectional within gist + prompt)
- Prompt tokens: Can attend to GIST + previous prompt tokens (causal on prompt)
- Generated tokens: Can ONLY attend to GIST (not prompt!)

Why this works:
- During training: Gist learns to compress prompt information
- During inference: Generated tokens decode from compressed gist only
- Result: Prompt tokens can be dropped after gist computation
```

**Implementation:**
```python
def create_gist_attention_mask(
    num_gist_tokens: int,
    prompt_length: int,
    generation_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create attention mask for gist training.

    Returns:
        mask: [seq_len, seq_len] where 1 = attend, 0 = don't attend
    """
    total_len = num_gist_tokens + prompt_length + generation_length
    mask = torch.zeros(total_len, total_len, device=device)

    # Gist tokens: attend to everything
    mask[:num_gist_tokens, :] = 1

    # Prompt tokens: attend to gist + causal on prompt
    for i in range(num_gist_tokens, num_gist_tokens + prompt_length):
        mask[i, :num_gist_tokens] = 1  # Attend to all gist
        mask[i, num_gist_tokens:i+1] = 1  # Causal on prompt

    # Generated tokens: ONLY attend to gist
    mask[num_gist_tokens + prompt_length:, :num_gist_tokens] = 1

    return mask
```

### 2. Position Embeddings (CRITICAL for RoPE)

Llama uses RoPE (Rotary Position Embeddings). We need special handling:

```python
def create_gist_position_ids(
    num_gist_tokens: int,
    prompt_length: int,
    generation_length: int,
    device: torch.device
) -> torch.Tensor:
    """
    Position IDs with gist offset.

    Gist tokens: positions [0, K-1]
    Prompt tokens: positions [0, N-1] (restart at 0!)
    Generated tokens: positions [0, M-1] (restart at 0!)
    """
    position_ids = []

    # Gist tokens: [0, 1, 2, ..., K-1]
    position_ids.append(torch.arange(num_gist_tokens, device=device))

    # Prompt tokens: [0, 1, 2, ..., N-1] (restart!)
    position_ids.append(torch.arange(prompt_length, device=device))

    # Generated tokens: [0, 1, 2, ..., M-1] (restart!)
    position_ids.append(torch.arange(generation_length, device=device))

    return torch.cat(position_ids)
```

This is why **batch_size=1 is required** - can't easily batch different position patterns.

### 3. Training Procedure

**Phase 1: Gist Token Insertion**
```python
def insert_gist_tokens(
    input_embeds: torch.Tensor,  # [batch=1, seq_len, hidden]
    gist_embeds: torch.Tensor,   # [num_gist, hidden]
) -> torch.Tensor:
    """Insert gist tokens at the beginning."""
    batch_size = input_embeds.size(0)
    gist_batch = gist_embeds.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([gist_batch, input_embeds], dim=1)
```

**Phase 2: Forward Pass with Gist Masking**
```python
def forward_with_gist(
    model: AutoModelForCausalLM,
    input_embeds: torch.Tensor,
    gist_embeds: torch.Tensor,
    labels: torch.Tensor,
    num_gist_tokens: int
):
    # Insert gist tokens
    embeds_with_gist = insert_gist_tokens(input_embeds, gist_embeds)

    # Create gist attention mask
    mask = create_gist_attention_mask(
        num_gist_tokens=num_gist_tokens,
        prompt_length=input_embeds.size(1),
        generation_length=labels.size(1),
        device=input_embeds.device
    )

    # Create position IDs
    position_ids = create_gist_position_ids(
        num_gist_tokens=num_gist_tokens,
        prompt_length=input_embeds.size(1),
        generation_length=labels.size(1),
        device=input_embeds.device
    )

    # Forward pass with custom masking
    outputs = model(
        inputs_embeds=embeds_with_gist,
        attention_mask=mask,
        position_ids=position_ids,
        labels=labels
    )

    return outputs
```

**Phase 3: Training Loop**
```python
# Learnable gist embeddings
gist_embeds = nn.Parameter(torch.randn(num_gist_tokens, hidden_dim) * 0.02)

# Optimizer only for gist embeddings + LoRA
optimizer = AdamW([
    {'params': gist_embeds, 'lr': 1e-3},  # Higher LR for gist
    {'params': model.parameters(), 'lr': 5e-5}  # LoRA params
])

for epoch in range(num_epochs):
    for batch in dataloader:
        # Batch size MUST be 1
        assert batch['input_ids'].size(0) == 1

        # Get embeddings
        input_embeds = model.get_input_embeddings()(batch['input_ids'])

        # Forward with gist
        outputs = forward_with_gist(
            model, input_embeds, gist_embeds,
            batch['labels'], num_gist_tokens
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Dataset: Alpaca+ for Instruction Tuning

**Get the dataset:**
```bash
# From their repo or use HuggingFace
from datasets import load_dataset
alpaca = load_dataset("yahma/alpaca-cleaned")
```

**Format:**
```python
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",  # Optional context
    "output": "1. Eat a balanced diet..."
}
```

**Preprocessing:**
```python
def format_alpaca(example):
    """Format Alpaca example for gist training."""
    if example["input"]:
        prompt = f"{example['instruction']}\n\n{example['input']}"
    else:
        prompt = example["instruction"]

    response = example["output"]

    return {
        "prompt": prompt,
        "response": response
    }
```

### 5. Evaluation

**Metrics (from paper):**
- ROUGE-1, ROUGE-2, ROUGE-L
- Evaluated on:
  - **Seen tasks:** Tasks in training set
  - **Unseen tasks:** Held-out tasks
  - **Human eval:** Human-written prompts

**Our evaluation:**
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_gist(model, gist_embeds, eval_data, num_gist_tokens):
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for example in eval_data:
        # Get prompt embeddings
        prompt_ids = tokenizer(example['prompt'], return_tensors='pt').input_ids
        prompt_embeds = model.get_input_embeddings()(prompt_ids)

        # Insert gist
        embeds_with_gist = insert_gist_tokens(prompt_embeds, gist_embeds)

        # Generate (only gist tokens used for generation!)
        # This requires custom generation that only attends to gist
        generated = generate_from_gist(model, embeds_with_gist, num_gist_tokens)

        # Score
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        reference = example['response']

        score = scorer.score(reference, prediction)
        for k in scores:
            scores[k].append(score[k].fmeasure)

    return {k: np.mean(v) for k, v in scores.items()}
```

## Implementation Strategy

### Phase 1: Minimal Reproduction (1 week)
**Goal:** Reproduce paper's core results on Alpaca+

1. **Day 1-2:** Implement attention masking + position IDs
2. **Day 3:** Set up Alpaca+ dataset loading
3. **Day 4-5:** Training loop with batch_size=1
4. **Day 6:** Custom generation that uses only gist
5. **Day 7:** Evaluation + compare to paper

**Target:** Match their ROUGE scores for LLaMA-7B

### Phase 2: Adaptation to SQuAD (3 days)
**Goal:** Apply gist to our SQuAD task

1. **Day 1:** Adapt gist training for Q&A format
2. **Day 2:** Train on SQuAD with gist masking
3. **Day 3:** Evaluate F1/EM vs baselines

### Phase 3: Optimization (1 week)
**Goal:** Beat paper or find better variants

1. Try different gist token counts (1, 2, 5, 10)
2. Try different learning rates
3. Try different position encoding schemes
4. Try increasing batch size (if we can solve position IDs)

## Expected Results

**Paper's results (LLaMA-7B):**
- Up to 26× compression
- 40% FLOPs reduction
- 4.2% wall time speedup
- Minimal ROUGE degradation vs full prompt

**Our targets (Llama 3.1 8B):**
- ✓ Match or beat their LLaMA-7B results (similar model size)
- ✓ Achieve >20× compression with <10% quality drop
- ✓ Demonstrate speedup on SQuAD
- ✓ Show gist beats truncation baseline

## Critical Success Factors

1. ✅ **Get attention masking right** - This is the innovation
2. ✅ **Handle position IDs correctly** - Required for RoPE
3. ✅ **Use batch_size=1** - Don't fight the implementation
4. ✅ **Use instruction tuning dataset first** - Validate on their task
5. ✅ **Implement proper generation** - Only attend to gist

## Risks & Mitigations

**Risk 1:** Can't reproduce their numbers
- **Mitigation:** Start with their exact setup, then adapt
- **Fallback:** At minimum, beat truncation baseline

**Risk 2:** Batch size = 1 too slow
- **Mitigation:** Use gradient accumulation (they did this)
- **Alternative:** Try to solve batching with position IDs

**Risk 3:** Llama 3.1 behaves differently than LLaMA-7B
- **Mitigation:** Should be similar (same architecture)
- **Fallback:** Report results for Llama 3.1 specifically

## Implementation Checklist

### Core Components
- [ ] `GistAttentionMask` class
- [ ] `GistPositionIDs` class
- [ ] `GistTrainer` class with proper masking
- [ ] Custom generation function (gist-only attention)
- [ ] Alpaca+ dataset loader
- [ ] ROUGE evaluation

### Validation
- [ ] Unit test: attention mask shape
- [ ] Unit test: position IDs correctness
- [ ] Integration test: forward pass works
- [ ] Validation: reproduce paper's LLaMA-7B results
- [ ] Comparison: gist vs truncation vs full prompt

### Experiments
- [ ] Baseline: Full prompt (no compression)
- [ ] Baseline: Truncation to K tokens
- [ ] Gist: K=1 token
- [ ] Gist: K=5 tokens
- [ ] Gist: K=10 tokens

## Timeline

**Week 1:** Core implementation + validation
- Days 1-2: Attention masking + position IDs
- Days 3-4: Training loop
- Days 5-6: Generation + evaluation
- Day 7: Initial results

**Week 2:** Reproduction + adaptation
- Days 1-3: Reproduce on Alpaca+
- Days 4-5: Adapt to SQuAD
- Days 6-7: Baseline comparisons

**Week 3:** Optimization + analysis
- Days 1-3: Hyperparameter tuning
- Days 4-5: Ablations
- Days 6-7: Final results + writeup

**Total: 3 weeks to full reproduction**

## Next Steps

1. **Confirm approach:** Does this plan sound good?
2. **Start implementation:** Begin with attention masking
3. **Set up dataset:** Get Alpaca+ ready
4. **Validate incrementally:** Test each component

Let me know if you want me to start implementing the core components (attention masking, position IDs, training loop).
