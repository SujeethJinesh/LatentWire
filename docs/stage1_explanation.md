# Understanding Stage 1: Adapter-Only Training

## Normal Text Pipeline (82% F1 - WORKS)
```
Text: "What is the capital of France?"
  ↓
Tokenizer: [128000, 3923, 374, 279, 6864, 315, 9822, 30, ...]
  ↓
Embedding Layer: Token IDs → Embeddings [seq_len, 4096]
  ↓
LLM (Transformer): Process embeddings
  ↓
Output: "Paris"
```

## Stage 1 Pipeline (Target: 70% F1)
```
Text: "What is the capital of France?"
  ↓
Tokenizer: [128000, 3923, 374, 279, 6864, 315, 9822, 30, ...]
  ↓
Embedding Layer: Token IDs → Original Embeddings [seq_len, 4096]
  ↓
🔴 PCA Compression: [seq_len, 4096] → [seq_len, 512]  (8x compression)
  ↓
🟢 Adapter (TRAINED): [seq_len, 512] → [seq_len, 4096]  (reconstruct)
  ↓
LLM (Transformer): Process reconstructed embeddings
  ↓
Output: "Paris" (hopefully!)
```

## What We're Training

The adapter learns to **reconstruct** the original embeddings from compressed versions:

```python
# Training loop pseudo-code
original_embeddings = model.get_input_embeddings()(token_ids)  # [B, L, 4096]
compressed = PCA(original_embeddings)                           # [B, L, 512]
reconstructed = adapter(compressed)                             # [B, L, 4096]

# Loss 1: Reconstruction - make sure we preserve information
reconstruction_loss = MSE(reconstructed, original_embeddings)

# Loss 2: Task performance - make sure LLM still works
output = model(inputs_embeds=reconstructed)
task_loss = CrossEntropy(output, target_tokens)

total_loss = reconstruction_loss + task_loss
```

## Why PCA Instead of Random Projection?

### PCA (Principal Component Analysis)
- **Preserves maximum variance** in the data
- Finds the most important 512 dimensions out of 4096
- Information loss is minimized
- Reconstruction is easier because we kept the important parts
- Like doing JPEG compression - keeps important details

### Random Projection
- Just random linear transformation
- No guarantee of preserving important information
- Much harder to reconstruct
- Like randomly throwing away pixels - loses critical info

## Why We Need the Model Loaded

We need the full model for THREE critical things:

1. **Get Original Embeddings**
   ```python
   embeddings = model.get_input_embeddings()(token_ids)
   ```
   We need these as our training targets

2. **Test if Reconstruction Works**
   ```python
   output = model(inputs_embeds=reconstructed_embeddings)
   ```
   We need to verify the LLM can still understand our reconstructed embeddings

3. **Compute Task Loss**
   ```python
   loss = model(inputs_embeds=reconstructed, labels=answer_tokens).loss
   ```
   We need to ensure the model generates correct answers, not just matching embeddings

## Why This is Different from Full LatentWire

### Full LatentWire (Currently Broken)
```
Text → Byte Encoder → Latents [32, 256] → Adapter → Embeddings → LLM
       ↑                ↑                    ↑
    TRAINED         TRAINED              TRAINED
    (learning          (learning          (learning
     from scratch)     from scratch)      from scratch)
```

### Stage 1 (Should Work)
```
Text → Tokenizer → Embeddings → PCA → Compressed → Adapter → Embeddings → LLM
                      ↑           ↑         ↑           ↑
                   KNOWN GOOD   FIXED   ONLY THIS   TRAINED
                                        IS TRAINED
```

## The Key Insight

- **Stage 1**: We start with embeddings that ALREADY WORK (82% F1)
- We compress them only slightly (4096→512 instead of full tokenization→32)
- The adapter only needs to learn inverse of PCA, not create embeddings from scratch
- This SHOULD work because we're barely changing what already works

## Analogy

Think of it like learning to compress images:

- **Full LatentWire**: Learn to draw pictures from descriptions (very hard!)
- **Stage 1**: Learn to decompress JPEG images (much easier!)

We're testing if the adapter can learn decompression before trying the much harder task of creating representations from scratch.

## Options 2 & 3 Don't Need More Training

These are **diagnostic tools** for your existing checkpoint:

- **Option 2** (`progressive_ablation.sh`): Tests each component of your ALREADY TRAINED model
- **Option 3** (`test_embedding_match.py`): Checks if your EXISTING model's outputs match embedding statistics

They analyze what you've already trained to identify WHERE it fails, not train something new.