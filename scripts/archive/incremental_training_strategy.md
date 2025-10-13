# Incremental Training Strategy for LatentWire

## Current Status
✅ **PROVEN**: inputs_embeds interface works perfectly (82% F1)
❌ **BROKEN**: Full latent pipeline (0% F1)
🎯 **GOAL**: Build incrementally to identify and fix failure points

## Proposed Incremental Stages

### Stage 1: Adapter-Only Training (No Encoder)
**Goal**: Prove adapter can learn to project to embedding space
**Setup**:
- Start with pre-computed text embeddings
- Add small noise/compression (e.g., PCA to 256 dims)
- Train only adapter to reconstruct original embeddings
- Success metric: 70%+ F1 (close to 82% baseline)

```python
# Pseudo-code
text_embeds = model.get_input_embeddings()(input_ids)  # Shape: [B, L, 4096]
compressed = pca_compress(text_embeds, dim=256)        # Shape: [B, L, 256]
reconstructed = adapter(compressed)                     # Shape: [B, L, 4096]
loss = mse(reconstructed, text_embeds) + ce_loss(generated_tokens)
```

### Stage 2: Fixed Random Encoder + Trainable Adapter
**Goal**: Test if adapter can handle non-embedding inputs
**Setup**:
- Use fixed random projection as "encoder"
- Train adapter to map random features → embeddings
- This tests adapter's projection capability
- Success metric: 50%+ F1

```python
# Pseudo-code
random_encoder = nn.Linear(512, 256)  # Fixed, not trained
random_encoder.weight.data.normal_(0, 0.02)

byte_input = text.encode('utf-8')
random_features = random_encoder(byte_input)  # Fixed transformation
adapted = adapter(random_features)            # Trained
loss = ce_loss(model(inputs_embeds=adapted))
```

### Stage 3: Trainable Encoder + Adapter (Single Model)
**Goal**: Full pipeline for one model (Llama only)
**Setup**:
- Train encoder + adapter end-to-end
- Focus on single model first
- Use strong regularization to prevent collapse
- Success metric: 30%+ F1

```python
# Current approach but with:
- Stronger first-token supervision (weight=5.0)
- Embedding distribution matching loss
- Gradient clipping on encoder
- Lower learning rate for encoder than adapter
```

### Stage 4: Shared Latent Space (Two Models)
**Goal**: True interlingua between Llama and Qwen
**Setup**:
- Alternate training between models
- Shared encoder, model-specific adapters
- Cross-model distillation
- Success metric: 20%+ F1 on both models

## Key Insights from Failures

1. **Distribution Mismatch**: Latents don't match embedding statistics
   - Solution: Add explicit distribution matching loss
   - Calibrate per-example, not per-batch

2. **Mode Collapse**: Model predicts high-frequency tokens
   - Solution: Stronger entropy regularization
   - Diverse sampling during training

3. **Gradient Issues**: Encoder gradients too large/small
   - Solution: Different learning rates for encoder/adapter
   - Gradient clipping and warmup

4. **Information Bottleneck**: Too much compression too fast
   - Solution: Start with less compression (64 or 128 latents)
   - Gradually reduce during training

## Immediate Next Steps

### Quick Experiment 1: Adapter-Only Baseline
```bash
python latentwire/train_adapter_only.py \
  --use_precomputed_embeddings \
  --compress_ratio 0.9 \  # Light compression
  --adapter_lr 1e-3 \     # Higher LR for faster convergence
  --epochs 5 \
  --samples 5000
```

### Quick Experiment 2: Fix Distribution Matching
```bash
python latentwire/train.py \
  --distribution_matching_weight 1.0 \
  --calibration embed_rms \
  --per_example_calibration \
  --encoder_lr 1e-5 \      # Much lower than adapter
  --adapter_lr 1e-4 \
  --first_token_ce_weight 5.0
```

### Quick Experiment 3: Less Aggressive Compression
```bash
python latentwire/train.py \
  --latent_len 64 \       # Double the latents
  --d_z 512 \             # Larger dimension
  --encoder_dropout 0.2 \  # More regularization
  --samples 10000
```

## Success Criteria for Each Stage

| Stage | Description | Target F1 | First-Token Acc | Timeline |
|-------|-------------|-----------|-----------------|----------|
| 1 | Adapter-only | 70% | 60% | 1 hour |
| 2 | Fixed encoder | 50% | 40% | 2 hours |
| 3 | Full single model | 30% | 25% | 4 hours |
| 4 | Two models | 20% | 15% | 8 hours |

## Diagnostic Commands

```bash
# Test if adapter is learning
python scripts/test_embedding_match.py

# Run progressive ablation
bash scripts/progressive_ablation.sh runs/checkpoint_dir

# Test specific components
python scripts/incremental_validation.py --checkpoint runs/dir --step 1

# Compare distributions
python -c "from latentwire.diagnostics import compare_distributions; compare_distributions('runs/dir')"
```

## Key Principle

**Start with what works and add complexity gradually**. Each stage should maintain at least 60% of the previous stage's performance. If a stage drops below this threshold, stop and debug before proceeding.