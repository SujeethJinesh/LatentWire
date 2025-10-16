# Experiment 2: Sequence-Dimension PCA with Positional Decoding

## Executive Summary

Compress sequence length (300 → 128) by applying PCA **across the sequence dimension** instead of the feature dimension, while preserving positional information through a learned positional decoder.

**Key Innovation:** Unlike Phase 1a which compresses features (4096 → 1024), this compresses time/sequence (300 → 128) while keeping full feature dimension (4096).

## The Core Idea

### Standard (Phase 1a) Approach:
```python
embeddings = [300 tokens, 4096 dims]

# Compress each token's features independently:
for each of 300 tokens:
    token[i] = PCA(token[i])  # 4096 → 1024

result = [300 tokens, 1024 dims]  # Same sequence length!
```

### Sequence-PCA Approach (Experiment 2):
```python
embeddings = [300 tokens, 4096 dims]

# Transpose: treat each feature as a "time series"
embeddings_T = embeddings.T  # [4096 features, 300 time steps]

# Find principal temporal patterns across the sequence
for each of 4096 features:
    temporal_pattern[feature] = PCA(feature_across_time)  # 300 → 128

result_T = [4096 features, 128 temporal components]
result = result_T.T  # [128 temporal tokens, 4096 dims]
```

**Result:** Each of the 128 "temporal tokens" is a principal component - a weighted combination of the original 300 positions.

## The Fundamental Problem (Codex's Concern)

**Problem:** The 128 PCA components are linear combinations of original positions. The LLM has no way to know "the answer is at position 157" because positions are now mixed!

**Example:**
```
Original positions: [0, 1, 2, ..., 299]
After sequence-PCA: [component_0, component_1, ..., component_127]

component_0 might be: 0.3*pos_5 + 0.2*pos_47 + 0.1*pos_143 + ...
                      (a weighted average of many positions)

The LLM can't "point to" position 157 anymore!
```

## Our Solution: Positional Reconstruction Decoder

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING TIME                              │
└─────────────────────────────────────────────────────────────┘

Text → Embeddings [300, 4096]
         ↓
    Sequence-PCA [128, 4096]  ← Principal temporal components
         ↓
    Positional Decoder (learns to reconstruct position information)
         ↓
    Position-Aware Embeddings [128, 4096 + 128]
         ↓                         ↑
         └──────────────────── position embeddings
         ↓
    LLM Generate → Loss


┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE TIME                               │
└─────────────────────────────────────────────────────────────┘

Text → Embeddings [300, 4096]
         ↓
    Sequence-PCA [128, 4096]
         ↓
    Positional Decoder:
      - Maps each PCA component to a "position distribution"
      - component_i → [p(pos_0), p(pos_1), ..., p(pos_299)]
      - Extract expected position: E[position] for each component
         ↓
    Add learned position embeddings [128, 4096]
         ↓
    LLM Generate
```

### Detailed Components

#### 1. Sequence-Dimension PCA

**Fitting (offline, on training set):**

```python
# Collect embeddings from N training examples
all_embeddings = []  # List of [seq_i, 4096] tensors

for example in training_set[:5000]:
    emb = get_embeddings(example.text)  # [seq_i, 4096]
    # Pad/truncate to fixed length (300)
    emb_padded = pad_to_length(emb, 300)  # [300, 4096]
    all_embeddings.append(emb_padded)

# Stack: [N, 300, 4096]
all_embeddings = torch.stack(all_embeddings)  # [5000, 300, 4096]

# For each of 4096 features, collect its temporal pattern across all examples
temporal_data = all_embeddings.permute(2, 0, 1)  # [4096, 5000, 300]
temporal_data = temporal_data.reshape(4096, -1)  # [4096, 5000*300]

# Fit PCA on temporal patterns
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
pca.fit(temporal_data.T)  # Fit on [5000*300, 4096]

# The components are now "temporal patterns"
# Shape: [128, 4096]
```

**Transform (at runtime):**

```python
def sequence_pca_transform(embeddings):
    # embeddings: [seq_len, 4096]
    # Pad to 300 if needed
    if embeddings.shape[0] < 300:
        embeddings = pad_to_length(embeddings, 300)
    elif embeddings.shape[0] > 300:
        embeddings = embeddings[:300]  # Truncate

    # Transpose for sequence-wise PCA
    emb_T = embeddings.T  # [4096, 300]

    # Project each feature's temporal pattern
    compressed_T = pca.transform(emb_T)  # [4096, 128]

    # Transpose back
    compressed = compressed_T.T  # [128, 4096]

    return compressed
```

#### 2. Positional Decoder Network

**The key innovation: Learn to inject position information back into PCA components.**

```python
class PositionalDecoder(nn.Module):
    """
    Maps PCA components (which are position-agnostic) back to
    position-aware representations for the LLM.
    """

    def __init__(self, d_model=4096, n_components=128, max_positions=300):
        super().__init__()
        self.n_components = n_components
        self.max_positions = max_positions

        # Learned position embedding table
        self.position_embeddings = nn.Embedding(max_positions, d_model)

        # Network to predict "which positions contribute to this component"
        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_positions),
            nn.Softmax(dim=-1)  # Output: probability distribution over positions
        )

        # Refinement MLP
        self.refiner = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, pca_components, return_position_scores=False):
        """
        Args:
            pca_components: [batch, n_components=128, d_model=4096]

        Returns:
            position_aware_embeddings: [batch, 128, 4096]
            position_scores: [batch, 128, 300] (if return_position_scores=True)
        """
        batch_size, n_comp, d_model = pca_components.shape

        # For each component, predict which original positions it represents
        position_scores = self.position_predictor(pca_components)  # [batch, 128, 300]

        # Compute expected position for each component
        # This gives us a "virtual position" in [0, 299] for each component
        position_indices = torch.arange(self.max_positions, device=pca_components.device)
        expected_positions = (position_scores * position_indices).sum(dim=-1)  # [batch, 128]

        # Get position embeddings for expected positions
        # Use differentiable interpolation between integer positions
        expected_positions_floor = expected_positions.floor().long()
        expected_positions_ceil = expected_positions.ceil().long()
        frac = expected_positions - expected_positions_floor.float()

        pos_emb_floor = self.position_embeddings(expected_positions_floor)  # [batch, 128, d_model]
        pos_emb_ceil = self.position_embeddings(expected_positions_ceil)
        pos_emb = pos_emb_floor * (1 - frac.unsqueeze(-1)) + pos_emb_ceil * frac.unsqueeze(-1)

        # Add positional information to PCA components
        position_aware = pca_components + pos_emb

        # Refine
        position_aware = self.refiner(position_aware)

        if return_position_scores:
            return position_aware, position_scores
        return position_aware
```

#### 3. Training Objectives

**Multi-task training to ensure position information is preserved:**

```python
def compute_loss(model, pca_components, answer_ids, original_embeddings, original_positions):
    """
    Args:
        pca_components: [batch, 128, 4096] - sequence-PCA output
        answer_ids: [batch, ans_len] - gold answer tokens
        original_embeddings: [batch, 300, 4096] - original pre-PCA embeddings
        original_positions: [batch, 300] - which original positions are valid

    Returns:
        total_loss, loss_dict
    """

    # 1. Main generation loss
    position_aware, position_scores = decoder(pca_components, return_position_scores=True)

    answer_embeds = model.get_input_embeddings()(answer_ids[:, :-1])
    inputs_embeds = torch.cat([position_aware, answer_embeds], dim=1)

    # Standard next-token prediction
    outputs = model(inputs_embeds=inputs_embeds, labels=answer_ids)
    loss_generation = outputs.loss

    # 2. Position reconstruction loss
    # Penalize if position_scores don't align with where information actually came from
    # This requires tracking which original positions contain the answer
    # (This is computed during data preparation)

    # For each PCA component, we want its position_scores to be high
    # at positions that contributed to it via PCA

    # Get PCA contribution matrix (which original positions contributed to each component)
    # This can be computed from PCA components matrix
    pca_contributions = get_pca_contributions(pca_components, original_embeddings)
    # Shape: [batch, 128, 300] - shows which original positions influenced each component

    # Cross-entropy between predicted positions and PCA-implied positions
    loss_position = F.kl_div(
        position_scores.log(),
        pca_contributions,
        reduction='batchmean'
    )

    # 3. Reconstruction auxiliary loss (optional)
    # Can we reconstruct original embeddings from PCA + position decoder?
    # This ensures no information loss

    # Use position_scores to reconstruct original sequence
    reconstructed = position_scores @ original_embeddings  # [batch, 128, 300] @ [batch, 300, 4096]
    # → [batch, 128, 4096]

    loss_reconstruction = F.mse_loss(reconstructed, pca_components)

    # Total loss
    total_loss = (
        loss_generation +
        0.1 * loss_position +
        0.05 * loss_reconstruction
    )

    return total_loss, {
        'generation': loss_generation.item(),
        'position': loss_position.item(),
        'reconstruction': loss_reconstruction.item(),
    }
```

#### 4. Computing PCA Contributions (for position loss)

```python
def get_pca_contributions(pca_components, original_embeddings, pca_model):
    """
    Compute which original positions contributed to each PCA component.

    Args:
        pca_components: [batch, 128, 4096] - compressed
        original_embeddings: [batch, 300, 4096] - original
        pca_model: fitted PCA object

    Returns:
        contributions: [batch, 128, 300] - contribution weights
    """
    batch_size = original_embeddings.shape[0]
    contributions_list = []

    for b in range(batch_size):
        # Get this example's embeddings
        orig = original_embeddings[b]  # [300, 4096]

        # Transpose for sequence-PCA
        orig_T = orig.T  # [4096, 300]

        # Project onto PCA components
        # PCA components are [128, 4096] in the feature space
        # We want to know: for each of 128 components,
        # which of the 300 original positions contributed most?

        # The PCA components matrix tells us the weights
        components = torch.from_numpy(pca_model.components_).to(orig.device)  # [128, 4096]

        # For each component, compute how much each original position contributed
        # Component i's contribution from position j is:
        # sum over features k: components[i,k] * original[j,k]

        contrib = torch.matmul(components, orig.T)  # [128, 4096] @ [4096, 300] = [128, 300]

        # Normalize to get contribution weights (softmax to make it a distribution)
        contrib_probs = F.softmax(contrib, dim=-1)  # [128, 300]

        contributions_list.append(contrib_probs)

    contributions = torch.stack(contributions_list)  # [batch, 128, 300]
    return contributions
```

## Addressing Codex's Concerns

### Concern 1: "PCA is unsupervised and won't encode 'answer is here' structure"

**Solution:**
- PCA is unsupervised but we **supervise the decoder**
- The position predictor is trained with explicit position reconstruction loss
- The generation loss provides gradient signal for position-aware representations
- Together, these teach the system to preserve positional information through the bottleneck

### Concern 2: "Without positional decoding, LLM left guessing where information came from"

**Solution:**
- The Positional Decoder explicitly reconstructs position information
- Position embeddings are added to each PCA component based on its "expected position"
- The LLM receives [128, 4096] embeddings where each has position information
- Similar to how standard transformers add positional encodings

### Concern 3: "Need a way to map PCA components back to positional slots"

**Solution:**
- The `position_predictor` network learns this mapping
- Outputs a distribution over original positions [0-299] for each component
- Uses this to inject appropriate positional embeddings
- Position reconstruction loss ensures this mapping is accurate

## Expected Results

### Best Case:
- **F1: 15-25%** at 2.3× compression (300 → 128)
- Better than naive pooling (which gets 0%) because:
  - Full 4096 feature dimension preserved
  - Learned position injection
  - PCA finds optimal temporal patterns

### Realistic Case:
- **F1: 5-15%**
- Better than nothing, shows position preservation helps
- May need more sophisticated position decoder

### Worst Case:
- **F1: 0-5%**
- Position information still too scrambled
- PCA's linear assumption too limiting for QA

## Implementation Plan

### Phase 1: Basic Sequence-PCA (1 week)
1. Implement sequence-PCA fitting and transform
2. Train with simple position embedding addition
3. Measure baseline performance

### Phase 2: Positional Decoder (1 week)
4. Implement position predictor network
5. Add position reconstruction loss
6. Compare to Phase 1 baseline

### Phase 3: Refinements (1 week)
7. Try different decoder architectures
8. Tune loss weights
9. Experiment with PCA component counts (64, 96, 128, 192)

### Phase 4: Analysis (1 week)
10. Visualize which positions map to which components
11. Analyze failure modes
12. Compare to Experiment 1 results

## Key Hyperparameters to Sweep

1. **PCA components:** 64, 96, 128, 192, 256
2. **Position predictor architecture:**
   - Shallow MLP (current design)
   - Attention-based
   - Transformer block
3. **Loss weights:**
   - Position loss: [0.05, 0.1, 0.2]
   - Reconstruction loss: [0.01, 0.05, 0.1]
4. **Position embedding strategy:**
   - Learned embeddings (current)
   - Sinusoidal (RoPE-style)
   - Hybrid

## Comparison to Experiment 1

| Aspect | Experiment 1 (Pooling + LoRA) | Experiment 2 (Sequence-PCA) |
|--------|------------------------------|----------------------------|
| **Compression method** | Learned pooling (attention/conv) | PCA (unsupervised) |
| **What's preserved** | Local structure (via pooling windows) | Global temporal patterns |
| **Positional info** | Via positional encodings in pooling | Via learned positional decoder |
| **Adaptation** | LoRA in LLM | Positional decoder (no LLM changes) |
| **Expected F1** | 10-30% (more promising) | 5-20% (novel but risky) |
| **Novelty** | Moderate (LoRA + pooling known) | **High** (sequence-PCA unprecedented) |

## Recommendation

**Try Experiment 1 first** (learned pooling + LoRA) because:
- More likely to work (10-30% F1 expected)
- Faster to iterate
- Better understood

**Then try Experiment 2** if:
- Experiment 1 succeeds → Experiment 2 offers novel alternative
- Experiment 1 fails → Experiment 2's different approach might work
- We want to publish novelty → Sequence-PCA is unprecedented

## References & Related Work

1. **PCA for dimensionality reduction** - Standard ML technique, but typically on features not sequence
2. **Positional encodings in Transformers** - Our position decoder draws from this
3. **Gist Tokens** (Mu et al. 2023) - Sequence compression via learned pooling (Experiment 1 is similar)
4. **Our innovation** - PCA on sequence dimension + learned position reconstruction (completely novel)

---

**Status:** Design complete, ready for implementation after Experiment 1 results.
