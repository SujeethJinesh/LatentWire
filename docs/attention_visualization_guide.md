# Attention Visualization Guide

Comprehensive guide to extracting and visualizing attention patterns in transformers, with specific focus on analyzing attention to soft tokens in LatentWire.

**Last Updated**: January 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Extracting Attention Weights](#extracting-attention-weights)
3. [Visualization Techniques](#visualization-techniques)
4. [Measuring Attention to Soft Tokens](#measuring-attention-to-soft-tokens)
5. [Standard Libraries and Tools](#standard-libraries-and-tools)
6. [LatentWire-Specific Analysis](#latentwire-specific-analysis)
7. [References](#references)

---

## Overview

Attention patterns in transformers reveal how the model weighs different input tokens when making predictions. For LatentWire, understanding attention to soft tokens (learned latent embeddings) is critical for diagnosing why the model may struggle with first-token generation.

**Key Questions for LatentWire**:
- How much attention do decoder tokens pay to soft tokens vs. anchor text?
- Do certain layers/heads specialize in attending to soft tokens?
- Are attention patterns different between successful and failed generations?
- How does attention evolve during autoregressive generation?

---

## Extracting Attention Weights

### HuggingFace Transformers API

All HuggingFace models support attention extraction via the `output_attentions` parameter:

```python
from transformers import AutoModel, AutoTokenizer

# Load model with attention output enabled
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare input
inputs = tokenizer("The cat sat on the mat", return_tensors="pt")

# Forward pass
outputs = model(**inputs, output_attentions=True)

# Extract attentions
attentions = outputs.attentions  # Tuple of tensors, one per layer
```

**Attention Format**:
- `attentions`: Tuple of length `num_layers`
- Each element: `[batch_size, num_heads, seq_len, seq_len]`
- `attentions[layer][batch, head, query_pos, key_pos]` = attention weight from `query_pos` to `key_pos`

### For LatentWire (Llama/Qwen Models)

```python
# In LMWrapper.forward_with_prefix_loss or similar
out = self.model(
    inputs_embeds=inputs_embeds,
    attention_mask=attn_mask,
    output_attentions=True,  # Enable attention extraction
    return_dict=True,
)

# Access attentions
attentions = out.attentions  # Tuple of [B, num_heads, seq_len, seq_len]
```

**Important**: Attention extraction adds memory overhead. For Llama-3.1-8B with 32 layers and 32 heads:
- Single forward pass: ~4GB extra VRAM for attention storage
- Use sparingly for diagnostic analysis, not training

---

## Visualization Techniques

### 1. Attention Heatmaps (Seaborn/Matplotlib)

**Simple heatmap for single head**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract attention for layer 15, head 8, batch 0
attn = attentions[15][0, 8]  # [seq_len, seq_len]

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    attn.cpu().numpy(),
    cmap="viridis",
    vmin=0,
    vmax=1.0,
    cbar_kws={"label": "Attention Weight"},
)
plt.xlabel("Key Position (attending TO)")
plt.ylabel("Query Position (attending FROM)")
plt.title("Attention Heatmap - Layer 15, Head 8")
plt.savefig("attention_heatmap.png", dpi=150)
```

**With token labels**:

```python
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

sns.heatmap(
    attn.cpu().numpy(),
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
)
plt.xticks(rotation=45, ha='right')
```

**Highlighting soft token regions**:

```python
# Soft tokens are at positions [0:M]
M = 32  # latent_len

plt.axvline(x=M, color='red', linestyle='--', linewidth=2, label='Soft tokens boundary')
plt.axhline(y=M, color='red', linestyle='--', linewidth=2)
```

### 2. BertViz Interactive Visualizations

BertViz provides three complementary views:

**Installation**:
```bash
pip install bertviz
```

**Usage**:

```python
from bertviz import head_view, model_view, neuron_view

# Prepare attention tensors (list or tuple)
attention_tensors = [attn.cpu() for attn in attentions]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Model view: Bird's eye view of all layers and heads
model_view(attention_tensors, tokens)

# Head view: Detailed view of specific heads
head_view(attention_tensors, tokens)

# Neuron view: Individual query/key neurons (BERT/GPT-2/RoBERTa only)
neuron_view(attention_tensors, tokens)
```

**Customization**:

```python
# Display only specific layers
model_view(attention_tensors, tokens, include_layers=[5, 10, 15, 20, 25, 30])

# Dark/light mode
model_view(attention_tensors, tokens, display_mode="light")

# Save to HTML
model_view(attention_tensors, tokens, html_action='save')
```

**Note**: BertViz works best in Jupyter notebooks for interactive exploration. For batch scripts, save to HTML.

### 3. Aggregate Statistics Across Layers

**Plot mean attention by layer**:

```python
# Compute mean attention to first M tokens (soft tokens)
M = 32
means = []
stds = []

for layer_attn in attentions:
    # Focus on last position (first token generation)
    last_pos = layer_attn[:, :, -1, :]  # [B, num_heads, seq_len]
    soft_attn = last_pos[:, :, :M].sum(dim=-1)  # [B, num_heads]

    means.append(soft_attn.mean().item())
    stds.append(soft_attn.std().item())

plt.plot(range(len(means)), means, marker='o')
plt.fill_between(range(len(means)),
                 [m - s for m, s in zip(means, stds)],
                 [m + s for m, s in zip(means, stds)],
                 alpha=0.3)
plt.xlabel('Layer Index')
plt.ylabel('Mean Attention to Soft Tokens')
plt.title('Attention to Soft Tokens by Layer')
plt.grid(True, alpha=0.3)
plt.savefig('attention_by_layer.png', dpi=150)
```

---

## Measuring Attention to Soft Tokens

### Key Metrics

For LatentWire, we want to measure:

1. **Total attention to soft tokens**: Sum of attention weights from position T to positions [0:M]
2. **Per-head specialization**: Which heads attend strongly to soft tokens?
3. **Layer progression**: How does attention evolve through layers?
4. **Comparison to anchor**: Do models prefer anchor text over soft tokens?

### Implementation

```python
def compute_soft_token_attention_stats(
    attentions: tuple,
    prefix_len: int,
    anchor_len: int = 0,
):
    """
    Compute statistics about attention to soft tokens.

    Args:
        attentions: Tuple of [B, num_heads, seq_len, seq_len] tensors
        prefix_len: Number of soft tokens (M)
        anchor_len: Number of anchor text tokens

    Returns:
        Dictionary with per-layer and per-head statistics
    """
    stats = {"per_layer": []}

    for layer_idx, attn in enumerate(attentions):
        # Focus on last position (where we're generating first token)
        last_pos_attn = attn[:, :, -1, :]  # [B, num_heads, seq_len]

        # Attention to soft tokens [0:prefix_len]
        soft_attn = last_pos_attn[:, :, :prefix_len].sum(dim=-1)  # [B, num_heads]

        # Attention to anchor tokens [prefix_len:prefix_len+anchor_len]
        if anchor_len > 0:
            anchor_attn = last_pos_attn[:, :, prefix_len:prefix_len+anchor_len].sum(dim=-1)
        else:
            anchor_attn = torch.zeros_like(soft_attn)

        stats["per_layer"].append({
            "layer": layer_idx,
            "soft_mean": soft_attn.mean().item(),
            "soft_std": soft_attn.std().item(),
            "anchor_mean": anchor_attn.mean().item(),
            "per_head_soft": soft_attn.mean(dim=0).tolist(),  # Average over batch
        })

    return stats
```

### Interpretation Guidelines

**Healthy attention patterns** (expected for well-conditioned model):
- Early layers: ~0.3-0.5 total attention to soft tokens
- Middle layers: ~0.4-0.7 total attention to soft tokens
- Late layers: ~0.2-0.4 (as model integrates information)
- Anchor text: ~0.1-0.3 (providing context)

**Problematic patterns** (may indicate issues):
- **Attention collapse**: All layers < 0.1 to soft tokens → model ignoring latents
- **Anchor dominance**: Anchor > 0.7, soft < 0.1 → model relies only on anchor
- **Head uniformity**: All heads identical → lack of specialization
- **Layer uniformity**: All layers identical → shallow processing

---

## Standard Libraries and Tools

### 1. Matplotlib + Seaborn (Recommended)

**Pros**:
- Full control over visualization
- Integrates with scientific workflows
- Publication-quality figures
- No JavaScript dependencies

**Cons**:
- Not interactive
- Requires manual layout for complex visualizations

**Best for**: Batch analysis, paper figures, automated reporting

### 2. BertViz

**Pros**:
- Beautiful interactive visualizations
- Multiple complementary views
- Widely used in research
- Easy to use in Jupyter

**Cons**:
- Limited to BERT/GPT-2/RoBERTa for neuron view
- Requires HTML rendering
- Less customization than matplotlib

**Best for**: Interactive exploration, presentations, debugging

### 3. Attention Analysis Libraries

**exBERT** (https://exbert.net/):
- Web-based attention visualization
- Corpus-level analysis
- Pre-computed for popular models

**LIT (Language Interpretability Tool)** (https://pair-code.github.io/lit/):
- Comprehensive model analysis
- Includes attention visualization
- Interactive comparisons

**Captum** (https://captum.ai/):
- Attribution methods beyond attention
- Integrated gradients, saliency
- PyTorch-native

---

## LatentWire-Specific Analysis

### Diagnostic Questions

**Q1: Why is FirstTok@1 accuracy low (5-7%)?**

**Analysis approach**:
1. Extract attention for correct vs. incorrect examples
2. Compare attention to soft tokens between groups
3. Hypothesis: Incorrect examples may show attention collapse

**Code**:
```python
# Group examples by correctness
correct_attns = []
incorrect_attns = []

for ex, pred, gold in examples:
    attn_data = extract_attention(ex)
    if pred == gold:
        correct_attns.append(attn_data)
    else:
        incorrect_attns.append(attn_data)

# Compare mean attention to soft tokens
correct_soft_attn = compute_mean_soft_attention(correct_attns)
incorrect_soft_attn = compute_mean_soft_attention(incorrect_attns)

# Visualize difference
plt.plot(correct_soft_attn, label='Correct predictions')
plt.plot(incorrect_soft_attn, label='Incorrect predictions')
plt.legend()
```

**Q2: Are soft tokens informative, or does the model rely on anchor text?**

**Analysis approach**:
1. Measure attention to soft tokens vs. anchor text
2. If anchor >> soft, model is "cheating" via anchor
3. Solution: Reduce anchor text weight or improve soft token quality

**Q3: Do certain layers specialize in soft token processing?**

**Analysis approach**:
1. Compute attention to soft tokens per layer
2. Look for peaks (e.g., layer 15-20 might specialize)
3. Use this to inform architectural changes (e.g., cross-attention at peak layers)

### Integration with Training

**Logging attention statistics during training**:

```python
# In training loop, every N steps
if step % log_interval == 0:
    with torch.no_grad():
        # Sample batch
        attn_data = extract_attention_for_batch(batch)
        stats = compute_soft_token_attention_stats(attn_data)

        # Log to wandb/tensorboard
        wandb.log({
            "attn/soft_mean_layer0": stats["per_layer"][0]["soft_mean"],
            "attn/soft_mean_layer15": stats["per_layer"][15]["soft_mean"],
            "attn/soft_mean_layer31": stats["per_layer"][31]["soft_mean"],
        })
```

**Attention-based early stopping**:

If attention to soft tokens drops below threshold, training may have collapsed:

```python
if stats["per_layer"][15]["soft_mean"] < 0.05:
    print("WARNING: Attention collapse detected!")
    # Save checkpoint and investigate
```

---

## References

### Research Papers

1. **Visualizing Attention in Transformer-Based Language Representation Models** (Vig, 2019)
   - Introduced BertViz
   - [arXiv:1904.02679](https://arxiv.org/abs/1904.02679)

2. **On the Role of Attention in Prompt-tuning** (Oymak et al., ICML 2023)
   - Formal analysis of attention in soft prompting
   - Key insight: Soft prompts bias attention heads but cannot fundamentally alter learned patterns
   - [ICML 2023](https://proceedings.mlr.press/v202/oymak23a/oymak23a.pdf)

3. **Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs** (2025)
   - ID-SPAM: Input-dependent soft prompts with attention analysis
   - [arXiv:2506.05629](https://arxiv.org/html/2506.05629v1)

4. **Understanding Prompt Tuning and In-Context Learning via Meta-Learning** (2025)
   - Meta-learning perspective on prompt tuning
   - Attention pattern analysis
   - [arXiv:2505.17010](https://arxiv.org/html/2505.17010v1)

### Documentation and Tools

- **HuggingFace Model Outputs**: [https://huggingface.co/docs/transformers/en/main_classes/output](https://huggingface.co/docs/transformers/en/main_classes/output)
- **BertViz GitHub**: [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
- **How to Visualize Model Internals in HuggingFace**: [https://www.kdnuggets.com/how-to-visualize-model-internals-and-attention-in-hugging-face-transformers](https://www.kdnuggets.com/how-to-visualize-model-internals-and-attention-in-hugging-face-transformers)

### Key Insights from Literature

1. **Soft prompts bias attention heads** but cannot create entirely new attention patterns (Oymak et al., 2023)
   - Implication: LatentWire must leverage existing model capabilities, not create new ones
   - Solution: Ensure soft tokens encode information in a way that existing attention patterns can utilize

2. **Attention sinks** receive disproportionate attention despite low semantic relevance
   - Common in first token, special tokens
   - May explain why models attend to anchor text more than soft tokens

3. **Per-head specialization** is critical for effective soft prompting
   - Different heads should attend to different aspects of soft tokens
   - Uniformity across heads suggests under-utilization

---

## Quick Start

### Run the demo script

```bash
# Simple demo with BERT
python scripts/attention_extraction_demo.py --model bert-base-uncased

# With BertViz visualization
python scripts/attention_extraction_demo.py --model bert-base-uncased --use_bertviz
```

### Analyze a LatentWire checkpoint

```bash
# Full analysis
python scripts/analyze_attention.py \
    --ckpt runs/8B_clean_answer_ftce/epoch23 \
    --samples 50 \
    --dataset squad \
    --output_dir runs/attention_analysis
```

### Integrate into eval script

```python
# In latentwire/eval.py, add attention extraction option
if args.extract_attention:
    extractor = AttentionExtractor(wrapper)
    attn_data = extractor.extract_first_token_attention(
        prefix_embeds,
        anchor_token_text=anchor_text,
        append_bos_after_prefix=True,
    )
    # Save for later analysis
    torch.save(attn_data, f"{output_dir}/attention_ex{idx}.pt")
```

---

## Summary

**Key takeaways**:

1. **Extraction**: Use `output_attentions=True` in HuggingFace models
2. **Visualization**: Matplotlib/Seaborn for batch analysis, BertViz for interactive exploration
3. **Metrics**: Focus on attention to soft tokens vs. anchor text, per-layer and per-head
4. **Diagnosis**: Compare correct vs. incorrect examples, look for attention collapse or anchor dominance
5. **Integration**: Log attention statistics during training, use for early stopping

**For LatentWire**:
- Attention analysis can reveal why FirstTok@1 is low
- Key hypothesis: Model may be ignoring soft tokens in favor of anchor text
- Solution: Improve soft token quality, reduce anchor text weight, or add architectural biases (cross-attention)
