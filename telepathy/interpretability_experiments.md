# Interpretability Experiments for Telepathy Paper

## Executive Summary

This document proposes 5 concrete interpretability experiments to strengthen the Telepathy paper. Each experiment reveals *why* the system works, not just *that* it works. Total estimated time: 5-7 days with parallel execution on 4 H100s.

---

## Experiment 1: Cross-Attention Pattern Analysis
**Goal**: Prove the Perceiver learns semantic alignment, not just statistical matching.

### Implementation Steps
1. **Hook Installation** (2 hours)
   - Register forward hooks on all PerceiverResampler cross-attention layers
   - Capture attention weights: `[batch, heads, soft_tokens, source_tokens]`
   - Save patterns for 1000 test samples across different tasks

2. **Semantic Grouping Analysis** (4 hours)
   - Group source tokens by linguistic category (nouns, verbs, numbers, entities)
   - Compute attention entropy per soft token: `H = -Σ p_i log(p_i)`
   - Measure specialization: which soft tokens attend to which token types?

3. **Visualization** (2 hours)
   - Generate heatmaps showing attention patterns for key examples
   - Create averaged attention maps per linguistic category
   - Plot attention entropy distribution across soft tokens

### Expected Results
- **Hypothesis**: Soft tokens specialize - some focus on entities, others on relations
- **Success Metric**: >60% of soft tokens show <0.5 entropy (focused attention)
- **What It Proves**: The bridge learns linguistic structure, not random compression

### Code Skeleton
```python
def analyze_attention_patterns(bridge, test_loader):
    attention_maps = []

    def hook_fn(module, input, output):
        # output[1] contains attention weights
        attention_maps.append(output[1].detach())

    # Register hooks on cross_attn modules
    hooks = []
    for layer in bridge.perceiver.layers:
        h = layer["cross_attn"].register_forward_hook(hook_fn)
        hooks.append(h)

    # Collect attention patterns
    with torch.no_grad():
        for batch in test_loader:
            _ = bridge(batch["hidden_states"])

    # Analyze specialization
    entropy = compute_attention_entropy(attention_maps)
    specialization = identify_token_roles(attention_maps, batch["tokens"])

    return entropy, specialization
```

**Time Estimate**: 8 hours total (can run overnight)

---

## Experiment 2: Soft Token Probing Study
**Goal**: Extract interpretable features from soft tokens via linear probes.

### Implementation Steps
1. **Feature Collection** (3 hours)
   - Generate soft tokens for 10,000 samples from AG News
   - Label each sample with multiple attributes:
     - Task label (4 categories)
     - Sentiment (positive/negative)
     - Length bucket (short/medium/long)
     - Contains numbers (yes/no)
     - Contains named entities (yes/no)

2. **Probe Training** (4 hours)
   - Train linear classifiers on frozen soft tokens for each attribute
   - Use different pooling strategies: mean, max, first token
   - Track probe accuracy vs random baseline

3. **Feature Importance** (2 hours)
   - Apply SHAP/GradCAM to identify which soft tokens encode which features
   - Compute mutual information between soft tokens and attributes
   - Perform ablation: mask individual soft tokens and measure accuracy drop

### Expected Results
- **Task Probe**: 80-90% accuracy (soft tokens encode task information)
- **Sentiment Probe**: 60-70% accuracy (partial encoding of tone)
- **Length Probe**: 85-95% accuracy (strong geometric encoding)
- **What It Proves**: Soft tokens form a compositional code with interpretable features

### Code Skeleton
```python
class SoftTokenProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.probe = nn.Linear(input_dim, num_classes)

    def forward(self, soft_tokens, pooling="mean"):
        if pooling == "mean":
            features = soft_tokens.mean(dim=1)
        elif pooling == "max":
            features = soft_tokens.max(dim=1)[0]
        elif pooling == "first":
            features = soft_tokens[:, 0]
        return self.probe(features)

def train_probes(soft_tokens, labels_dict):
    results = {}
    for attribute, labels in labels_dict.items():
        probe = SoftTokenProbe(soft_tokens.shape[-1], len(set(labels)))
        # Training loop...
        accuracy = evaluate_probe(probe, soft_tokens, labels)
        results[attribute] = accuracy
    return results
```

**Time Estimate**: 9 hours total (parallelizable across attributes)

---

## Experiment 3: Progressive Ablation Trajectories
**Goal**: Understand graceful degradation and component importance.

### Implementation Steps
1. **Component Ablations** (6 hours)
   - Systematically disable components and measure performance:
     - Remove StatisticalNormalizer → raw hidden states
     - Reduce Perceiver depth: 4 → 3 → 2 → 1 → 0 layers
     - Reduce soft tokens: 64 → 32 → 16 → 8 → 4 → 2 → 1
     - Disable self-attention in Perceiver (cross-attention only)
     - Remove FFN layers in Perceiver

2. **Noise Injection** (3 hours)
   - Add Gaussian noise at different scales to:
     - Source hidden states (σ = 0.01, 0.1, 0.5, 1.0)
     - Soft tokens before projection (σ = 0.01, 0.1, 0.5, 1.0)
     - Statistical normalizer parameters (±10%, ±50%, ±100%)

3. **Trajectory Plotting** (2 hours)
   - Plot accuracy vs ablation severity
   - Create phase diagrams showing failure modes
   - Identify critical thresholds where performance collapses

### Expected Results
- **Normalizer**: Removing causes immediate 80% accuracy drop
- **Perceiver Depth**: Each layer adds ~15% accuracy
- **Soft Tokens**: Performance plateaus at 32 tokens (diminishing returns)
- **What It Proves**: Architecture choices are necessary, not arbitrary

### Code Skeleton
```python
def ablation_study(base_bridge, test_loader):
    results = {}

    # Test reducing Perceiver depth
    for depth in [4, 3, 2, 1, 0]:
        bridge = create_bridge_variant(depth=depth)
        acc = evaluate(bridge, test_loader)
        results[f"depth_{depth}"] = acc

    # Test reducing soft tokens
    for num_tokens in [64, 32, 16, 8, 4, 2, 1]:
        bridge = create_bridge_variant(num_tokens=num_tokens)
        acc = evaluate(bridge, test_loader)
        results[f"tokens_{num_tokens}"] = acc

    # Test noise robustness
    for noise_std in [0.01, 0.1, 0.5, 1.0]:
        acc = evaluate_with_noise(base_bridge, test_loader, noise_std)
        results[f"noise_{noise_std}"] = acc

    return results
```

**Time Estimate**: 11 hours total (highly parallelizable)

---

## Experiment 4: Gradient-Based Feature Attribution
**Goal**: Identify which source tokens most influence target predictions.

### Implementation Steps
1. **Integrated Gradients** (4 hours)
   - Implement IG from source tokens to target logits
   - Create attribution maps for 1000 test samples
   - Compare attributions for correct vs incorrect predictions

2. **Attention Rollout** (3 hours)
   - Compute attention flow from source to soft tokens
   - Track information propagation through Perceiver layers
   - Identify "information highways" in the architecture

3. **Counterfactual Analysis** (4 hours)
   - Mask high-attribution tokens and measure accuracy drop
   - Replace tokens with synonyms and measure stability
   - Identify minimal token sets that preserve accuracy

### Expected Results
- **Attribution Concentration**: 80% of attribution in 20% of tokens
- **Correct vs Incorrect**: Incorrect predictions show dispersed attribution
- **What It Proves**: The bridge learns to focus on task-relevant information

### Code Skeleton
```python
def integrated_gradients(bridge, source_hidden, baseline, target_idx, steps=50):
    alphas = torch.linspace(0, 1, steps).to(source_hidden.device)
    gradients = []

    for alpha in alphas:
        interpolated = baseline + alpha * (source_hidden - baseline)
        interpolated.requires_grad = True

        soft_tokens = bridge(interpolated)
        output = target_model(inputs_embeds=soft_tokens)
        loss = output.logits[0, 0, target_idx]

        loss.backward()
        gradients.append(interpolated.grad.clone())

    avg_gradients = torch.stack(gradients).mean(dim=0)
    attribution = (source_hidden - baseline) * avg_gradients
    return attribution
```

**Time Estimate**: 11 hours total

---

## Experiment 5: Causal Intervention Studies
**Goal**: Prove soft tokens encode causal features, not just correlations.

### Implementation Steps
1. **Soft Token Surgery** (4 hours)
   - Swap soft tokens between examples of different classes
   - Interpolate between soft tokens: `z_interp = α*z_A + (1-α)*z_B`
   - Measure how predictions change with interpolation factor α

2. **Concept Injection** (4 hours)
   - Train "concept vectors" for specific attributes (sports, politics, etc.)
   - Add concept vectors to soft tokens: `z' = z + β*v_concept`
   - Measure if target model generates concept-related text

3. **Adversarial Steering** (3 hours)
   - Use PGD to find minimal soft token perturbations that:
     - Change predicted class
     - Maintain fluency (low perplexity)
     - Preserve most information (small L2 norm)

### Expected Results
- **Token Swapping**: 70% of swaps change prediction to donor class
- **Interpolation**: Smooth transition between classes at α ≈ 0.5
- **Concept Injection**: 60% success rate in steering generation
- **What It Proves**: Soft tokens causally control target behavior

### Code Skeleton
```python
def soft_token_interpolation(bridge, example_a, example_b, alphas):
    results = []

    # Get soft tokens for both examples
    z_a = bridge(example_a["hidden_states"])
    z_b = bridge(example_b["hidden_states"])

    for alpha in alphas:
        # Interpolate
        z_interp = alpha * z_a + (1 - alpha) * z_b

        # Generate from interpolated tokens
        output = target_model.generate(inputs_embeds=z_interp)

        # Measure class prediction
        pred_class = classify(output)
        results.append({
            "alpha": alpha,
            "pred_class": pred_class,
            "closer_to_a": pred_class == example_a["label"]
        })

    return results

def concept_injection(bridge, soft_tokens, concept_vector, betas):
    results = []

    for beta in betas:
        # Inject concept
        z_modified = soft_tokens + beta * concept_vector

        # Generate text
        output = target_model.generate(inputs_embeds=z_modified)

        # Check if concept appears
        contains_concept = check_concept_presence(output)
        results.append({
            "beta": beta,
            "contains_concept": contains_concept
        })

    return results
```

**Time Estimate**: 11 hours total

---

## Implementation Priority & Timeline

### Day 1-2: Infrastructure Setup
- Set up experiment framework with proper logging
- Implement hook system for attention capture
- Create visualization utilities

### Day 3-4: Core Experiments
- **Parallel Track A** (2 H100s): Attention Analysis + Soft Token Probing
- **Parallel Track B** (2 H100s): Ablation Studies + Feature Attribution

### Day 5: Causal Studies
- Run intervention experiments
- Generate all visualizations

### Day 6-7: Analysis & Writing
- Statistical significance testing
- Create publication-quality figures
- Write interpretability section for paper

---

## Expected Impact on Paper

These experiments will add a new **"Interpretability Analysis"** section showing:

1. **Attention Specialization** (Fig 4): Heatmaps showing soft tokens learn roles
2. **Feature Probing** (Table 3): Linear probe accuracies for various attributes
3. **Ablation Curves** (Fig 5): Graceful degradation with component removal
4. **Attribution Maps** (Fig 6): Which source tokens matter most
5. **Causal Control** (Fig 7): Interpolation and steering demonstrations

**Key Insight for Reviewers**: The bridge doesn't just compress—it learns a structured, interpretable protocol for cross-model communication. Soft tokens form a "lingua franca" that captures semantic content while discarding model-specific encoding details.

---

## Resource Requirements

- **Compute**: 4 H100 GPUs for 5-7 days
- **Storage**: ~50GB for attention maps and intermediate results
- **Memory**: 40GB GPU RAM per experiment (can reduce with smaller batches)

## Success Criteria

The experiments succeed if they demonstrate:
1. Soft tokens are **interpretable** (>70% probe accuracy)
2. Attention is **structured** (clear specialization patterns)
3. Ablations show **graceful degradation** (not catastrophic failure)
4. Interventions **causally** affect outputs (not just correlation)
5. Results are **reproducible** (low variance across seeds)

## Risk Mitigation

- **Risk**: Experiments show random/uninterpretable patterns
- **Mitigation**: Focus on relative comparisons (trained vs random baseline)

- **Risk**: Computational requirements too high
- **Mitigation**: Start with smaller sample sizes, scale up if promising

- **Risk**: Results don't support main claims
- **Mitigation**: Frame as "understanding limitations" rather than failure

---

## Conclusion

These experiments transform Telepathy from a "black box that works" to an "understood system with clear mechanisms." They provide the interpretability that reviewers expect from a top-tier venue, demonstrating not just empirical success but also scientific understanding of *why* latent communication succeeds.