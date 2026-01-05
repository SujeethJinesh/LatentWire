# Linear Probe Methodology for LLM Hidden States

This document describes the standard methodology for linear probe baselines on LLM hidden states, based on recent research (2025-2026) and best practices from the machine learning community.

## Overview

Linear probes are lightweight classifiers trained on frozen LLM hidden states to predict task labels. They measure how much task-relevant information is linearly accessible in the model's internal representations.

## Implementation: `telepathy/linear_probe_hidden_states.py`

### 1. Hidden State Extraction

**Standard Method (HuggingFace Transformers):**

```python
# Enable hidden states output
outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# outputs.hidden_states is a tuple:
# (embeddings, layer1, layer2, ..., layerN)
hidden = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
```

**Layer Selection:**
- `layer_idx=0`: Initial embeddings (before any transformer layers)
- `layer_idx=1` to `layer_idx=N-1`: Intermediate transformer layers
- `layer_idx=N`: Final layer output (same as `last_hidden_state`)

**Common Practice:** Middle layers (around layer 12-20 for 32-layer models) often yield best results for downstream tasks.

**Pooling Methods:**
- `last_token`: Use final non-padding token (best for autoregressive models like Llama)
- `mean`: Average over all non-padding tokens
- `first_token`/`cls`: Use first token (for BERT-style models)

### 2. Feature Normalization

**Why normalize?** Logistic regression is sensitive to feature scale. Normalization improves numerical stability and convergence.

**Standard Methods:**

1. **L2 Normalization (Recommended for hidden states):**
   ```python
   from sklearn.preprocessing import normalize
   X_normalized = normalize(X, norm='l2', axis=1)  # ||x|| = 1 per example
   ```

2. **Standardization (Z-score):**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)  # Zero mean, unit variance per feature
   ```

**Default:** L2 normalization (matches common practice in NLP research)

### 3. Linear Probe Training

**Standard Setup (scikit-learn LogisticRegression):**

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    C=1.0,              # Inverse regularization (higher = less regularization)
    max_iter=1000,      # Maximum iterations
    solver='lbfgs',     # Robust default solver
    multi_class='auto', # Handles binary and multi-class
    random_state=42,    # Reproducibility
    n_jobs=-1,          # Parallel processing
)
```

**Key Parameters:**
- **C (regularization)**: Default C=1.0. L2 regularization is applied by default in scikit-learn.
  - Higher C = less regularization
  - Lower C = more regularization
  - Typical range: 0.01 to 100
- **solver='lbfgs'**: Recommended for most cases. For large datasets, consider 'saga'.
- **multi_class='auto'**: Handles both binary and multi-class classification.

**Why L2 Regularization?**
- Improves numerical stability
- Prevents overfitting on small datasets
- Standard practice in machine learning (unlike traditional statistics)

### 4. Train/Test Splits

**Standard Practice:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 80/20 split
    random_state=seed,
    stratify=y,       # Balanced class distribution in both splits
)
```

**Important:**
- Use `stratify=y` to ensure balanced class distributions
- Common split: 80% train, 20% test
- For small datasets, consider cross-validation

### 5. Multi-Seed Evaluation Protocol

**Why Multiple Seeds?**
- Single train/test split can be unrepresentative
- Random seed affects split quality and model initialization
- Multiple seeds provide confidence intervals and statistical reliability

**Standard Protocol (Based on Recent Research):**

```python
num_seeds = 5  # Minimum 3, recommended 5-10

results = []
for seed in range(num_seeds):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = LogisticRegression(C=1.0, random_state=seed, ...)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    results.append(test_acc)

# Report: mean ± std
print(f"Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
```

**Benefits:**
- More robust performance estimates
- Confidence intervals (mean ± std)
- Statistical significance testing (compare to baselines)

### 6. Cross-Validation

**5-Fold Cross-Validation (Additional Validation):**

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    clf, X_train, y_train,
    cv=5,              # 5 folds
    scoring='accuracy',
    n_jobs=-1
)

print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**When to Use:**
- Small datasets (< 1000 examples)
- Need more robust performance estimates
- Complement to train/test split evaluation

### 7. Statistical Comparison to Other Methods

**Paired t-test for Significance:**

```python
from scipy import stats

# Compare two methods across multiple seeds
method1_scores = [0.85, 0.87, 0.86, 0.88, 0.85]  # Your method
method2_scores = [0.80, 0.82, 0.81, 0.83, 0.80]  # Baseline

t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)

if p_value < 0.05:
    print("Statistically significant difference (p < 0.05)")
```

**Permutation Tests (Gold Standard for Significance):**

```python
from sklearn.model_selection import permutation_test_score

score, perm_scores, p_value = permutation_test_score(
    clf, X, y,
    scoring='accuracy',
    cv=5,
    n_permutations=100,  # Minimum 100, recommended 1000+
    random_state=42
)

print(f"p-value: {p_value:.4f}")  # Probability score is due to chance
```

## Example Usage

### Single Layer Evaluation

```bash
# Evaluate layer 16 with 5 random seeds
python telepathy/linear_probe_hidden_states.py \
    --dataset sst2 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer_idx 16 \
    --num_seeds 5 \
    --normalize l2 \
    --pooling last_token \
    --test_size 0.2 \
    --output_dir runs/linear_probe

# Or use the bash script:
bash telepathy/run_linear_probe.sh
```

### Layer Sweep (Find Optimal Layer)

```bash
# Evaluate layers 0, 4, 8, ..., 32
python telepathy/linear_probe_hidden_states.py \
    --dataset sst2 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer_sweep \
    --layer_start 0 \
    --layer_end 33 \
    --layer_step 4 \
    --num_seeds 3 \
    --output_dir runs/linear_probe_sweep

# Or use the bash script:
bash telepathy/run_linear_probe_sweep.sh
```

## Research References

### 2025-2026 Papers

1. **Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation** (Dec 2025)
   - https://arxiv.org/abs/2512.22245
   - Brier-score-trained linear probes on hidden states
   - Compares to verbalized confidence, self-consistency, majority baselines

2. **PING Framework: Probing INternal states of Generative models** (Sep 2025)
   - https://www.medrxiv.org/content/10.1101/2025.09.17.25336018v2.full
   - Lightweight probes on frozen transformers
   - Extracts residual stream, attention, MLP outputs
   - Pooling, concatenation, and probe training methodology

3. **Paired Seed Evaluation: Statistical Reliability for Learning-based Simulators** (Dec 2025)
   - https://www.arxiv.org/pdf/2512.24145
   - Paired seed evaluation improves statistical reliability
   - Uniformly narrower confidence intervals vs independent evaluation

4. **Can Linear Probes Measure LLM Uncertainty?** (Nov 2025)
   - https://arxiv.org/abs/2510.04108
   - Compares MSP (max softmax probability) baseline
   - Raw hidden states vs Ridge features

### Classic Machine Learning Resources

5. **scikit-learn LogisticRegression Documentation**
   - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
   - Solver recommendations (lbfgs for general, saga for large datasets)
   - Regularization by default (L2)

6. **Cross-validation: evaluating estimator performance**
   - https://scikit-learn.org/stable/modules/cross_validation.html
   - K-fold CV, stratified splits, permutation testing

7. **HuggingFace Model Outputs Documentation**
   - https://huggingface.co/docs/transformers/main_classes/output
   - How to extract hidden states with `output_hidden_states=True`

## Best Practices Summary

1. **Extract hidden states from multiple layers** (especially middle layers)
2. **Normalize features** using L2 normalization or standardization
3. **Use scikit-learn LogisticRegression** with default L2 regularization (C=1.0)
4. **Stratified train/test splits** (80/20 or 70/30)
5. **Multi-seed evaluation** (minimum 3 seeds, recommended 5-10)
6. **Report mean ± std** across seeds
7. **Statistical significance testing** when comparing to baselines
8. **Cross-validation** for additional robustness (especially on small datasets)

## Code Structure

```
telepathy/
├── linear_probe_hidden_states.py  # Main implementation
├── run_linear_probe.sh            # Single layer evaluation
├── run_linear_probe_sweep.sh      # Layer sweep
└── LINEAR_PROBE_METHODOLOGY.md    # This document
```

## Example Output

```json
{
  "layer_16": {
    "layer_idx": 16,
    "num_seeds": 5,
    "test_accuracy_mean": 0.8543,
    "test_accuracy_std": 0.0127,
    "test_accuracy_min": 0.8380,
    "test_accuracy_max": 0.8690,
    "test_f1_mean": 0.8521,
    "test_f1_std": 0.0135,
    "per_seed_results": [
      {
        "seed": 0,
        "test_accuracy": 0.8543,
        "test_f1": 0.8521,
        "train_accuracy": 0.8912,
        "train_f1": 0.8895,
        "cv_mean": 0.8756,
        "cv_std": 0.0089
      },
      ...
    ]
  }
}
```

## Next Steps

1. **Compare to your LatentWire system**: Evaluate if latent representations outperform single-layer probes
2. **Layer analysis**: Which layers contain most task-relevant information?
3. **Transfer experiments**: Train probe on one dataset, test on another
4. **Statistical significance**: Run permutation tests to validate improvements
5. **Visualization**: t-SNE/UMAP of hidden states to understand representation geometry
