# Example: Layer-16 Hidden States from Llama 3.1 8B

This is a minimal, self-contained example showing exactly how to extract layer-16 hidden states from Llama and train a linear probe.

## Quick Start

```bash
# Single command to run layer-16 probe on SST-2
cd /Users/sujeethjinesh/Desktop/LatentWire
export PYTHONPATH=.
python telepathy/linear_probe_hidden_states.py \
    --dataset sst2 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer_idx 16 \
    --num_seeds 5 \
    --output_dir runs/linear_probe_layer16
```

## Step-by-Step Code Example

### 1. Extract Layer-16 Hidden States

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

# Example text
texts = [
    "This movie was absolutely fantastic!",
    "I hated every minute of this film.",
]

# Tokenize
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
).to("cuda")

# Extract hidden states
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,  # KEY: Enable hidden states
        return_dict=True,
    )

# outputs.hidden_states is a tuple of length 33 for Llama 3.1 8B:
# [embeddings, layer1, layer2, ..., layer32]
# Index 0 = embeddings (before any transformer layer)
# Index 16 = layer 16 (middle layer)
# Index 32 = layer 32 (final layer)

layer_16_hidden = outputs.hidden_states[16]  # [batch=2, seq_len, hidden_dim=4096]
print(f"Layer 16 shape: {layer_16_hidden.shape}")
# Output: Layer 16 shape: torch.Size([2, seq_len, 4096])

# Pool to single vector per example (use last non-padding token)
attention_mask = inputs["attention_mask"]
seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index
batch_indices = torch.arange(layer_16_hidden.size(0))
pooled_hidden = layer_16_hidden[batch_indices, seq_lengths]  # [batch=2, 4096]

print(f"Pooled shape: {pooled_hidden.shape}")
# Output: Pooled shape: torch.Size([2, 4096])
```

### 2. Train Linear Probe

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Convert to numpy
X = pooled_hidden.cpu().numpy()  # [num_examples, 4096]
y = np.array([1, 0])  # 1 = positive, 0 = negative

# L2 normalization (standard practice)
X_normalized = normalize(X, norm='l2', axis=1)  # ||x|| = 1 per example

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train logistic regression
clf = LogisticRegression(
    C=1.0,              # Inverse regularization strength
    max_iter=1000,
    solver='lbfgs',
    random_state=42,
)
clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

### 3. Multi-Seed Evaluation (Robust)

```python
# Run with multiple random seeds for confidence intervals
results = []

for seed in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    results.append(test_acc)

# Report mean ± std
mean_acc = np.mean(results)
std_acc = np.std(results)
print(f"Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
```

## Understanding Llama 3.1 8B Layer Structure

```python
# Llama 3.1 8B has 32 transformer layers
# outputs.hidden_states gives 33 tensors:

layer_0  = outputs.hidden_states[0]   # Embeddings (before transformer)
layer_1  = outputs.hidden_states[1]   # After 1st transformer layer
layer_2  = outputs.hidden_states[2]   # After 2nd transformer layer
...
layer_16 = outputs.hidden_states[16]  # After 16th transformer layer (MIDDLE)
...
layer_31 = outputs.hidden_states[31]  # After 31st transformer layer
layer_32 = outputs.hidden_states[32]  # After 32nd transformer layer (FINAL)

# Note: layer_32 == outputs.last_hidden_state
```

## Why Layer 16?

- **Middle layers often perform best** for downstream tasks
- Layer 16 is the midpoint of Llama 3.1 8B (32 layers total)
- Research shows intermediate layers capture both:
  - Low-level linguistic features (from early layers)
  - High-level semantic features (from later layers)

## Expected Results on SST-2

Based on typical linear probe performance:

- **Layer 16 (middle)**: ~85-90% accuracy
- **Layer 0 (embeddings)**: ~70-75% accuracy
- **Layer 32 (final)**: ~80-85% accuracy

Middle layers often outperform both early and final layers for classification.

## Full Working Script

See `telepathy/linear_probe_hidden_states.py` for a complete implementation with:
- Batch processing for efficiency
- Multiple datasets (SST-2, AG News, TREC)
- Layer sweep experiments
- Statistical significance testing
- Comprehensive logging and result saving

## Run Examples

```bash
# Layer 16 only (recommended starting point)
python telepathy/linear_probe_hidden_states.py \
    --dataset sst2 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer_idx 16 \
    --num_seeds 5 \
    --output_dir runs/linear_probe_layer16

# Compare multiple layers (find best layer)
python telepathy/linear_probe_hidden_states.py \
    --dataset sst2 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer_sweep \
    --layer_start 0 \
    --layer_end 33 \
    --layer_step 4 \
    --num_seeds 3 \
    --output_dir runs/linear_probe_sweep

# Use bash scripts for convenience
bash telepathy/run_linear_probe.sh
bash telepathy/run_linear_probe_sweep.sh
```

## Output Format

Results are saved to `runs/linear_probe_layer16/results.json`:

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

## Troubleshooting

### Out of Memory (OOM)

If you run out of GPU memory:

```bash
# Reduce batch size
python telepathy/linear_probe_hidden_states.py \
    --batch_size 4 \  # Default is 8
    ...

# Or use CPU (slower but no memory issues)
python telepathy/linear_probe_hidden_states.py \
    --device cpu \
    ...
```

### Model Download Issues

If you can't download the model:

```python
# Use a smaller model for testing
--model_id meta-llama/Llama-2-7b-hf  # Smaller, faster
```

### Dataset Loading Issues

```python
# Check dataset is available
from datasets import load_dataset
dataset = load_dataset("glue", "sst2", split="train")
print(f"Dataset loaded: {len(dataset)} examples")
```

## Next Steps

1. **Run layer sweep** to find optimal layer for your task
2. **Compare to your LatentWire system** - does it beat single-layer probes?
3. **Statistical testing** - use permutation tests to validate improvements
4. **Transfer experiments** - train on one dataset, test on another
5. **Visualization** - t-SNE/UMAP of layer-16 hidden states

## Additional Resources

- Main implementation: `telepathy/linear_probe_hidden_states.py`
- Methodology doc: `telepathy/LINEAR_PROBE_METHODOLOGY.md`
- Research references in methodology doc
