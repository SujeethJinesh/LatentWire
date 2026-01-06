# Metrics Module Documentation

This directory contains the metric computation modules copied from the LatentWire project.

## Files

### 1. `core_utils.py`
Main utility file containing core metric implementations and helper functions.

**Key Functions:**
- `em(pred, truth)`: Computes exact match score
- `f1(pred, truth)`: Computes token-level F1 score
- `squad_em_f1(preds, truths)`: SQuAD-style EM/F1 with multiple reference answers
- `batch_metrics(preds, golds)`: Batch computation of EM and F1
- `_normalize(text)`: Text normalization for metric computation
- Additional utilities for prompt building, anchoring, and data processing

### 2. `rouge_xsum_metrics.py`
ROUGE metric implementation specifically designed for XSUM summarization tasks.

**Key Features:**
- Supports multiple ROUGE backends (rouge_score, evaluate libraries)
- Implements ROUGE-1, ROUGE-2, and ROUGE-L metrics
- Proper error handling and batch processing
- Statistical analysis capabilities
- XSUM-specific optimizations

**Main Function:**
- `compute_rouge_xsum(predictions, references, use_stemmer=True)`: Computes all ROUGE metrics

### 3. `metrics.py`
Consolidated metrics module that provides a unified interface.

**Key Functions:**
- `em()`, `f1()`, `squad_em_f1()`: Core QA metrics
- `compute_rouge()`: Wrapper for ROUGE metrics
- `accuracy()`: Classification accuracy
- `compute_all_metrics()`: All-in-one metric computation based on task type
- `dump_metrics()`, `load_metrics()`: Utility functions for saving/loading

## Usage Examples

### Question Answering (EM/F1)
```python
from metrics import em, f1, squad_em_f1

# Single prediction
em_score = em("Barack Obama", "obama")  # Returns 1.0
f1_score = f1("The quick brown fox", "quick fox")  # Returns partial match score

# Multiple predictions with multiple references
predictions = ["Paris", "London", "Berlin"]
references = [["Paris", "paris"], ["London"], ["Berlin", "BERLIN"]]
avg_em, avg_f1 = squad_em_f1(predictions, references)
```

### Summarization (ROUGE)
```python
from rouge_xsum_metrics import compute_rouge_xsum

predictions = ["This is a summary", "Another summary"]
references = ["This is the reference", "Reference summary"]

results = compute_rouge_xsum(predictions, references, use_stemmer=True)
print(f"ROUGE-1 F1: {results.rouge1_f1}")
print(f"ROUGE-2 F1: {results.rouge2_f1}")
print(f"ROUGE-L F1: {results.rougeL_f1}")
```

### All Metrics at Once
```python
from metrics import compute_all_metrics

# For QA task
predictions = ["answer1", "answer2"]
references = [["correct1", "alt1"], ["correct2"]]
metrics = compute_all_metrics(predictions, references, task_type="qa")
# Returns: {"em": 0.x, "f1": 0.y}

# For summarization
metrics = compute_all_metrics(predictions, references, task_type="summarization")
# Returns: {"rouge1": 0.x, "rouge2": 0.y, "rougeL": 0.z, "f1": 0.w}
```

## Dependencies

- **Core metrics (EM/F1)**: No external dependencies beyond standard library
- **ROUGE metrics**: Requires either `rouge-score` or `evaluate` library
  ```bash
  pip install rouge-score
  # or
  pip install evaluate
  ```

## Notes

- All text metrics perform normalization (lowercasing, removing articles, punctuation)
- SQuAD metrics take the maximum score across multiple reference answers
- ROUGE metrics support optional Porter stemming for better matching
- The consolidated `metrics.py` provides a task-agnostic interface