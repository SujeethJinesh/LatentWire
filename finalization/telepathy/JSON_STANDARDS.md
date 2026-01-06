# JSON Output Standards for LatentWire/Telepathy

This document defines the standardized JSON output formats for all experimental scripts in the LatentWire/Telepathy project.

## Overview

All experimental scripts that produce JSON output must conform to the schemas defined in `json_schema.json`. This ensures:

1. **Consistency**: All scripts produce compatible outputs
2. **Interoperability**: Results can be aggregated and compared across experiments
3. **Validation**: Outputs can be automatically validated for correctness
4. **Documentation**: Clear structure makes results self-documenting

## Key Files

- **`json_schema.json`**: The canonical JSON schema definition
- **`validate_json_outputs.py`**: Script to validate outputs against schema
- **`JSON_STANDARDS.md`**: This documentation file

## Standard Field Definitions

### Core Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `accuracy` | number | 0-100 | Percentage accuracy (NOT 0-1) |
| `correct` | integer | ≥0 | Number of correct predictions |
| `total` | integer | ≥1 | Total number of examples |
| `f1` | number | 0-1 | F1 score (harmonic mean of precision/recall) |
| `latency_ms` | number | ≥0 | Latency in milliseconds |
| `memory_mb` | number | ≥0 | Memory usage in megabytes |
| `compression_ratio` | number | ≥0 | Compression ratio achieved |

### Metadata Fields

| Field | Type | Format | Description |
|-------|------|--------|-------------|
| `timestamp` | string | `YYYYMMDD_HHMMSS` or ISO 8601 | When experiment was run |
| `seeds` | array[int] | e.g., `[42, 123, 456]` | Random seeds used |
| `datasets` | array[string] | e.g., `["sst2", "agnews"]` | Datasets evaluated |
| `direction` | string | e.g., `"FORWARD (Llama→Mistral)"` | Bridge direction |

### Statistical Fields

| Field | Type | Description |
|-------|------|-------------|
| `mean` | number | Mean value across seeds/samples |
| `std` | number | Standard deviation |
| `ci_lower` | number | Lower bound of 95% confidence interval |
| `ci_upper` | number | Upper bound of 95% confidence interval |
| `p_value` | number (0-1) | Statistical significance p-value |
| `effect_size` | number | Cohen's d or similar effect size |

## Output Formats by Script

### 1. `run_unified_comparison.py`

Primary output for comparing multiple methods on classification tasks.

```json
{
  "meta": {
    "timestamp": "20251221_124013",
    "direction": "FORWARD (Llama→Mistral)",
    "sender": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "receiver": "mistralai/Mistral-7B-Instruct-v0.3",
    "soft_tokens": 8,
    "train_steps": 2000,
    "eval_samples": 200,
    "seeds": [42, 123, 456],
    "ensemble_enabled": false
  },
  "per_seed_results": {
    "sst2": {
      "42": {
        "random_chance": 50.0,
        "bridge": {
          "accuracy": 86.5,
          "correct": 173,
          "total": 200,
          "latency_ms": 52.03,
          "train_info": {
            "final_loss": 0.154
          }
        },
        "prompt_tuning": {...},
        "text_relay": {...}
      },
      "123": {...},
      "456": {...}
    }
  },
  "aggregated_results": {
    "sst2": {
      "bridge": {
        "accuracy_mean": 88.0,
        "accuracy_std": 2.5,
        "accuracy_ci": {
          "mean": 88.0,
          "ci_lower": 85.5,
          "ci_upper": 90.5
        }
      }
    }
  }
}
```

### 2. `run_comprehensive_revision.py`

Orchestrates multiple experimental phases for paper revision.

```json
{
  "metadata": {
    "timestamp": "20251221_090000",
    "seeds": [42, 123, 456],
    "datasets": ["sst2", "agnews", "trec"],
    "model_size": "medium",
    "model_config": {
      "sender": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "receiver": "mistralai/Mistral-7B-Instruct-v0.3",
      "batch_size": 2,
      "description": "Medium models (7-8B params)"
    }
  },
  "phases": {
    "1": {
      "phase": 1,
      "description": "Statistical rigor with full test sets",
      "datasets": ["sst2", "agnews"],
      "results": {...}
    },
    "2": {
      "phase": 2,
      "description": "Linear probe baseline",
      "linear_probe_results": {...}
    }
  }
}
```

### 3. `aggregate_results.py`

Aggregates results from multiple experiments.

```json
{
  "metadata": {
    "generated": "2026-01-04T20:46:16.136825",
    "n_experiments": 13,
    "n_aggregated": 13,
    "seeds": [42, 123, 456],
    "datasets": ["sst2", "agnews", "trec"],
    "models": ["llama3.1-8b", "mistral-7b"]
  },
  "raw_results": {
    "('bridge', 'sst2', 'default')": {
      "seed": [42, 123, 456],
      "accuracy": [96.5, 96.0, 97.5],
      "f1": [0.95, 0.94, 0.96],
      "latency_ms": [45, 46, 44]
    }
  },
  "aggregated_results": {
    "('bridge', 'sst2', 'default')": {
      "accuracy": {
        "mean": 96.67,
        "ci_lower": 95.5,
        "ci_upper": 97.5,
        "std": 0.76,
        "n_samples": 3
      }
    }
  },
  "significance_tests": {
    "bridge_vs_prompt_tuning_sst2": {
      "p_value": 0.001,
      "significant": true,
      "test_type": "mcnemar",
      "effect_size": 1.2
    }
  }
}
```

## Validation

### Running Validation

```bash
# Validate a single file
python telepathy/validate_json_outputs.py --file runs/results.json

# Validate all files in a directory
python telepathy/validate_json_outputs.py --dir runs/

# Check all runs
python telepathy/validate_json_outputs.py --check-all

# Check consistency across files
python telepathy/validate_json_outputs.py --dir runs/ --check-consistency

# Attempt automatic fixes (creates backups)
python telepathy/validate_json_outputs.py --dir runs/ --fix
```

### Common Validation Errors

1. **Percentage Format**: Accuracy must be 0-100, not 0-1
   - ❌ `"accuracy": 0.865`
   - ✅ `"accuracy": 86.5`

2. **Missing Required Fields**: All accuracy metrics need `correct` and `total`
   - ❌ `{"accuracy": 86.5}`
   - ✅ `{"accuracy": 86.5, "correct": 173, "total": 200}`

3. **Timestamp Format**: Use `YYYYMMDD_HHMMSS` or ISO 8601
   - ❌ `"timestamp": "2025-12-21"`
   - ✅ `"timestamp": "20251221_124013"`

4. **Nested Structure**: Results must be properly nested by dataset and seed
   - ❌ `"results": {"accuracy": 86.5}`
   - ✅ `"per_seed_results": {"sst2": {"42": {...}}}`

## Best Practices

### 1. Always Include Metadata

```python
results = {
    "meta": {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seeds": args.seeds,
        "datasets": args.datasets,
        # Include all relevant configuration
    }
}
```

### 2. Use Consistent Keys

- Use `accuracy` not `acc` or `accuracy_score`
- Use `latency_ms` not `latency` or `time_ms`
- Use `memory_mb` not `mem` or `memory_usage`

### 3. Include Both Raw and Aggregated

```python
results = {
    "per_seed_results": {...},  # Raw results for each seed
    "aggregated_results": {...}  # Mean, std, CI across seeds
}
```

### 4. Make Results Self-Contained

Include all information needed to understand the results without external context:

```python
results = {
    "meta": {
        "experiment_type": "bridge_transfer",
        "model_pair": "llama-8b_to_mistral-7b",
        "hyperparameters": {
            "learning_rate": 5e-4,
            "batch_size": 4,
            "soft_tokens": 8
        }
    }
}
```

### 5. Handle Missing Data Gracefully

```python
if method_failed:
    results[method] = {
        "accuracy": None,
        "error": "Training diverged",
        "partial_results": {...}
    }
```

## Integration Example

```python
import json
from datetime import datetime
from typing import Dict, List

def save_experiment_results(
    results: Dict,
    output_dir: str,
    validate: bool = True
) -> None:
    """Save results following JSON standards."""

    # Ensure proper structure
    if "meta" not in results:
        results["meta"] = {}

    # Add timestamp if missing
    if "timestamp" not in results["meta"]:
        results["meta"]["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert accuracies to percentages if needed
    def fix_accuracies(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if "accuracy" in key and isinstance(value, (int, float)):
                    if 0 <= value <= 1:
                        obj[key] = value * 100
                elif isinstance(value, (dict, list)):
                    fix_accuracies(value)

    fix_accuracies(results)

    # Save
    output_path = f"{output_dir}/results_{results['meta']['timestamp']}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Validate if requested
    if validate:
        from telepathy.validate_json_outputs import validate_json_file, load_schema
        schema = load_schema()
        is_valid, error = validate_json_file(output_path, schema)
        if not is_valid:
            print(f"Warning: Output validation failed: {error}")

    print(f"Results saved to: {output_path}")
```

## Checklist for New Scripts

When creating a new experimental script:

- [ ] Review `json_schema.json` for the appropriate format
- [ ] Use standardized field names (see table above)
- [ ] Include complete metadata
- [ ] Provide both raw and aggregated results
- [ ] Use 0-100 for percentages, not 0-1
- [ ] Include confidence intervals for aggregated metrics
- [ ] Test with `validate_json_outputs.py`
- [ ] Handle errors gracefully with partial results
- [ ] Make output self-contained and interpretable

## Questions?

For questions or to propose changes to the schema, please:

1. Check existing examples in `runs/` directory
2. Review the schema in `json_schema.json`
3. Test changes with `validate_json_outputs.py`
4. Document any new fields or formats in this file