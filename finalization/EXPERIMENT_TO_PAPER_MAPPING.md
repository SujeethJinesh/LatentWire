# Experiment to Paper Mapping Document

## Document Purpose
This document maps experiments in `telepathy/run_complete_evaluation.py` to the paper tables and sections in `finalization/latex/`. It identifies metrics being collected, JSON paths for extraction, and gaps between experiments and paper requirements.

---

## 1. Experiments Being Run

### 1.1 Datasets Evaluated
| Dataset Key | Name | Classes | Test Split | Load Function |
|-------------|------|---------|------------|---------------|
| `sst2` | SST-2 | 2 | validation | glue:sst2 |
| `agnews` | AG News | 4 | test | fancyzhx/ag_news |
| `trec` | TREC | 6 | test | trec |

### 1.2 Models Used
| Role | Model ID | Size |
|------|----------|------|
| Source (Sender) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B |
| Target (Receiver) | `mistralai/Mistral-7B-Instruct-v0.3` | 7B |

### 1.3 Random Seeds
- Seeds: `[42, 123, 456]` (3 seeds for statistical analysis)

---

## 2. Experiment Methods and Metrics

### 2.1 Zero-Shot Baselines (`run_zeroshot_baseline`)
**Purpose**: Establish individual model performance for super-additivity claims

| Output File | JSON Structure |
|-------------|----------------|
| `zeroshot/{model}_{dataset}_seed{N}.json` | See below |

```json
{
  "dataset": "sst2",
  "model_name": "llama" | "mistral",
  "seed": 42,
  "accuracy": 88.5,
  "correct": 770,
  "total": 872,
  "latency_mean_ms": 45.2,
  "latency_std_ms": 5.1,
  "predictions_sample": [...]
}
```

**Metrics Collected**:
- `accuracy` (percentage)
- `latency_mean_ms`, `latency_std_ms`
- Per-example `predictions` (for McNemar's test)

---

### 2.2 Text-Relay Baseline (`run_text_relay_baseline`)
**Purpose**: Llama summarizes -> text -> Mistral classifies (fair two-model comparison)

| Output File | JSON Structure |
|-------------|----------------|
| `text_relay/text_relay_{dataset}_seed{N}.json` | See below |

```json
{
  "dataset": "sst2",
  "method": "text_relay",
  "seed": 42,
  "accuracy": 82.1,
  "correct": 716,
  "total": 872,
  "latency_mean_ms": 120.5,
  "latency_std_ms": 15.3,
  "predictions_sample": [...]
}
```

---

### 2.3 Prompt Tuning Baseline (`train_prompt_tuning_baseline`)
**Purpose**: Learnable soft prompts (Mistral-only, no sender model)

| Output File | JSON Structure |
|-------------|----------------|
| `prompt_tuning/prompt_tuning_{dataset}_seed{N}.json` | See below |

```json
{
  "dataset": "sst2",
  "method": "prompt_tuning",
  "seed": 42,
  "accuracy": 49.5,
  "correct": 432,
  "total": 872,
  "num_soft_tokens": 8,
  "predictions_sample": [...]
}
```

**Config**:
- `soft_tokens`: 8
- `steps`: 1500
- `lr`: 2e-4

---

### 2.4 LoRA Baseline (`train_lora_baseline`)
**Purpose**: Fine-tuning baseline (Mistral with LoRA adapters)

| Output File | JSON Structure |
|-------------|----------------|
| `lora/lora_{dataset}_seed{N}.json` | See below |

```json
{
  "dataset": "sst2",
  "method": "lora",
  "seed": 42,
  "accuracy": 92.0,
  "correct": 802,
  "total": 872,
  "trainable_params": 4200000,
  "predictions_sample": [...]
}
```

**Config**:
- `rank`: 8
- `alpha`: 16
- `epochs`: 2
- `max_train_samples`: 2000

---

### 2.5 Linear Probe Baseline (`run_linear_probe`)
**Purpose**: Logistic regression on Llama hidden states (layer 16)

| Output File | JSON Structure |
|-------------|----------------|
| `linear_probe/linear_probe_{dataset}_seed{N}.json` | See below |

```json
{
  "dataset": "sst2",
  "method": "linear_probe",
  "seed": 42,
  "layer_idx": 16,
  "train_accuracy": 90.2,
  "test_accuracy": 84.5,
  "accuracy": 84.5,
  "train_f1": 89.8,
  "test_f1": 83.9,
  "predictions_sample": [...]
}
```

---

### 2.6 Telepathy Bridge (`train_bridge_for_dataset`)
**Purpose**: Main method - Llama encodes -> bridge -> Mistral classifies

| Output File | JSON Structure |
|-------------|----------------|
| `bridge/{dataset}_seed{N}_tokens{K}_results.json` | See below |

```json
{
  "dataset": "sst2",
  "seed": 42,
  "num_soft_tokens": 8,
  "accuracy": 96.7,
  "correct": 843,
  "total": 872,
  "compression_ratio_avg": 4.2,
  "compression_ratios_sample": [...],
  "predictions_sample": [...],
  "outputs": [...]
}
```

**Token Ablation Configs**: `[8, 16, 32]` tokens

**Bridge Config**:
- `depth`: 2
- `heads`: 8
- `source_layer`: 31
- `train_steps`: 1500
- `diversity_weight`: 0.1

---

### 2.7 Latency Measurement (`measure_fair_latency`)
**Purpose**: Fair speedup comparison (both methods do same classification task)

| Output File | JSON Structure |
|-------------|----------------|
| `latency/latency_results.json` | See below |

```json
{
  "sst2": {
    "bridge_mean_ms": 52.3,
    "bridge_std_ms": 3.1,
    "text_relay_mean_ms": 120.5,
    "text_relay_std_ms": 15.3,
    "mistral_direct_mean_ms": 45.0,
    "mistral_direct_std_ms": 2.8,
    "speedup_vs_text_relay": 2.3
  },
  "average": {...}
}
```

---

### 2.8 Statistical Analysis (`run_statistical_tests_with_correction`)
**Output File**: `statistical_analysis_corrected.json`

```json
{
  "timestamp": "2025-01-09T...",
  "correction_method": "bonferroni",
  "alpha": 0.05,
  "comparisons": {
    "sst2": {
      "dataset_name": "SST-2",
      "num_classes": 2,
      "random_chance": 50.0,
      "methods": {
        "bridge_8": {
          "mean": 96.7,
          "std": 0.6,
          "ci_95": [96.1, 97.3],
          "n_seeds": 3,
          "values": [96.2, 96.8, 97.1]
        },
        "zeroshot_llama": {
          "mean": 88.0,
          "std": 0.0,
          "ci_95": [88.0, 88.0],
          "vs_baseline": {
            "difference": -8.7,
            "p_value_raw": 0.001,
            "p_value_corrected": 0.006,
            "significant": true,
            "cohens_d": 2.87
          }
        }
      }
    }
  },
  "mcnemar_tests": {
    "sst2": {
      "zeroshot_llama": {
        "statistic_mean": 12.45,
        "p_value_combined": 0.0001,
        "contingency_table_combined": [[700, 50], [20, 102]]
      }
    }
  }
}
```

---

## 3. Paper Sections to Update

### 3.1 Main Results Table (`tables.tex` - Table 1)
**Location**: `/finalization/latex/tables.tex` lines 9-32

**Current Placeholders vs. Experiment Data**:

| Placeholder | Experiment Source | JSON Path |
|-------------|-------------------|-----------|
| `[FULL_LLAMA_EM]` | **NOT COLLECTED** - SQuAD EM not in telepathy experiments | N/A |
| `[FULL_LLAMA_F1]` | **NOT COLLECTED** - SQuAD F1 not in telepathy experiments | N/A |
| `[FULL_QWEN_EM]` | **NOT COLLECTED** - Uses Mistral, not Qwen | N/A |
| `[LW_LLAMA_EM]` | **NOT COLLECTED** - SQuAD EM not in telepathy experiments | N/A |
| `[LW_LLAMA_F1]` | **NOT COLLECTED** - SQuAD F1 not in telepathy experiments | N/A |

**CRITICAL GAP**: Tables.tex expects SQuAD EM/F1 with Llama+Qwen. Experiments use SST-2/AG News/TREC with Llama+Mistral.

---

### 3.2 Ablation Study Table (`tables.tex` - Table 2)
**Location**: `/finalization/latex/tables.tex` lines 38-64

**Current Placeholders**:
| Placeholder | Status |
|-------------|--------|
| `[FULL_FIRSTTOK]` | **NOT COLLECTED** - FirstTok@1 not measured in classification |
| `[NO_KTOKEN_*]` | **NOT COLLECTED** - K-token ablation not in telepathy script |
| `[NO_KD_*]` | **NOT COLLECTED** - KD ablation not in telepathy script |

**CRITICAL GAP**: Ablation study placeholders require LatentWire training experiments, not telepathy classification.

---

### 3.3 Compression Analysis Table (`tables.tex` - Table 3)
**Location**: `/finalization/latex/tables.tex` lines 69-99

**Mappable Metrics**:
| Placeholder | JSON Path | Status |
|-------------|-----------|--------|
| `[LW32_FP16_BYTES]` | `bridge_results['compression_ratio_avg']` | Partially available |
| `[LW32_FP16_RATIO]` | Calculated: `32 * 256 * 2 = 16384 bytes` | Can compute |

**Compression Calculation in Script**:
```python
# From calculate_compression_ratio()
text_bytes = len(text.encode('utf-8'))
compressed_bytes = num_tokens * 256 * 16 / 8  # d_z=256, fp16
compression_ratio = text_bytes / compressed_bytes
```

---

### 3.4 Runtime Performance Table (`tables.tex` - Table 4)
**Location**: `/finalization/latex/tables.tex` lines 105-123

**Mappable Metrics**:
| Placeholder | JSON Path | Status |
|-------------|-----------|--------|
| `[LW_TOTAL_TIME]` | `latency_results['average']['bridge_mean_ms']` | Available |
| `[LW_TOTAL_SPEEDUP]` | `latency_results['average']['speedup_vs_text_relay']` | Available |

---

### 3.5 Scaling Analysis Table (`tables.tex` - Table 5)
**Location**: `/finalization/latex/tables.tex` lines 129-154

**Token Ablation Mapping**:
| Config | Placeholder Pattern | JSON Path |
|--------|---------------------|-----------|
| M=32, d_z=256 | `[M32_D256_*]` | `bridge_32` results |
| M=16, d_z=256 | `[M16_D256_*]` | **NOT COLLECTED** (only 8, 16, 32 tokens) |

**Partial Match**: TOKEN_ABLATION_CONFIGS = [8, 16, 32], but table expects [16, 32, 64, 128] with varying d_z.

---

### 3.6 Main Results Table (`main_results.tex`)
**Location**: `/finalization/latex/main_results.tex`

This table shows **classification accuracy** on SST-2, AG News, TREC - which **MATCHES** the telepathy experiments.

**Direct Mapping**:
| Row | JSON Path |
|-----|-----------|
| LatentWire AG News | `bridge_8` results where `dataset='agnews'` |
| LatentWire SST-2 | `bridge_8` results where `dataset='sst2'` |
| LatentWire TREC | `bridge_8` results where `dataset='trec'` |
| Prompt Tuning | `prompt_tuning` results |
| LoRA | `lora` results |
| Linear Probe | `linear_probe` results |
| zeroshot | `zeroshot_llama` / `zeroshot_mistral` results |

---

## 4. Key Metrics Extraction Guide

### 4.1 From `complete_results.json`

```python
# Load results
import json
with open('runs/paper_results/run_TIMESTAMP/complete_results.json') as f:
    results = json.load(f)

# Extract bridge accuracy for SST-2
bridge_sst2_results = [r for r in results['bridge_8'] if r['dataset'] == 'sst2']
mean_acc = sum(r['accuracy'] for r in bridge_sst2_results) / len(bridge_sst2_results)
# -> main_results.tex: LatentWire SST-2 Accuracy

# Extract latency
bridge_latency = results['latency']['average']['bridge_mean_ms']
# -> tables.tex: [LW_TOTAL_TIME]

speedup = results['latency']['average']['speedup_vs_text_relay']
# -> tables.tex: [LW_TOTAL_SPEEDUP]

# Extract compression ratio
compression = bridge_sst2_results[0]['compression_ratio_avg']
# -> main_results.tex: Compression column
```

### 4.2 From `statistical_analysis_corrected.json`

```python
with open('statistical_analysis_corrected.json') as f:
    stats = json.load(f)

# Get mean +/- std for paper table
sst2_bridge = stats['comparisons']['sst2']['methods']['bridge_8']
formatted = f"{sst2_bridge['mean']:.1f} +/- {sst2_bridge['std']:.1f}"
# -> main_results.tex: "96.7 +/- 0.6"

# Get significance stars
vs_baseline = stats['comparisons']['sst2']['methods']['prompt_tuning']['vs_baseline']
p_corrected = vs_baseline['p_value_corrected']
stars = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else ""
# -> main_results.tex: significance markers

# Get effect size
cohens_d = vs_baseline['cohens_d']
# -> statistical_analysis.tex: Cohen's d column
```

### 4.3 From McNemar Tests

```python
mcnemar = stats['mcnemar_tests']['sst2']['prompt_tuning']
stat = mcnemar['statistic_mean']
p_val = mcnemar['p_value_combined']
# -> statistical_analysis.tex: McNemar results
```

---

## 5. Statistical Analysis Mapping

### 5.1 Bootstrap CI -> Paper Claims
- **Method**: BCa bootstrap with 10,000 resamples
- **Output**: 95% CI bounds in `methods[method]['ci_95']`
- **Paper Usage**: Report as `mean [lower, upper]` or `mean +/- std`

### 5.2 McNemar Test -> Significance Sections
- **Purpose**: Per-example prediction comparison
- **Output**: `mcnemar_tests[dataset][method]`
- **Paper Usage**: "Bridge makes different errors than [method] (McNemar p < 0.001)"

### 5.3 Bonferroni Correction -> Reported p-values
- **Raw p-values**: `vs_baseline['p_value_raw']`
- **Corrected p-values**: `vs_baseline['p_value_corrected']`
- **Paper Usage**: ALWAYS report corrected p-values when making multiple comparisons
- **Formula**: `p_corrected = p_raw * num_comparisons`

### 5.4 Effect Sizes
- **Cohen's d**: `vs_baseline['cohens_d']`
- **Interpretation**:
  - |d| < 0.2: negligible
  - 0.2 <= |d| < 0.5: small
  - 0.5 <= |d| < 0.8: medium
  - |d| >= 0.8: large

---

## 6. Critical Gaps and Concerns

### 6.1 Dataset Mismatch (CRITICAL)
| Paper Expects | Experiments Provide |
|---------------|---------------------|
| SQuAD (Question Answering) | SST-2, AG News, TREC (Classification) |
| EM/F1 metrics | Accuracy metrics |
| Generative evaluation | Classification evaluation |

**Impact**: `tables.tex` Table 1, 2, 5, 6 have placeholders for SQuAD EM/F1 that cannot be filled from telepathy experiments.

### 6.2 Model Mismatch (CRITICAL)
| Paper Expects | Experiments Provide |
|---------------|---------------------|
| Llama + Qwen | Llama + Mistral |
| Qwen2.5-7B | Mistral-7B-Instruct-v0.3 |

**Impact**: All `[*_QWEN_*]` placeholders in `tables.tex` cannot be filled.

### 6.3 Missing Ablation Experiments
The telepathy script does NOT run:
- K-token CE ablation (k=1 vs k=4 vs k=8)
- KD ablation
- Calibration ablation
- Anchor text ablation
- Encoder architecture ablation

**Impact**: `tables.tex` Table 2 (Ablation Study) cannot be filled from telepathy experiments.

### 6.4 Missing Token Configurations
| Table Expects | Experiments Provide |
|---------------|---------------------|
| M = 4, 16, 32, 64, 128 | M = 8, 16, 32 |
| d_z = 128, 256, 512 | d_z = 256 (fixed) |

### 6.5 Unfillable Placeholders Summary

**From `tables.tex`**:
- All SQuAD EM/F1 placeholders (~40 placeholders)
- All Qwen-related placeholders (~20 placeholders)
- All ablation study placeholders (~36 placeholders)
- First-token analysis placeholders (~18 placeholders)
- Training efficiency placeholders (~20 placeholders)
- Cross-dataset generalization placeholders (~24 placeholders)

**Can Be Filled**:
- `main_results.tex` - Classification accuracy on SST-2, AG News, TREC
- `ablation_study.tex` - Partially (SST-2 ablation, not full ablations)
- `statistical_analysis.tex` - Statistical test results
- Latency/speedup metrics

---

## 7. Recommendations

### 7.1 To Use Telepathy Results
1. Create new paper tables specifically for classification experiments
2. Update `main_results.tex` with actual values from `complete_results.json`
3. Document that experiments use Llama+Mistral on classification tasks
4. Add note explaining different evaluation paradigm from SQuAD

### 7.2 To Fill All `tables.tex` Placeholders
1. Run LatentWire training on SQuAD with Llama+Qwen (`latentwire/train.py`)
2. Run ablation experiments removing each component
3. Run scaling study with M = [16, 32, 64, 128] and d_z = [128, 256, 512]
4. Run cross-dataset generalization (train on SQuAD, test on HotpotQA, NQ, TriviaQA)

### 7.3 Hybrid Approach
- Use telepathy classification results for a new "Classification Transfer" section
- Keep existing `tables.tex` for SQuAD QA results (requires separate experiments)
- Present as complementary evaluations (generation vs classification)

---

## 8. File Location Summary

| Output Type | Location |
|-------------|----------|
| Complete results | `runs/paper_results/run_TIMESTAMP/complete_results.json` |
| Statistical analysis | `runs/paper_results/run_TIMESTAMP/statistical_analysis_corrected.json` |
| LaTeX tables (generated) | `runs/paper_results/run_TIMESTAMP/paper_tables_comprehensive.tex` |
| Latency results | `runs/paper_results/run_TIMESTAMP/latency/latency_results.json` |
| Per-method results | `runs/paper_results/run_TIMESTAMP/{method}/{method}_{dataset}_seed{N}.json` |
| Bridge checkpoints | `runs/paper_results/run_TIMESTAMP/bridge/{dataset}_seed{N}_tokens{K}_bridge.pt` |

---

*Document generated: January 2025*
*Last updated to reflect run_complete_evaluation.py revision addressing reviewer concerns*
