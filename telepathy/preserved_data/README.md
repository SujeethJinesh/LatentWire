# Preserved Data for Paper

This directory contains preserved experimental data, results, and source code snapshots for paper writing.

---

## Directory Structure

```
preserved_data/
├── README.md                              # This file
├── exp001_sst2_signal_check/              # First successful experiment
├── exp002_sst2_baselines/                 # SST-2 baseline comparisons
├── exp003_comprehensive_ablations/        # 25 configs, 6 dimensions
├── exp004_agnews/                         # 4-class classification (old prompts)
├── exp005_sst2_corrected_prompts/         # SST-2 with fair comparison
├── exp006_agnews_corrected_prompts/       # AG News with fair comparison
└── exp007_gsm8k_latent_cot/               # GSM8K negative result
```

---

## Experiment Index

| ID | Name | Status | Key Result |
|----|------|--------|------------|
| exp001 | SST-2 Signal Check | ✅ Complete | 93.46% accuracy |
| exp002 | SST-2 Baselines | ✅ Complete | Bridge matches Mistral text (93.5%) |
| **exp003** | **Comprehensive Ablations** | ✅ **Complete** | **Layer 31 + 8 tokens = 96.5%** |
| exp004 | AG News (old prompts) | ✅ Complete | 92.3% (pre-correction) |
| **exp005** | **SST-2 Corrected Prompts** | ✅ **Complete** | **94.72% (exceeds Mistral 93.5%)** |
| **exp006** | **AG News Corrected Prompts** | ✅ **Complete** | **88.9% (+18.4pp vs Mistral text)** |
| **exp007** | **GSM8K Latent CoT** | ✅ **Complete** | **2.0% - NEGATIVE RESULT** |

---

## Paper-Ready Results Summary

### Classification (SUCCESS)

| Task | Bridge | Best Baseline | Delta |
|------|--------|---------------|-------|
| SST-2 (sentiment) | 94.7% | 93.5% (Mistral) | **+1.2pp** |
| AG News (topic) | 88.9% | 74.5% (Llama) | **+14.4pp** |

**Key Finding**: Bridge exceeds text baselines on classification tasks.

### Reasoning (FAILURE)

| Task | Bridge | Best Baseline | Delta |
|------|--------|---------------|-------|
| GSM8K (math) | 2.0% | 76.5% (Llama) | **-74.5pp** |

**Key Finding**: Latent CoT does not enable reasoning. The architecture works for pattern matching but not computation.

---

## Hero Results

1. **Science Category Fix** (exp006): Mistral text 37% → Bridge 89% (+52pp)
2. **Information Bottleneck Validated** (exp003): 8 tokens > 32 tokens
3. **Cross-Model Transfer Exceeds Same-Model Text**: Bridge (Llama→Mistral) > Mistral text

---

## Corrected vs Original Experiments

**exp004 vs exp006 (AG News)**:
- exp004: Old prompts, 92.3% accuracy (potentially unfair comparison)
- exp006: Corrected prompts, 88.9% accuracy (fair comparison)

**Use exp005-007 for paper** - these have consistent prompts for fair baseline comparison.

---

## Naming Convention

- `expXXX_name/` - Experiment folder
- `EXPERIMENT_SUMMARY.md` - Required for every experiment
- `config.json` - Training/eval configuration
- `*_results.json` - Structured results
- `source_code/` - Code snapshot at time of experiment

---

## Paper Sections Mapping

| Paper Section | Experiments |
|---------------|-------------|
| Method | exp001 (architecture description) |
| Signal Validation | exp001, exp002 |
| **Ablations** | **exp003 (comprehensive - 25 configs, 6 dimensions)** |
| Classification | exp005 (SST-2), exp006 (AG News) |
| Reasoning | exp007 (GSM8K - negative result) |

---

## How to Add New Experiments

```bash
# 1. Create experiment folder
mkdir -p telepathy/preserved_data/expXXX_name

# 2. Copy results
cp runs/run_id/*.json telepathy/preserved_data/expXXX_name/

# 3. Write EXPERIMENT_SUMMARY.md
# Use existing experiments as template

# 4. Update this README's experiment index
```
