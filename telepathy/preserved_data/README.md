# Preserved Data for Paper

This directory contains preserved experimental data, results, and source code snapshots for paper writing.

---

## Directory Structure

```
preserved_data/
├── README.md                    # This file
├── exp001_sst2_signal_check/    # First successful experiment
│   ├── EXPERIMENT_SUMMARY.md    # Detailed experiment documentation
│   ├── config.json              # Training configuration
│   ├── eval_sst2_results.json   # Full evaluation results
│   ├── training_metrics.txt     # Key training metrics
│   ├── quick_eval_trajectory.txt # Accuracy during training
│   └── source_code/             # Code snapshot at time of experiment
│       ├── latent_bridge_v15.py
│       ├── train_telepathy_sst2.py
│       ├── eval_telepathy_sst2.py
│       └── run_sst2_signal_check.sh
├── exp002_baselines/            # [PENDING] Baseline comparisons
└── exp003_agnews/               # [PENDING] 4-class classification
```

---

## Experiment Index

| ID | Name | Status | Key Result |
|----|------|--------|------------|
| exp001 | SST-2 Signal Check | ✅ Complete | 93.46% accuracy |
| exp002 | SST-2 Baselines | ⏳ Pending | - |
| exp003 | AG News (4-class) | ⏳ Pending | - |
| exp004 | QA Task | ⏳ Pending | - |
| exp005 | GSM8K | ⏳ Pending | - |

---

## Naming Convention

- `expXXX_name/` - Experiment folder
- `EXPERIMENT_SUMMARY.md` - Required for every experiment
- `config.json` - Training/eval configuration
- `*_results.json` - Structured results
- `source_code/` - Code snapshot at time of experiment

---

## Requirements for Paper-Ready Experiments

Each experiment folder MUST contain:

1. **EXPERIMENT_SUMMARY.md** with:
   - Objective and hypothesis
   - Method (architecture, training params)
   - Results (tables, key metrics)
   - Significance and conclusions

2. **Configuration files** (JSON)

3. **Results files** (JSON)

4. **Source code snapshot** (for reproducibility)

---

## How to Add New Experiments

```bash
# 1. Create experiment folder
mkdir -p telepathy/preserved_data/expXXX_name

# 2. Copy results
cp runs/run_id/*.json telepathy/preserved_data/expXXX_name/

# 3. Copy source code
mkdir -p telepathy/preserved_data/expXXX_name/source_code
cp telepathy/relevant_scripts.py telepathy/preserved_data/expXXX_name/source_code/

# 4. Write EXPERIMENT_SUMMARY.md
# Use exp001 as template

# 5. Update this README's experiment index
```

---

## Paper Sections Mapping

| Paper Section | Experiments |
|---------------|-------------|
| Method | exp001 (architecture description) |
| Signal Validation | exp001, exp002 |
| Scaling | exp003, exp004, exp005 |
| Ablations | TBD |
