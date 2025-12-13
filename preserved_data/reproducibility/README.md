# Reproducibility Guide for Cross-Model Communication Experiments

**Last Updated**: 2025-12-13
**Project**: LatentWire / Telepathy

This directory contains all configurations and expected results needed to reproduce the key findings.

## Quick Start

```bash
# From project root
git pull && rm -rf runs && PYTHONPATH=. bash preserved_data/reproducibility/run_all_experiments.sh
```

## Key Results Summary

| Task | Bridge | Text-Relay | Mistral Text | Llama Text | Notes |
|------|--------|------------|--------------|------------|-------|
| SST-2 (2-class) | **94.7%** | 71.0% | 93.5% | — | Bridge beats relay by 24pp |
| AG News (4-class) | **88.9%** | 64.5% | 70.5% | — | Bridge beats relay by 24pp |
| Banking77 (77-class) | **21.5%** | TBD | 19.5% | 22.0% | Bridge matches Llama ceiling |
| Passkey (5-digit) | 0% exact | — | — | — | 23.4% digit accuracy |

## Key Findings

1. **Text-Baseline Parity**: Bridge achieves sender model ceiling on Banking77
2. **Bridge >> Text-Relay**: +24pp improvement on SST-2 and AG News
3. **Inverse Token Scaling**: More tokens = worse performance (mode collapse)
4. **16 Tokens Optimal**: Best results consistently at 16 soft tokens

---

## Experiment Configurations

### 1. SST-2 Sentiment Classification

**Expected Results**: 94.7% accuracy (Bridge), 93.5% (Mistral text)

```bash
# Training
python telepathy/train_telepathy_sst2.py \
    --source_layer 31 \
    --soft_tokens 8 \
    --steps 2000 \
    --batch_size 8 \
    --lr 1e-4 \
    --diversity_weight 0.1 \
    --output_dir runs/sst2

# Evaluation
python telepathy/eval_telepathy_sst2.py \
    --checkpoint runs/sst2/bridge_final.pt \
    --num_samples 200
```

### 2. AG News Topic Classification

**Expected Results**: 88.9% accuracy (Bridge), 70.5% (Mistral text)

```bash
# Training
python telepathy/train_telepathy_agnews.py \
    --source_layer 31 \
    --soft_tokens 8 \
    --steps 2000 \
    --batch_size 8 \
    --lr 1e-4 \
    --diversity_weight 0.1 \
    --output_dir runs/agnews

# Evaluation
python telepathy/eval_telepathy_agnews.py \
    --checkpoint runs/agnews/bridge_final.pt \
    --num_samples 200
```

### 3. Banking77 Intent Classification (77 classes)

**Expected Results**: 21.5% accuracy at 16 tokens (matches Llama text ceiling of 22.0%)

```bash
# Training (Token Ablation)
for TOKENS in 16 32 64 128; do
    python telepathy/train_telepathy_banking77.py \
        --output_dir runs/banking77_${TOKENS}tok \
        --soft_tokens $TOKENS \
        --steps 3000 \
        --batch_size 8 \
        --eval_every 500 \
        --gpu 0
done

# Text Baselines
python telepathy/eval_text_relay_baseline.py \
    --banking77 \
    --num_samples 200 \
    --output_dir runs/banking77_baselines

# Text-Relay Baseline
python telepathy/eval_text_relay_baseline.py \
    --banking77_relay \
    --num_samples 200 \
    --output_dir runs/banking77_relay
```

**Token Ablation Results**:
| Tokens | Accuracy | Notes |
|--------|----------|-------|
| 16 | 21.5% | Best (matches text ceiling) |
| 32 | 13.5% | Mode collapse starting |
| 64 | 7.5% | Severe mode collapse |
| 128 | 1.0% | Random (complete collapse) |

### 4. Passkey Retrieval (5-digit codes)

**Expected Results**: 0% exact match, ~23% digit accuracy at 16 tokens

```bash
# Training (Token Ablation)
for TOKENS in 16 32 64 128; do
    python telepathy/train_telepathy_passkey.py \
        --output_dir runs/passkey_${TOKENS}tok \
        --soft_tokens $TOKENS \
        --steps 1000 \
        --batch_size 8 \
        --eval_every 200 \
        --gpu 0
done
```

**Token Ablation Results**:
| Tokens | Exact Match | Digit Accuracy |
|--------|-------------|----------------|
| 16 | 0% | 23.4% |
| 32 | 0% | 22.8% |
| 64 | 0% | 18.4% |
| 128 | 0% | 9.8% (random) |

### 5. Text-Relay Baselines (SST-2, AG News)

**Expected Results**: SST-2 71.0%, AG News 64.5%

```bash
python telepathy/eval_text_relay_baseline.py \
    --num_samples 200 \
    --output_dir runs/text_relay
```

---

## Hardware Requirements

- **GPU**: 1-4x H100 80GB (or equivalent)
- **Memory**: 64GB+ system RAM
- **Storage**: ~50GB for models and checkpoints

## Runtime Estimates

| Experiment | GPUs | Time |
|------------|------|------|
| SST-2 Training | 1 | ~30 min |
| AG News Training | 1 | ~30 min |
| Banking77 Ablation | 4 (parallel) | ~45 min |
| Passkey Ablation | 4 (parallel) | ~20 min |
| Text Baselines | 1 | ~30 min |
| **Total** | 4 | **~2.5 hours** |

---

## Model Versions

- **Llama**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3`

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| soft_tokens | 8-16 | 16 for Banking77/Passkey, 8 for SST-2/AG News |
| source_layer | 31 | Last layer of Llama |
| learning_rate | 1e-4 | Adam optimizer |
| batch_size | 8 | Per GPU |
| diversity_weight | 0.1 | Prevents mode collapse |
| steps | 2000-3000 | Task dependent |

---

## Preserved Data Locations

```
preserved_data/
├── phase20_inverse_scaling_2025-12-13/   # Token ablation results
│   ├── banking77_16tok_*/                 # 21.5% accuracy
│   ├── banking77_32tok_*/                 # 13.5% accuracy
│   ├── banking77_64tok_*/                 # 7.5% accuracy
│   ├── banking77_128tok_*/                # 1.0% accuracy
│   ├── passkey_*/                         # All passkey results
│   └── text_relay_*/                      # SST-2 71%, AG 64.5%
├── phase21_text_baselines_2025-12-13/    # Direct text baselines
│   └── banking77_baselines_*/             # Mistral 19.5%, Llama 22.0%
└── reproducibility/                       # This directory
    ├── README.md                          # This file
    └── run_all_experiments.sh             # Master script
```

---

## Citation

If you use these results, please cite the project repository.
