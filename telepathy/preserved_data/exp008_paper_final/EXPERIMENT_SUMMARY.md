# Experiment: Paper Final Results (exp008)

**Date**: 2025-12-13 to 2025-12-14
**Status**: IN PROGRESS (TREC multi-seed pending)
**Purpose**: Final experiments for MLSys 2025 paper submission

---

## Executive Summary

This experiment set provides the final results for the Telepathy paper. Key findings:

1. **Prompt-tuning baseline = random chance** - Proves sender model is essential
2. **Super-additive performance** - Bridge exceeds both individual models
3. **Multi-seed validation** - SST-2 and AG News show stable results

---

## Completed Experiments

### Bridge (Llama → Mistral)

| Dataset | Seeds | Mean ± Std | Individual Results |
|---------|-------|------------|-------------------|
| SST-2 | 3/3 ✅ | **96.7% ± 0.6%** | 96.5, 96.0, 97.5 |
| AG News | 3/3 ✅ | **90.7% ± 0.5%** | 90.0, 91.0, 91.0 |
| TREC | 0/3 ⏳ | 94.5% (single) | Pending multi-seed |
| Banking77 | 1/1 | 21.5% | Single seed OK for inverse scaling demo |

### Prompt-Tuning Baseline (Mistral only, no Llama)

| Dataset | Seeds | Mean ± Std | vs Random Chance |
|---------|-------|------------|------------------|
| SST-2 | 6/6 ✅ | **49.5% ± 0.0%** | ≈ 50% (random) |
| AG News | 6/6 ✅ | **19.8% ± 7.5%** | < 25% (random) |
| TREC | 3/3 ✅ | **19.0% ± 5.0%** | ≈ 16.7% (random) |

---

## Key Findings

### 1. Sender Model is Essential (+47pp)
- Prompt-tuning on Mistral alone: 49.5% (random chance)
- Bridge with Llama: 96.7%
- **Improvement: +47.2 percentage points**

### 2. Super-Additive Performance
- SST-2: Bridge 96.7% > Llama 92.0% > Mistral 88.5%
- AG News: Bridge 90.7% > Llama 79.0% = Mistral 79.0%
- TREC: Bridge 94.5% > Llama 53.5% > Mistral 43.0%

### 3. Layer Configuration
- SST-2: Layer 16 works well (96.7%)
- AG News/TREC: Layer 31 (final layer)
- Ablation shows deeper layers generally better for classification

---

## Remaining Work

### Critical (Must Complete)
- [ ] TREC Bridge multi-seed (3 seeds)
  - Script: `run_trec_multiseed.sh`
  - Est. time: ~15 min on H100

### Nice to Have (Not Required)
- [ ] Banking77 multi-seed (for completeness)
- [ ] Layer ablation with 8 tokens (currently have 32-token ablation)
- [ ] Reverse direction (Mistral → Llama)

---

## Configuration

### Bridge Architecture
- Type: Perceiver Resampler (LatentBridgeV15)
- Parameters: 188K trainable
- Soft tokens: 8 (SST-2, AG News), 16 (TREC, Banking77)
- Depth: 2 cross-attention layers
- Internal dim: 512

### Training
- Steps: 2000 (SST-2, AG News, TREC), 3000 (Banking77)
- Learning rate: 2e-4
- Batch size: 8-16
- Optimizer: AdamW
- Seeds: 42, 123, 456

### Models
- Sender: Llama 3.1 8B Instruct (frozen)
- Receiver: Mistral 7B Instruct v0.3 (frozen)

---

## File Locations

Results stored in: `telepathy/runs/paper_experiments_20251213_190318/`

```
paper_experiments_20251213_190318/
├── bridge_multiseed/
│   ├── sst2_seed{42,123,456}/     ✅ Complete
│   ├── agnews_seed{42,123,456}/   ✅ Complete
│   └── trec_seed{42,123,456}/     ⏳ Pending
├── prompt_tuning_baseline/
│   ├── sst2_seed{42,123,456}/     ✅ Complete
│   ├── agnews_seed{42,123,456}/   ✅ Complete
│   └── trec_seed{42,123,456}/     ✅ Complete
└── experiments_*.log              # Training logs
```

---

## Paper Updates Required After TREC

When TREC multi-seed completes:
1. Update Table 1: TREC column with mean±std
2. Update multi-seed appendix with TREC values
3. Recompile paper

---

## Reproduction

To reproduce all experiments:
```bash
cd telepathy
PYTHONPATH=.. bash run_paper_experiments.sh
```

To run only missing TREC:
```bash
cd telepathy
PYTHONPATH=.. bash run_trec_multiseed.sh
```
