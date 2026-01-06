# Paper Data Collection Checklist

## Overview
Complete checklist of ALL experiments, metrics, and data needed for the LatentWire paper publication.

---

## ✅ Core Metrics (All Experiments)

### Primary Task Metrics
- [ ] **Exact Match (EM)** - For QA tasks (SQuAD, HotpotQA)
- [ ] **F1 Score** - Token-level F1 for QA tasks
- [ ] **Accuracy** - For classification tasks (SST-2, AG News, TREC)
- [ ] **BLEU Score** - For generation quality (GSM8k)
- [ ] **ROUGE-L** - For summarization quality

### Compression Metrics
- [ ] **Compression Ratio** - Text bytes / Latent bytes
- [ ] **Wire Bytes** - Actual transmitted data size
- [ ] **Space Savings** - Percentage reduction
- [ ] **Quantization Levels** - fp16, int8, int4 compression

### Efficiency Metrics
- [ ] **Inference Time** - Wall-clock time per sample
- [ ] **Throughput** - Samples per second
- [ ] **Memory Usage** - Peak GPU memory
- [ ] **FLOPs** - Computational efficiency

### Learning Metrics
- [ ] **First Token Accuracy** - Critical for generation
- [ ] **K-Token Accuracy** - For K=1,2,3,4 supervision
- [ ] **NLL/Token** - Negative log-likelihood on gold answers
- [ ] **Perplexity** - Language modeling quality

---

## ✅ Datasets (All Must Be Tested)

### Question Answering
- [ ] **SQuAD v1.1** - Extractive QA (87,599 train / 10,570 dev)
- [ ] **HotpotQA** - Multi-hop reasoning (90,447 train / 7,405 dev)

### Text Classification
- [ ] **SST-2** - Binary sentiment (67,349 train / 872 dev)
- [ ] **AG News** - 4-class topic (120,000 train / 7,600 test)
- [ ] **TREC** - 6-class question type (5,452 train / 500 test)

### Mathematical Reasoning
- [ ] **GSM8k** - Grade school math (7,473 train / 1,319 test)

### Cross-Dataset Generalization
- [ ] Train on SQuAD → Test on HotpotQA
- [ ] Train on SST-2 → Test on AG News
- [ ] Train on all → Test on each

---

## ✅ Baselines (All Required)

### Primary Baselines
- [ ] **Text Baseline** - Full prompt via text (upper bound)
- [ ] **Token Budget** - Text truncated to M tokens
- [ ] **Random Latent** - Random vectors (lower bound)

### Compression Baselines
- [ ] **LLMLingua** - State-of-the-art prompt compression
  - [ ] Compression rate = 0.25
  - [ ] Compression rate = 0.5
  - [ ] Compression rate = 0.75
- [ ] **Linear Probe** - Direct embedding mapping
  - [ ] Llama → Qwen
  - [ ] Qwen → Llama
  - [ ] With/without fine-tuning

### Ensemble Methods
- [ ] **Joint Rescoring** - Two-model answer selection
- [ ] **Oracle Selection** - Best possible answer (upper bound)

---

## ✅ Ablation Studies

### Architecture Ablations
- [ ] **Encoder Type**
  - [ ] Byte-level encoder (default)
  - [ ] Character-level encoder
  - [ ] Token-level encoder
  - [ ] No encoder (direct projection)

- [ ] **Latent Dimensions**
  - [ ] M = 8, 16, 32, 64, 128 tokens
  - [ ] d_z = 128, 256, 512, 1024 dimensions

- [ ] **Adapter Architecture**
  - [ ] Linear adapter (default)
  - [ ] MLP adapter (2-layer)
  - [ ] Attention-based adapter
  - [ ] No adapter (direct use)

### Training Ablations
- [ ] **Loss Components**
  - [ ] CE only (no KD)
  - [ ] KD only (no CE)
  - [ ] CE + KD (default)
  - [ ] + Contrastive loss
  - [ ] + Reconstruction loss

- [ ] **K-Token Supervision**
  - [ ] K = 1 (first token only)
  - [ ] K = 2, 3, 4, 8 tokens
  - [ ] Full sequence supervision

- [ ] **Calibration Methods**
  - [ ] No calibration
  - [ ] Batch RMS (global)
  - [ ] Example RMS (default)
  - [ ] Layer norm
  - [ ] Learned calibration

### Data Ablations
- [ ] **Training Data Size**
  - [ ] 1k, 5k, 10k, 50k, 87k samples
  - [ ] Effect on convergence and quality

- [ ] **Answer Length**
  - [ ] Short (1-5 tokens)
  - [ ] Medium (5-20 tokens)
  - [ ] Long (20+ tokens)

- [ ] **Anchor Text**
  - [ ] No anchor
  - [ ] "Answer: " (default)
  - [ ] Task-specific anchors

---

## ✅ Statistical Testing (All Required)

### Significance Tests
- [ ] **Paired t-test** - For matched samples
- [ ] **Wilcoxon signed-rank** - Non-parametric alternative
- [ ] **McNemar's test** - For binary correctness
- [ ] **ANOVA** - For multi-condition comparisons

### Confidence Intervals
- [ ] **Bootstrap CI** - 95% confidence intervals
  - [ ] n=10,000 bootstrap samples
  - [ ] BCa method for bias correction
- [ ] **Standard Error** - For all mean metrics

### Multiple Comparisons
- [ ] **Bonferroni correction** - For family-wise error
- [ ] **Holm-Bonferroni** - Sequential correction
- [ ] **FDR control** - Benjamini-Hochberg

### Effect Sizes
- [ ] **Cohen's d** - Standardized mean difference
- [ ] **Odds ratio** - For binary outcomes
- [ ] **R-squared** - Variance explained

---

## ✅ Model Configurations

### Source Models (Frozen)
- [ ] **Llama-3.1-8B-Instruct** - Primary source
- [ ] **Llama-3.1-70B** - Scaling experiment
- [ ] **Llama-2-7B** - Previous generation

### Target Models (Frozen)
- [ ] **Qwen2.5-7B-Instruct** - Primary target
- [ ] **Qwen2.5-14B** - Scaling experiment
- [ ] **Mistral-7B** - Alternative family

### Training Configurations
- [ ] **Standard** - 24 epochs, batch 64
- [ ] **Quick** - 8 epochs for ablations
- [ ] **Extended** - 48 epochs for convergence

---

## ✅ Hyperparameter Sweeps

### Critical Parameters
- [ ] **Learning Rate** - 1e-5, 5e-5, 1e-4, 5e-4
- [ ] **Batch Size** - 8, 16, 32, 64, 128
- [ ] **Warmup Steps** - 0, 500, 1000, 2000
- [ ] **Weight Decay** - 0, 0.01, 0.1

### Regularization
- [ ] **Dropout** - 0, 0.1, 0.2, 0.3
- [ ] **Gradient Clipping** - 1.0, 5.0, 10.0
- [ ] **Label Smoothing** - 0, 0.1, 0.2

### Generation Parameters
- [ ] **Temperature** - 0.1, 0.5, 0.7, 1.0
- [ ] **Top-p** - 0.9, 0.95, 1.0
- [ ] **Top-k** - 10, 50, 100
- [ ] **EOS Ban Steps** - 0, 2, 4, 8

---

## ✅ Visualization Requirements

### Training Curves
- [ ] Loss curves (train/val)
- [ ] Metric curves (EM, F1)
- [ ] Learning rate schedule
- [ ] Gradient norms

### Performance Plots
- [ ] Compression vs Quality tradeoff
- [ ] Latent size vs Performance
- [ ] Training size vs Convergence
- [ ] Cross-dataset transfer matrix

### Analysis Figures
- [ ] Attention patterns
- [ ] Latent space visualization (t-SNE/UMAP)
- [ ] Token importance heatmaps
- [ ] Error analysis breakdown

---

## ✅ Reproducibility Requirements

### Seeds and Randomness
- [ ] **Random Seeds** - 42, 123, 456 (minimum 3)
- [ ] **Report mean ± std** - For all metrics
- [ ] **Deterministic operations** - Where possible

### Environment Documentation
- [ ] **Package versions** - requirements.txt
- [ ] **Hardware specs** - GPUs, memory
- [ ] **Training time** - Wall-clock hours
- [ ] **Checkpoint sizes** - Storage requirements

### Code and Data Release
- [ ] **Training scripts** - Complete pipeline
- [ ] **Evaluation scripts** - All baselines
- [ ] **Pretrained models** - Best checkpoints
- [ ] **Results tables** - Raw JSON data

---

## ✅ Compute Requirements

### Training Runs
- [ ] **Main experiments** - 4×H100, 12-24 hours each
- [ ] **Ablations** - 1×H100, 2-4 hours each
- [ ] **Hyperparameter search** - 100+ GPU hours
- [ ] **Statistical tests** - Multiple seeds

### Storage
- [ ] **Checkpoints** - ~10GB per full experiment
- [ ] **Logs** - ~100MB per run
- [ ] **Processed data** - ~5GB total
- [ ] **Results** - ~1GB JSON/CSV

---

## ✅ Paper Tables (Must Generate)

### Table 1: Main Results
- [ ] All models × All datasets
- [ ] Mean ± std over 3 seeds
- [ ] Bold best, underline second

### Table 2: Compression Analysis
- [ ] Compression ratios
- [ ] Wire bytes
- [ ] Quality retention

### Table 3: Ablation Results
- [ ] Component contributions
- [ ] Statistical significance

### Table 4: Baseline Comparison
- [ ] vs LLMLingua
- [ ] vs Linear Probe
- [ ] vs Token Budget

### Table 5: Cross-Dataset Transfer
- [ ] Generalization matrix
- [ ] Zero-shot performance

---

## ✅ Paper Figures (Must Generate)

### Figure 1: System Architecture
- [ ] Encoder-adapter pipeline
- [ ] Frozen LLM integration

### Figure 2: Performance Curves
- [ ] Quality vs Compression
- [ ] Convergence plots

### Figure 3: Ablation Impact
- [ ] Bar charts of contributions
- [ ] With error bars

### Figure 4: Qualitative Examples
- [ ] Success cases
- [ ] Failure analysis

---

## ✅ Completion Criteria

Before submission, verify:
- [ ] All experiments run with ≥3 seeds
- [ ] All baselines implemented and tested
- [ ] Statistical significance computed
- [ ] Confidence intervals reported
- [ ] Ablations cover key components
- [ ] Cross-dataset evaluation complete
- [ ] Hyperparameters properly tuned
- [ ] Results reproducible from scripts
- [ ] Tables and figures camera-ready
- [ ] Raw data available for review

---

## Priority Order for Time-Constrained Execution

### Week 1: Core System
1. Main training pipeline (SQuAD, 3 seeds)
2. Text and token-budget baselines
3. Basic metrics (EM, F1, compression)

### Week 2: Baselines
1. LLMLingua implementation
2. Linear probe baseline
3. Statistical testing framework

### Week 3: Ablations
1. Latent size ablations (M)
2. K-token supervision study
3. Loss component analysis

### Week 4: Generalization
1. Cross-dataset evaluation
2. Additional datasets (SST-2, AG News)
3. Model scaling experiments

### Week 5: Polish
1. Hyperparameter tuning
2. Visualization generation
3. Statistical validation
4. Paper writing

---

## Notes

- Start with small experiments to debug pipeline
- Run full experiments only after validation
- Keep detailed logs of all runs
- Back up results regularly
- Generate tables/figures incrementally
- Document any deviations from plan

---

*Last Updated: January 2025*
*Version: 1.0.0*