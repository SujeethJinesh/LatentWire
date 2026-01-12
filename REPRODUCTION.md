# Reproducing Telepathy Results (MLSys 2025)

This guide provides step-by-step instructions for reproducing all experimental results presented in the Telepathy paper. The Telepathy system enables cross-model communication via learned soft tokens, where a "sender" LLM (Llama) transmits information to a "receiver" LLM (Mistral) through a trained latent bridge.

## Table of Contents
1. [Quick Start](#quick-start-reproduce-all)
2. [Hardware Requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Reproducing Individual Results](#reproducing-individual-results)
   - [Table 1: Main Classification Results](#table-1-main-classification-results)
   - [Table 2: Reasoning Benchmarks](#table-2-reasoning-benchmarks)
   - [Table 3: Baseline Comparisons](#table-3-baseline-comparisons)
   - [Table 4: Latency and Throughput](#table-4-latency-and-throughput)
   - [Table 5: Token Ablation Study](#table-5-token-ablation-study)
   - [Figure 1: t-SNE Visualizations](#figure-1-t-sne-visualizations)
   - [Figure 2: Attention Analysis](#figure-2-attention-analysis)
5. [Expected Results](#expected-results)
6. [Statistical Testing](#statistical-testing)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (Reproduce All)

To reproduce all paper results with a single command on an HPC cluster with SLURM:

```bash
# Clone the repository
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire

# Install dependencies
pip install -r requirements.txt

# Submit the comprehensive evaluation job (HPC with SLURM)
sbatch telepathy/submit_enhanced_paper_eval.slurm
```

This runs the complete evaluation including:
- 3 random seeds (42, 123, 456) for statistical significance
- 2 token configurations (8, 24) for ablation study
- 4 datasets: SST-2, AG News, BoolQ, GSM8K
- All baselines: zero-shot, few-shot, prompt tuning, linear probe, LoRA, text-relay
- Latency and memory benchmarks
- Statistical significance testing

**Estimated runtime**: 10-12 hours on a single H100 GPU.

---

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA A100 40GB or H100 80GB
- **RAM**: 64GB system memory
- **Disk**: 50GB free space (for models and checkpoints)
- **CUDA**: 11.8+ or 12.x

### Recommended Setup
- **GPU**: NVIDIA H100 80GB
- **RAM**: 128GB system memory
- **Disk**: 100GB SSD
- **CUDA**: 12.1+

### Expected Resource Usage
| Experiment | GPU Memory | Time (H100) |
|------------|------------|-------------|
| Bridge Training (1 dataset) | ~45GB | 15-20 min |
| Full Evaluation Suite | ~50GB | 10-12 hours |
| Linear Probe Baseline | ~25GB | 5-10 min |
| Latency Benchmarks | ~50GB | 30 min |

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check transformers can load models
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct'); print('Llama tokenizer loaded')"
```

### 5. Hugging Face Authentication
Some models require authentication. Set your token:
```bash
huggingface-cli login
# Or set environment variable:
export HF_TOKEN="your_token_here"
```

---

## Reproducing Individual Results

### Table 1: Main Classification Results

**Datasets**: SST-2 (sentiment), AG News (topic), TREC (question type)

#### Training the Bridge

```bash
# SST-2 (2-class sentiment classification)
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --soft_tokens 8 \
    --steps 2000 \
    --source_layer 31 \
    --batch_size 16 \
    --lr 2e-4 \
    --seed 42 \
    --output_dir runs/sst2

# AG News (4-class topic classification)
python telepathy/train_telepathy.py \
    --dataset agnews \
    --soft_tokens 8 \
    --steps 3000 \
    --source_layer 31 \
    --batch_size 16 \
    --lr 2e-4 \
    --seed 42 \
    --output_dir runs/agnews

# TREC (6-class question classification)
python telepathy/train_telepathy.py \
    --dataset trec \
    --soft_tokens 8 \
    --steps 2000 \
    --source_layer 31 \
    --batch_size 16 \
    --lr 2e-4 \
    --seed 42 \
    --output_dir runs/trec
```

#### Evaluating the Bridge

```bash
# SST-2 Evaluation
python telepathy/eval_telepathy.py \
    --checkpoint runs/sst2/bridge_sst2.pt \
    --dataset sst2 \
    --num_samples 872 \
    --output_dir runs/sst2

# AG News Evaluation
python telepathy/eval_telepathy.py \
    --checkpoint runs/agnews/bridge_agnews.pt \
    --dataset agnews \
    --num_samples 1000 \
    --output_dir runs/agnews

# TREC Evaluation
python telepathy/eval_telepathy.py \
    --checkpoint runs/trec/bridge_trec.pt \
    --dataset trec \
    --num_samples 500 \
    --output_dir runs/trec
```

#### Multi-Seed Evaluation (for statistical significance)

```bash
# Run training and evaluation with 3 seeds
for SEED in 42 123 456; do
    python telepathy/train_telepathy.py \
        --dataset sst2 \
        --soft_tokens 8 \
        --steps 2000 \
        --seed $SEED \
        --output_dir runs/sst2_seed${SEED}

    python telepathy/eval_telepathy.py \
        --checkpoint runs/sst2_seed${SEED}/bridge_sst2.pt \
        --dataset sst2 \
        --output_dir runs/sst2_seed${SEED}
done
```

---

### Table 2: Reasoning Benchmarks

**Datasets**: BoolQ (yes/no QA), GSM8K (math reasoning)

Note: Reasoning tasks use the same bridge architecture but require longer sequences.

```bash
# BoolQ - Boolean Question Answering
python telepathy/train_telepathy.py \
    --dataset boolq \
    --soft_tokens 16 \
    --steps 3000 \
    --batch_size 8 \
    --seed 42 \
    --output_dir runs/boolq

# GSM8K - Math Word Problems (requires special answer extraction)
# Note: GSM8K evaluation uses exact match on final numerical answer
python telepathy/train_telepathy.py \
    --dataset gsm8k \
    --soft_tokens 24 \
    --steps 5000 \
    --batch_size 8 \
    --seed 42 \
    --output_dir runs/gsm8k
```

---

### Table 3: Baseline Comparisons

The paper compares Telepathy against several baselines:

#### Zero-Shot Baseline

```bash
# Zero-shot evaluation on both Llama and Mistral
python telepathy/run_baselines.py \
    --baseline zeroshot \
    --dataset sst2 \
    --max_samples 500 \
    --output_dir runs/baselines

python telepathy/run_baselines.py \
    --baseline zeroshot \
    --dataset agnews \
    --max_samples 500 \
    --output_dir runs/baselines
```

#### Few-Shot Baseline

```bash
# 5-shot in-context learning
python telepathy/run_baselines.py \
    --baseline fewshot \
    --dataset sst2 \
    --shots 5 \
    --seeds 42 123 456 \
    --max_samples 500 \
    --output_dir runs/baselines
```

#### LoRA Baseline

```bash
# LoRA fine-tuning (rank=8)
python telepathy/run_baselines.py \
    --baseline lora \
    --dataset sst2 \
    --rank 8 \
    --epochs 3 \
    --max_train_samples 2000 \
    --seeds 42 123 456 \
    --output_dir runs/baselines
```

#### Prompt Tuning Baseline

```bash
# Learnable soft prompts (no sender model)
python telepathy/run_baselines.py \
    --baseline prompt_tuning \
    --dataset sst2 \
    --soft_tokens 8 \
    --steps 2000 \
    --seeds 42 123 456 \
    --output_dir runs/baselines
```

#### Linear Probe Baseline

This baseline demonstrates that the Perceiver architecture adds value beyond simple linear projection:

```bash
# Linear probe on Llama hidden states
python telepathy/linear_probe_baseline.py \
    --datasets sst2 agnews trec \
    --layers 16 20 24 28 31 \
    --pooling last_token \
    --max_train_samples 5000 \
    --max_test_samples 1000 \
    --seeds 42 123 456 \
    --output_dir runs/linear_probe
```

---

### Table 4: Latency and Throughput

#### Latency Benchmark

```bash
# Single-sample latency comparison
python telepathy/run_benchmarks.py \
    --benchmark latency \
    --checkpoint runs/sst2/bridge_sst2.pt \
    --soft_tokens 8 \
    --num_trials 50 \
    --warmup 5 \
    --output_dir runs/benchmarks
```

#### Batched Throughput Benchmark

```bash
# Throughput at various batch sizes
python telepathy/run_benchmarks.py \
    --benchmark batched \
    --batch_sizes 1 2 4 8 16 \
    --num_samples 100 \
    --output_dir runs/benchmarks
```

#### Memory Benchmark

```bash
# Peak GPU memory comparison
python telepathy/run_benchmarks.py \
    --benchmark memory \
    --output_dir runs/benchmarks
```

---

### Table 5: Token Ablation Study

The paper evaluates different numbers of soft tokens (information bottleneck):

```bash
# Token ablation: 4, 8, 16, 24, 32 tokens
for TOKENS in 4 8 16 24 32; do
    python telepathy/train_telepathy.py \
        --dataset sst2 \
        --soft_tokens $TOKENS \
        --steps 2000 \
        --seed 42 \
        --output_dir runs/ablation_tokens${TOKENS}

    python telepathy/eval_telepathy.py \
        --checkpoint runs/ablation_tokens${TOKENS}/bridge_sst2.pt \
        --dataset sst2 \
        --soft_tokens $TOKENS \
        --output_dir runs/ablation_tokens${TOKENS}
done
```

---

### Figure 1: t-SNE Visualizations

Visualize learned latent representations:

```bash
python telepathy/generate_tsne_visualization.py \
    --checkpoint runs/sst2/bridge_sst2.pt \
    --dataset sst2 \
    --num_samples 500 \
    --output_dir figures/tsne
```

---

### Figure 2: Attention Analysis

Analyze attention patterns in the bridge:

```bash
python telepathy/attention_analysis.py \
    --checkpoint runs/sst2/bridge_sst2.pt \
    --dataset sst2 \
    --num_samples 100 \
    --output_dir figures/attention
```

---

## Expected Results

### Classification Accuracy (Table 1)

| Dataset | Random | Zero-Shot | Few-Shot | Bridge (8 tokens) |
|---------|--------|-----------|----------|-------------------|
| SST-2   | 50.0%  | 85-90%    | 88-92%   | 90-94%           |
| AG News | 25.0%  | 70-75%    | 75-80%   | 82-87%           |
| TREC    | 16.7%  | 55-65%    | 65-75%   | 75-82%           |

### Baseline Comparison (Table 3)

| Method | SST-2 | AG News | TREC | Params |
|--------|-------|---------|------|--------|
| Zero-Shot Mistral | 85-90% | 70-75% | 55-65% | 0 |
| 5-Shot | 88-92% | 75-80% | 65-75% | 0 |
| Prompt Tuning | 70-80% | 60-70% | 55-65% | ~32K |
| Linear Probe (L24) | 88-92% | 80-85% | 70-78% | ~8K |
| LoRA (r=8) | 90-94% | 82-87% | 75-82% | ~4M |
| **Bridge (Ours)** | **90-94%** | **82-87%** | **75-82%** | **~1M** |

### Latency Results (Table 4)

| Method | Latency (ms) | Speedup |
|--------|--------------|---------|
| Direct Text | 50-80 | 1.0x |
| Text Relay | 150-250 | - |
| Bridge (8 tokens) | 80-120 | 1.5-2.0x vs relay |

---

## Statistical Testing

### Running Statistical Analysis

After running experiments with multiple seeds, aggregate results:

```bash
python telepathy/aggregate_results.py \
    --base_dir runs/ \
    --output_dir results/
```

### Statistical Significance Tests

```bash
python telepathy/statistical_tests.py \
    --results_path results/aggregated_results.json \
    --output_path results/statistical_analysis.json
```

This computes:
- Bootstrap 95% confidence intervals
- Paired t-tests between methods
- McNemar's test for paired classification
- Bonferroni correction for multiple comparisons
- Cohen's d effect sizes

### Generating LaTeX Tables

```bash
python scripts/generate_paper_tables.py \
    --results_path results/aggregated_results.json \
    --output_path results/paper_tables.tex
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: `torch.cuda.OutOfMemoryError`

**Solutions**:
```bash
# Reduce batch size
python telepathy/train_telepathy.py --batch_size 8  # instead of 16

# Use gradient accumulation
python telepathy/train_telepathy.py --batch_size 4 --grad_accum 4

# Clear cache between experiments
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Model Loading Issues

**Problem**: `OSError: Unable to load model`

**Solutions**:
```bash
# Ensure Hugging Face authentication
huggingface-cli login

# Set cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# For Llama models, ensure access is granted
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError`

**Solutions**:
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=.

# Install missing dependencies
pip install -r requirements.txt

# For peft (LoRA baseline)
pip install peft>=0.5.0
```

#### 4. Slow Training

**Problem**: Training takes longer than expected

**Solutions**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Use bfloat16 (default, but verify)
python telepathy/train_telepathy.py --bf16

# Increase batch size if memory allows
python telepathy/train_telepathy.py --batch_size 32
```

#### 5. Inconsistent Results Across Seeds

**Problem**: High variance between seeds (>5% std)

**Solutions**:
- Ensure reproducibility flags are set:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```
- Use deterministic algorithms:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
- Run with more seeds and report confidence intervals

### HPC/SLURM-Specific Issues

#### Job Fails with "Account not found"

Ensure correct SLURM settings:
```bash
#SBATCH --account=YOUR_ACCOUNT  # Replace with your allocation
#SBATCH --partition=gpu         # Or 'preempt' depending on cluster
```

#### Running Out of Time

Increase time limit or use fast mode:
```bash
#SBATCH --time=24:00:00  # Increase from 12 hours

# Or use fast mode (reduced training steps)
python telepathy/run_enhanced_paper_evaluation.py --fast_mode
```

### Getting Help

If you encounter issues not covered here:
1. Check the existing GitHub issues
2. Open a new issue with:
   - Full error traceback
   - Python and package versions (`pip freeze`)
   - GPU type and CUDA version
   - Commands you ran

---

## File Structure Reference

```
LatentWire/
├── telepathy/
│   ├── train_telepathy.py          # Main training script
│   ├── eval_telepathy.py           # Evaluation script
│   ├── run_baselines.py            # Baseline methods
│   ├── run_benchmarks.py           # Latency/throughput benchmarks
│   ├── linear_probe_baseline.py    # Linear probe implementation
│   ├── statistical_tests.py        # Statistical analysis
│   ├── aggregate_results.py        # Results aggregation
│   ├── generate_tsne_visualization.py
│   ├── attention_analysis.py
│   └── submit_enhanced_paper_eval.slurm  # SLURM job script
├── scripts/
│   ├── generate_paper_tables.py    # LaTeX table generation
│   └── statistical_testing.py      # Additional stats
├── requirements.txt
├── REPRODUCTION.md                  # This file
└── runs/                           # Output directory (created)
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{telepathy2025,
  title={Telepathy: Cross-Model Communication via Learned Soft Tokens},
  author={...},
  booktitle={MLSys},
  year={2025}
}
```

---

## Acknowledgments

This research was supported by [funding sources]. We thank [acknowledgments].
