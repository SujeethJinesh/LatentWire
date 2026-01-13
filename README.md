# Telepathy

Continuous latent communication for cross-model inference. MLSys 2025.

## Overview

Telepathy enables cross-model communication via learned soft tokens, where a "sender" LLM (Llama) transmits information to a "receiver" LLM (Mistral) through a trained latent bridge. This achieves significant latency reduction over text-based communication while maintaining task accuracy.

## Key Results

| Dataset | Telepathy | Text-Relay | Speedup |
|---------|-----------|------------|---------|
| SST-2   | 93.7%     | 41.3%      | 4.7x    |
| AG News | 90.7%     | 1.0%       | 4.7x    |

## Installation

```bash
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Hugging Face authentication (required for Llama models)
huggingface-cli login
```

### Hardware Requirements

- **Minimum**: NVIDIA A100 40GB, 64GB RAM, CUDA 11.8+
- **Recommended**: NVIDIA H100 80GB, 128GB RAM, CUDA 12.1+

## Quick Start

```bash
# Train bridge on SST-2
python telepathy/train_telepathy.py --dataset sst2

# Evaluate
python telepathy/eval_telepathy.py --dataset sst2 --checkpoint runs/sst2/bridge.pt
```

## Reproducing Paper Results

### Full Reproduction (HPC with SLURM)

```bash
sbatch telepathy/submit_enhanced_paper_eval.slurm
```

This runs the complete evaluation including:
- 3 random seeds (42, 123, 456) for statistical significance
- 2 token configurations (8, 24) for ablation study
- 4 datasets: SST-2, AG News, BoolQ, GSM8K
- All baselines: zero-shot, few-shot, prompt tuning, linear probe, LoRA, text-relay
- Latency and memory benchmarks

**Estimated runtime**: 10-12 hours on a single H100 GPU.

### Individual Experiments

#### Table 1: Classification Results

```bash
# SST-2
python telepathy/train_telepathy.py --dataset sst2 --soft_tokens 8 --steps 2000 --seed 42 --output_dir runs/sst2
python telepathy/eval_telepathy.py --checkpoint runs/sst2/bridge_sst2.pt --dataset sst2 --output_dir runs/sst2

# AG News
python telepathy/train_telepathy.py --dataset agnews --soft_tokens 8 --steps 3000 --seed 42 --output_dir runs/agnews
python telepathy/eval_telepathy.py --checkpoint runs/agnews/bridge_agnews.pt --dataset agnews --output_dir runs/agnews
```

#### Table 2: Reasoning Benchmarks

```bash
# BoolQ
python telepathy/train_telepathy.py --dataset boolq --soft_tokens 16 --steps 3000 --seed 42 --output_dir runs/boolq

# GSM8K
python telepathy/train_telepathy.py --dataset gsm8k --soft_tokens 24 --steps 5000 --seed 42 --output_dir runs/gsm8k
```

#### Table 3: Baseline Comparisons

```bash
# Zero-shot
python telepathy/run_baselines.py --baseline zeroshot --dataset sst2 --output_dir runs/baselines

# Few-shot
python telepathy/run_baselines.py --baseline fewshot --dataset sst2 --shots 5 --output_dir runs/baselines

# LoRA
python telepathy/run_baselines.py --baseline lora --dataset sst2 --rank 8 --output_dir runs/baselines

# Prompt tuning
python telepathy/run_baselines.py --baseline prompt_tuning --dataset sst2 --soft_tokens 8 --output_dir runs/baselines

# Linear probe
python telepathy/linear_probe_baseline.py --datasets sst2 agnews --layers 16 20 24 28 31 --output_dir runs/linear_probe
```

#### Table 4: Latency Benchmarks

```bash
python telepathy/run_benchmarks.py --benchmark latency --checkpoint runs/sst2/bridge_sst2.pt --output_dir runs/benchmarks
python telepathy/run_benchmarks.py --benchmark batched --batch_sizes 1 2 4 8 16 --output_dir runs/benchmarks
```

#### Table 5: Token Ablation

```bash
for TOKENS in 4 8 16 24 32; do
    python telepathy/train_telepathy.py --dataset sst2 --soft_tokens $TOKENS --output_dir runs/ablation_tokens${TOKENS}
    python telepathy/eval_telepathy.py --checkpoint runs/ablation_tokens${TOKENS}/bridge_sst2.pt --dataset sst2 --soft_tokens $TOKENS
done
```

#### Visualizations

```bash
# t-SNE
python telepathy/generate_tsne_visualization.py --checkpoint runs/sst2/bridge_sst2.pt --dataset sst2 --output_dir figures/tsne

# Attention analysis
python telepathy/attention_analysis.py --checkpoint runs/sst2/bridge_sst2.pt --dataset sst2 --output_dir figures/attention
```

### Statistical Testing

```bash
python telepathy/aggregate_results.py --base_dir runs/ --output_dir results/
python telepathy/statistical_tests.py --results_path results/aggregated_results.json --output_path results/statistical_analysis.json
```

## Repository Structure

```
LatentWire/
├── latentwire/                 # Core library
│   ├── train.py                # Training loop
│   ├── eval.py                 # Evaluation
│   ├── models.py               # Encoder, Adapter, LMWrapper
│   └── ...
├── telepathy/                  # Paper experiments
│   ├── train_telepathy.py      # Unified training
│   ├── eval_telepathy.py       # Unified evaluation
│   ├── run_baselines.py        # All baselines
│   ├── run_benchmarks.py       # Latency/throughput
│   ├── linear_probe_baseline.py
│   ├── run_enhanced_paper_evaluation.py
│   └── paper_writing/          # LaTeX source
├── scripts/                    # Analysis utilities
│   └── generate_paper_tables.py
├── requirements.txt
└── runs/                       # Output directory (created)
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size (`--batch_size 8`) or use gradient accumulation (`--grad_accum 4`).

**Model Loading Issues**: Ensure Hugging Face authentication and model access permissions for Llama.

**Import Errors**: Set `export PYTHONPATH=.` before running scripts.

## Citation

```bibtex
@inproceedings{telepathy2025,
  title={Telepathy: Continuous Latent Communication for Cross-Model Inference},
  author={...},
  booktitle={MLSys},
  year={2025}
}
```

## License

MIT
