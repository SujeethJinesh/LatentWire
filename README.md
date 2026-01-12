# Telepathy: Continuous Latent Communication for Cross-Model Inference

**MLSys 2025**

Telepathy enables direct communication between heterogeneous Large Language Models through learned continuous representations, eliminating expensive text generation in multi-agent systems.

## Overview

We introduce a neural bridge architecture that transfers hidden states from a sender model (Llama 3.1-8B) to a receiver model (Mistral-7B) via learned soft tokens. Our key findings:

- **Classification Success**: 96.7% accuracy on SST-2, 90.7% on AG News sentiment/topic classification
- **8x Latency Reduction**: 37ms vs 298ms for text-relay approaches
- **Compression-Classification Tradeoff**: Lossy compression aids classification by removing irrelevant details, but fails on tasks requiring exact information preservation (GSM8K reasoning: 0% vs 48.5% text baseline)

The core insight is that continuous latent bridges excel at transferring high-level semantic features for classification, but the information loss inherent in compression prevents faithful reasoning transfer.

## Installation

```bash
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch 2.2+, CUDA 11.8+ (for GPU training)

## Quick Start

### Train a Telepathy Bridge

```bash
# Train on SST-2 sentiment classification
python telepathy/train_telepathy_sst2.py \
    --sender_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --receiver_model mistralai/Mistral-7B-Instruct-v0.3 \
    --num_soft_tokens 16 \
    --epochs 10 \
    --output_dir runs/telepathy_sst2
```

### Evaluate

```bash
# Evaluate trained bridge
python telepathy/eval_telepathy_sst2.py \
    --checkpoint runs/telepathy_sst2/best_model.pt \
    --samples 500
```

### Run Paper Experiments

```bash
# Full paper evaluation (requires 4x H100 GPUs)
sbatch telepathy/submit_enhanced_paper_eval.slurm

# Or run locally
python telepathy/run_enhanced_paper_evaluation.py
```

## Repository Structure

```
latentwire/                    # Core library
    train.py                   # Training loop
    eval.py                    # Evaluation utilities
    models.py                  # Encoder, Adapter, LMWrapper
    losses.py                  # Loss functions
    data.py                    # Dataset loading

telepathy/                     # Main experiments
    train_telepathy_*.py       # Training scripts per dataset
    eval_telepathy_*.py        # Evaluation scripts
    run_enhanced_paper_evaluation.py   # Paper experiments
    submit_enhanced_paper_eval.slurm   # HPC submission
    paper_writing/             # LaTeX source and figures

scripts/                       # Analysis and baselines
    statistical_testing.py     # Statistical significance tests
    baselines/                 # Baseline implementations

preserved_data/                # Reproducibility archives
```

## Results

| Dataset | Telepathy | Text-Relay | Speedup |
|---------|-----------|------------|---------|
| SST-2   | 96.7%     | 95.5%      | 8x      |
| AG News | 90.7%     | 85.2%      | 8x      |
| TREC-6  | 89.4%     | 82.1%      | 8x      |
| GSM8K   | 0.0%      | 48.5%      | N/A     |

## Citation

```bibtex
@inproceedings{jinesh2025telepathy,
  title={Telepathy: Continuous Latent Communication for Cross-Model Inference},
  author={Jinesh, Sujeeth},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
