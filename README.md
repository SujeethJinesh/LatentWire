# Telepathy: Neural Bridges for Cross-Model LLM Communication

[![CI](https://github.com/SujeethJinesh/LatentWire/actions/workflows/ci.yml/badge.svg)](https://github.com/SujeethJinesh/LatentWire/actions/workflows/ci.yml)
[![Tests](https://github.com/SujeethJinesh/LatentWire/actions/workflows/tests.yml/badge.svg)](https://github.com/SujeethJinesh/LatentWire/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/SujeethJinesh/LatentWire/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Telepathy** enables direct communication between heterogeneous Large Language Models through learned soft tokens, eliminating the need for expensive text generation. By extracting hidden states from a sender model (Llama 3.1 8B) and transforming them via a neural bridge, receiver models (Mistral 7B) can process rich continuous representations 22× faster than traditional text-relay approaches while achieving up to 96.0% accuracy on multi-class classification tasks.

This repository also contains **LatentWire**, our initial research framework that attempted universal text communication through compressed representations. While LatentWire failed to achieve coherent text reconstruction, it provided crucial insights that led to Telepathy's focused approach on task-specific transfer.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Testing](#-testing)
- [Architecture](#-architecture)
- [Training Pipeline](#-training-pipeline)
- [Evaluation](#-evaluation)
- [Development](#-development)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)

## 🎯 Overview

### Telepathy: Production-Ready Neural Bridges

Telepathy solves the fundamental inefficiency in multi-agent AI systems where LLMs must generate text autoregressively for communication. Our neural bridge architecture:
- **Eliminates text generation**: Direct transfer of hidden states through learned soft tokens
- **22× faster**: 37ms vs 835ms for text-relay approaches
- **Super-additive accuracy**: Achieves higher accuracy than either model alone on TREC-6
- **Cross-model transfer**: Llama 3.1 8B → Mistral 7B with Perceiver Resampler
- **Inverse token scaling**: Fewer soft tokens (16) outperform more tokens (128)

### Key Results

| Dataset | Telepathy | Text-Relay | Prompt-Tuning | Individual Best | Speedup |
|---------|-----------|------------|---------------|-----------------|---------|
| AG News | 89.5%     | 70.0%      | 82.5%         | 70.5% (Mistral) | 22×     |
| TREC-6  | 96.0%     | 47.0%      | 90.0%         | 67.5% (Mistral) | 22×     |
| SST-2*  | 49.5%     | 95.0%      | 97.5%         | 96.5% (Llama)   | N/A     |

*SST-2 represents a failure mode where binary classification tasks fail completely

### LatentWire: Research Framework

The repository also includes LatentWire, our initial research attempting universal text communication:
- **Goal**: Compress arbitrary text into soft tokens for cross-model communication
- **Result**: Failed to achieve coherent text reconstruction (BLEU < 0.05)
- **Learning**: High-level semantic transfer succeeds even when exact reconstruction fails
- **Impact**: Motivated Telepathy's focused approach on task-specific transfer

## ✨ Key Features

### Core Capabilities
- 🔄 **Heterogeneous Model Support**: Works with Llama and Qwen families
- 📊 **Comprehensive Diagnostics**: Real-time loss tracking and gradient analysis
- 🎛️ **Modular Architecture**: Pluggable encoders, adapters, and training objectives
- 🚀 **Scalable Training**: Supports both smoke tests and hero-scale experiments

### Implemented Techniques
- **Soft Prompt Tuning**: Continuous prompts in embedding space
- **K-Token Supervision**: Multi-token teacher forcing
- **Knowledge Distillation**: Transfer from text-prompted teachers
- **Deep Prefix Tuning**: Per-layer key-value prompts
- **Gist Reconstruction**: Auxiliary loss for information preservation
- **Latent Adapters**: Cross-attention mechanisms
- **Embedding Calibration**: Statistical matching of embeddings

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For LoRA/Prefix tuning
pip install peft

# For 4-bit quantization (Linux/CUDA only)
pip install bitsandbytes

# For development
pip install black isort ruff mypy
```

## 🚀 Quick Start

### 1. Run Smoke Tests

Test the setup with minimal resources:

```bash
# Single model (Llama only)
bash scripts/run_llama_single.sh

# Multi-model (Llama + Qwen)
bash scripts/run_scoped_softprompt_multi.sh

# With specific configuration
python -m latentwire.cli.train --config configs/smoke/base.json
```

### 2. Run Embedding Baselines

Validate the inputs_embeds mechanism:

```bash
bash scripts/run_embedding_baselines.sh
```

### 3. Run Full Training

```bash
# Hero-scale training
bash scripts/run_llama_single.sh --hero

# Custom hyperparameters
export LATENT_LEN_LIST=32,48,64
export D_Z_LIST=256,384
bash scripts/run_llama_single.sh
```

## 📝 Quick Examples

### Training a Basic Model
```python
from latentwire.train import main as train_main

# Train with basic configuration
args = [
    "--llama_id", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "--samples", "1000",
    "--epochs", "3",
    "--batch_size", "8",
    "--latent_len", "32",
    "--d_z", "256",
    "--dataset", "squad"
]
train_main(args)
```

### Evaluating a Checkpoint
```python
from latentwire.eval import main as eval_main

# Evaluate trained model
args = [
    "--ckpt", "runs/my_run/epoch3",
    "--samples", "200",
    "--dataset", "squad",
    "--fresh_eval"
]
results = eval_main(args)
print(f"F1 Score: {results['latent']['f1']:.4f}")
```

### Using Specific Features
```bash
# Train with LoRA adaptation
python latentwire/train.py \
    --config configs/smoke/lora.json \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

# Train with Gist reconstruction head
python latentwire/train.py \
    --config configs/smoke/gist_head.json \
    --use_gist_head \
    --gist_weight 0.1 \
    --gist_target_len 48

# Train with Deep Prefix tuning
python latentwire/train.py \
    --config configs/smoke/deep_prefix.json \
    --use_deep_prefix \
    --prefix_len 10
```

### Analyzing Results
```python
import json
from pathlib import Path

# Load training diagnostics
diagnostics = []
with open("runs/my_run/diagnostics.jsonl") as f:
    for line in f:
        diagnostics.append(json.loads(line))

# Plot training curves
import matplotlib.pyplot as plt

steps = [d["step"] for d in diagnostics]
losses = [d["loss"] for d in diagnostics]
first_acc = [d.get("first_acc", 0) for d in diagnostics]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(steps, first_acc)
plt.xlabel("Step")
plt.ylabel("First Token Accuracy")
plt.title("First Token Accuracy")
plt.show()
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
./scripts/run_tests.sh

# Quick tests only (fast subset)
./scripts/run_tests.sh --quick

# With coverage report
./scripts/run_tests.sh --coverage

# Specific test file
pytest tests/features/test_gist_head.py -v

# Run in parallel (faster)
pytest -n auto

# With verbose output
pytest -v --tb=short
```

### Test Coverage

Our comprehensive test suite includes:
- ✅ **89 tests** passing with >89% code coverage
- ✅ **8 smoke configurations** validated for all features
- ✅ **7 embedding baseline tests** for inputs_embeds validation
- ✅ **Feature-specific tests**:
  - LoRA adaptation (with/without peft)
  - Prefix tuning mechanisms
  - Deep prefix KV prompts
  - Gist reconstruction head
  - Latent adapters (cross-attention)
  - Latent refiner (transformer smoother)
  - Coprocessor KV manipulation
- ✅ **Integration tests** for end-to-end pipelines
- ✅ **Configuration tests** for JSON loading and CLI conversion

Test organization:
```
tests/
├── test_embedding_baseline.py  # Core inputs_embeds tests
├── features/
│   ├── test_gist_head.py      # Gist reconstruction tests
│   ├── test_latent_refiner.py # Refiner module tests
│   └── test_prefix_tuning.py  # Prefix tuning tests
└── integration/
    └── test_smoke_configs.py   # Config integration tests
```

### CI/CD Pipeline

Our automated testing runs on **GitHub Actions** with:

#### Continuous Integration (ci.yml)
- **Multi-version testing**: Python 3.9, 3.10, 3.11
- **Coverage reporting**: Automated coverage badge updates
- **Code quality checks**:
  - Black formatting verification
  - Ruff linting
  - Type checking with mypy
- **Security scanning**: Dependencies and vulnerabilities
- **Scheduled runs**: Weekly regression testing

#### Quick Tests (tests.yml)
- **Fast feedback**: Runs on every push and PR
- **Core test suite**: Essential tests only
- **Merge protection**: Blocks merging if tests fail

View test results: [GitHub Actions](https://github.com/SujeethJinesh/LatentWire/actions)

## 🏗️ Architecture

### System Overview

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│   Encoder   │────▶│  Latent  │────▶│   Adapter   │
│  (Shared)   │     │   Wire   │     │ (Per-Model) │
└─────────────┘     └──────────┘     └─────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   LLM (Llama/Qwen)     │
              │   inputs_embeds API    │
              └────────────────────────┘
```

### Key Components

| Component | Description | File |
|-----------|-------------|------|
| **Encoder** | Transforms text to latent representations | `models.py` |
| **Adapter** | Maps latents to model-specific embeddings | `models.py` |
| **LMWrapper** | Unified interface for different LLMs | `models.py` |
| **Loss Functions** | K-token CE, KD, alignment losses | `losses.py` |
| **Feature Registry** | Manages optional features (LoRA, etc.) | `feature_registry.py` |

### Supported Features

| Feature | Flag | Description |
|---------|------|-------------|
| **LoRA** | `--use_lora` | Low-rank adaptation for parameter efficiency |
| **Prefix Tuning** | `--use_prefix` | Learnable prefix tokens |
| **Deep Prefix** | `--use_deep_prefix` | Per-layer KV prompts |
| **Latent Adapters** | `--use_latent_adapters` | Cross-attention layers |
| **Coprocessor** | `--use_coprocessor` | KV cache manipulation |
| **Gist Head** | `--use_gist_head` | Reconstruction auxiliary loss |
| **Latent Refiner** | `--use_latent_refiner` | Transformer smoother |

## 📊 Training Pipeline

### Stage A: Latent Fitting
- Text/latent alternating warmup
- First-token supervision
- Gradient diagnostics

### Stage B: Prefix Training
- Latent-only batches
- Extended schedules
- Per-loss monitoring

### Stage C: Evaluation
- Compare latent vs text
- Measure compression ratio
- Compute task metrics

### Configuration Files

```bash
configs/
├── smoke/          # Quick test configs
│   ├── base.json
│   ├── lora.json
│   ├── prefix.json
│   └── ...
└── baseline/       # Baseline configs
    └── embedding_baselines.json
```

### Monitoring Training

```python
# Read diagnostics
import json
from pathlib import Path

# Load diagnostics stream
with open("runs/my_run/diagnostics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Analyze training progress
for m in metrics:
    print(f"Step {m['step']}: Loss={m['loss']:.4f}, FirstAcc={m.get('first_acc', 0):.2%}")
```

## 📈 Evaluation

### Metrics Tracked

| Metric | Description | Target |
|--------|-------------|--------|
| **F1 Score** | Token-level F1 on answers | >60% of baseline |
| **Exact Match** | Perfect answer matches | >40% of baseline |
| **First Token Acc** | First token prediction accuracy | >20% |
| **NLL/Token** | Negative log-likelihood | <2× baseline |
| **Compression** | Reduction in byte size | ≥4× |

### Running Evaluation

```bash
# Evaluate checkpoint
python -m latentwire.cli.eval \
    --ckpt runs/my_run/ckpt \
    --dataset squad \
    --samples 200

# Compare baselines
python scripts/compare_baselines.py \
    --latent runs/latent/eval.json \
    --text runs/text/eval.json
```

## 💻 Development

### Project Structure

```
LatentWire/
├── latentwire/           # Main package
│   ├── models.py         # Core models (Encoder, Adapter, LMWrapper)
│   ├── losses.py         # Loss functions (K-token CE, KD, alignment)
│   ├── train.py          # Training loop with Phase A improvements
│   ├── eval.py           # Evaluation with baselines
│   ├── data.py           # Dataset loading (SQuAD, HotpotQA)
│   ├── metrics.py        # EM/F1 scoring, NLL computation
│   ├── prefix_utils.py   # Calibration, BOS policy, anchoring
│   ├── common.py         # Chat templates, text utilities
│   └── features/         # Optional features (LoRA, Prefix, etc.)
├── tests/                # Comprehensive test suite
│   ├── test_embedding_baseline.py
│   ├── features/         # Feature-specific tests
│   └── integration/      # End-to-end tests
├── configs/              # Configuration files
│   ├── smoke/            # Quick test configs (8 variants)
│   └── baseline/         # Baseline configurations
├── scripts/              # Runner scripts
│   ├── run_tests.sh      # Test runner with coverage
│   ├── run_pipeline.sh   # Training/eval pipeline
│   └── run_*.sh          # Various experiment scripts
├── .github/              # CI/CD configuration
│   └── workflows/        # GitHub Actions
└── docs/                 # Documentation

```

### Development Environment Setup

#### 1. Initial Setup
```bash
# Clone and setup
git clone https://github.com/SujeethJinesh/LatentWire.git
cd LatentWire

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

#### 2. Environment Variables
```bash
# Add to your .bashrc/.zshrc or create .env file
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac MPS
export TOKENIZERS_PARALLELISM=false   # Avoid tokenizer warnings
```

### Code Quality Tools

#### Formatting & Linting
```bash
# Auto-format code
black latentwire tests
isort latentwire tests

# Check formatting without changes
black --check latentwire tests

# Lint with ruff
ruff check latentwire tests

# Fix auto-fixable issues
ruff check --fix latentwire tests
```

#### Type Checking
```bash
# Run type checking
mypy latentwire

# With stricter settings
mypy --strict latentwire
```

#### Testing
```bash
# Run all tests with coverage
./scripts/run_tests.sh --coverage

# Run specific test modules
pytest tests/features/ -v
pytest tests/integration/ -v

# Run with different verbosity
pytest -q     # Quiet
pytest -v     # Verbose
pytest -vv    # Very verbose

# Debug test failures
pytest --pdb  # Drop into debugger on failure
pytest --lf   # Run only last failed tests
```

### Debugging Techniques

#### 1. Training Diagnostics
```bash
# Monitor training in real-time
tail -f runs/*/diagnostics.jsonl | jq '.'

# Parse specific metrics
cat runs/*/diagnostics.jsonl | jq '.loss'

# Check gradient norms
cat runs/*/diagnostics.jsonl | jq '.grad_norm'
```

#### 2. Debug Flags
```python
# Add debug output to training
python latentwire/train.py --debug

# Profile memory usage
python latentwire/train.py --profile_memory

# Log tensor shapes
python latentwire/train.py --log_shapes
```

#### 3. Interactive Debugging
```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use IPython for better experience
from IPython import embed; embed()

# For PyTorch tensors
tensor.shape, tensor.dtype, tensor.device
tensor.min(), tensor.max(), tensor.mean()
```

#### 4. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce batch_size, use gradient accumulation |
| NaN losses | Check learning rate, add gradient clipping |
| Slow training | Enable mixed precision, use DataLoader workers |
| Tokenization errors | Verify BOS/EOS handling, check padding |
| Test failures | Clear cache, check dependencies, update fixtures |

### Performance Profiling

```bash
# Profile CPU usage
python -m cProfile -o profile.prof latentwire/train.py
python -m pstats profile.prof

# Profile GPU usage
nvidia-smi dmon -s um -d 1

# Memory profiling
python -m memory_profiler latentwire/train.py

# PyTorch profiler
# Add to code: with torch.profiler.profile() as prof: ...
```

### Best Practices

1. **Version Control**
   - Commit frequently with clear messages
   - Use conventional commits (feat:, fix:, docs:, etc.)
   - Keep commits focused and atomic

2. **Testing**
   - Write tests for new features
   - Run tests before committing
   - Maintain >80% code coverage

3. **Documentation**
   - Update docstrings for API changes
   - Keep README current
   - Document non-obvious design decisions

4. **Experiment Tracking**
   - Log all hyperparameters
   - Save diagnostics and checkpoints
   - Document results in LOG.md


## 📚 Documentation

- **[Research Proposal](RESEARCH_PROPOSAL.md)** - Detailed technical approach
- **[Experiment Log](LOG.md)** - Chronological experiment results
- **[Phase 2 Plan](PLAN.md)** - Roadmap for accuracy improvements
- **[API Reference](docs/api.md)** - Code documentation

## 📝 Citation

If you use LatentWire in your research, please cite:

```bibtex
@software{latentwire2024,
  title = {LatentWire: Shared Continuous Representations for Heterogeneous LLMs},
  author = {Jinesh, Sujeeth},
  year = {2024},
  url = {https://github.com/SujeethJinesh/LatentWire}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for model implementations
- PyTorch team for the deep learning framework
- Research papers that inspired this work (see [papers/](papers/))

---

<p align="center">
  <b>LatentWire Research Project</b><br>
  Shared Continuous Representations for Heterogeneous LLMs
</p>