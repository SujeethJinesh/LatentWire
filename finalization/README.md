# LatentWire: Continuous Interlingua for Cross-Model Communication

A research framework implementing a continuous interlingua for efficient information transfer between heterogeneous language models without retokenization.

## Quick Start

```bash
# Training
bash RUN.sh train --dataset squad --epochs 24 --latent_len 32

# Evaluation
bash RUN.sh eval --checkpoint runs/checkpoint --samples 200

# Full experiment
bash RUN.sh experiment --name baseline_experiment

# SLURM submission (HPC)
bash RUN.sh slurm --job training --time 12:00:00
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU support)

### Dependencies
```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets
pip install numpy scipy
pip install tqdm wandb
pip install matplotlib seaborn
pip install pandas
pip install scikit-learn
```

### Optional Dependencies
```bash
# For LLMLingua baseline
pip install llmlingua

# For statistical testing
pip install pingouin

# For attention visualization
pip install bertviz
```

## Project Structure

```
finalization/
├── LATENTWIRE.py      # Complete Python implementation (all modules)
├── RUN.sh             # All scripts and runners
├── README.md          # This documentation
└── latex/             # LaTeX paper and figures
    ├── main.tex
    ├── figures/
    └── tables/
```

## Core Components

### 1. Training Module
- K-token teacher-forced cross-entropy loss
- Knowledge distillation from text-prompted teachers
- Per-example calibration with RMS matching
- Mixed precision training with gradient accumulation

### 2. Evaluation Module
- Multiple baselines: text, latent, token-budget, joint rescoring
- Metrics: EM, F1, NLL, first-token accuracy
- Compression analysis: wire bytes, quantization levels
- Multi-seed evaluation for statistical significance

### 3. Model Architecture
- **Encoder**: Byte-level transformer producing fixed-size latents
- **Adapters**: Model-specific linear projections with calibration
- **Bridge**: Cross-attention compression to M soft tokens
- **LM Wrappers**: Frozen Llama and Qwen models with adapter integration

### 4. Data Pipeline
- Datasets: SQuAD, HotpotQA, SST-2, AG News, TREC, GSM8k
- Dynamic batching with proper padding and masking
- Consistent BOS/EOS handling across models
- Anchor text insertion between prefix and answer

### 5. Baselines
- **Linear Probe**: Direct mapping from one model to another
- **LLMLingua**: State-of-the-art prompt compression
- **Token Budget**: Fair comparison with text truncated to M tokens
- **Joint Rescoring**: Two-model ensemble selection

## Configuration

### Training Configuration
```python
{
    "dataset": "squad",
    "samples": 87599,
    "epochs": 24,
    "batch_size": 64,
    "latent_len": 32,     # M: number of soft tokens
    "d_z": 256,           # Latent dimension
    "encoder_type": "byte",
    "first_token_ce_weight": 0.5,
    "k_token_ce": 4,      # Supervise first K tokens
    "kd_tau": 1.0,        # KD temperature
    "lr": 1e-4,
    "warmup_steps": 500
}
```

### Model Configuration
```python
{
    "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "target_model": "Qwen/Qwen2.5-7B-Instruct",
    "calibration": "embed_rms",
    "anchor_text": "Answer: ",
    "append_bos_after_prefix": true,
    "first_token_top_p": 0.95,
    "first_token_temperature": 0.7,
    "eos_ban_steps": 4
}
```

## Key Innovations

### 1. K-Token Supervision
Instead of only supervising the first token, we supervise K tokens to provide richer learning signal:
```python
loss = sum([ce_loss(logits[:, i], labels[:, i]) for i in range(K)]) / K
```

### 2. Per-Example Calibration
Match embedding statistics per example, not per batch:
```python
scale = target_rms / (embed_rms + 1e-8)
calibrated = embeddings * scale
```

### 3. Proper PAD Token Masking
Prevent gradient contamination from padding:
```python
labels[input_ids == pad_token_id] = -100
attention_mask[input_ids == pad_token_id] = 0
```

### 4. Consistent BOS Policy
Ensure BOS token alignment between training and evaluation:
```python
if append_bos_after_prefix:
    ids = torch.cat([prefix_ids, bos_token, answer_ids])
```

## Experiments

### Main Experiments

1. **Baseline Establishment**
   ```bash
   bash RUN.sh experiment --name baseline --dataset squad
   ```

2. **Compression Analysis**
   ```bash
   bash RUN.sh experiment --name compression --latent_len 16,32,64
   ```

3. **Cross-Dataset Generalization**
   ```bash
   bash RUN.sh experiment --name generalization --dataset hotpot,sst2,agnews
   ```

4. **Statistical Significance**
   ```bash
   python LATENTWIRE.py statistical_test --exp1 runs/exp1 --exp2 runs/exp2
   ```

### SLURM Submission (HPC)

For HPC environments with SLURM:

```bash
# Basic training job
bash RUN.sh slurm --job training --gpus 4 --time 12:00:00

# Evaluation job
bash RUN.sh slurm --job eval --gpus 1 --time 2:00:00

# Full experiment pipeline
bash RUN.sh slurm --job experiment --gpus 4 --time 24:00:00
```

## Results

### Current Performance (Phase A)
- **Text Baseline**: F1 ~0.80-0.85
- **Latent (M=32)**: F1 ~0.10-0.20, FirstTok@1 ~12-20%
- **Compression**: 4-6× reduction in wire bytes
- **Speed**: 2-3× faster inference than text baseline

### Target Metrics
- FirstTok@1: 20-30% at M=32
- F1: 0.30-0.40 with honest compression
- Compression: ≥4× while maintaining quality
- Cross-model transfer without retokenization

## Troubleshooting

### Common Issues

1. **OOM Errors**
   - Reduce batch size: `--batch_size 32`
   - Enable gradient accumulation: `--grad_accum 4`
   - Use mixed precision: `--fp16`

2. **PAD Token Issues**
   - Ensure proper masking in labels
   - Check tokenizer configuration
   - Verify BOS/EOS policies

3. **Low First-Token Accuracy**
   - Increase K in k-token supervision
   - Adjust KD temperature
   - Check calibration method

4. **Checkpoint Loading**
   - Verify path exists
   - Check compatibility with current code
   - Use `--fresh_eval` for clean evaluation

## Development

### Running Tests
```bash
# Full test suite
python LATENTWIRE.py test

# Specific module
python LATENTWIRE.py test --module training

# Quick smoke test
python LATENTWIRE.py test --quick
```

### Adding New Datasets
Datasets should implement the base interface:
```python
class MyDataset(BaseDataset):
    def __getitem__(self, idx):
        return {
            "prefix": "...",
            "answer": "...",
            "full_text": prefix + anchor + answer
        }
```

### Adding New Models
Models need an adapter configuration:
```python
if "mymodel" in model_id.lower():
    model_dim = 4096  # Model's hidden dimension
```

## Citation

```bibtex
@article{latentwire2025,
  title={LatentWire: Continuous Interlingua for Cross-Model Communication},
  author={LatentWire Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with clear description

## Contact

For questions or collaboration:
- GitHub Issues: [Link to repository]
- Email: [Contact email]

---

## Appendix A: Detailed Module Documentation

### Data Module (lines 2000-3500 in LATENTWIRE.py)
- SQuAD, HotpotQA, SST-2, AG News dataset loaders
- Dynamic collation with proper padding
- Support for distributed training
- Context building for multi-hop QA

### Models Module (lines 3500-5500 in LATENTWIRE.py)
- ByteEncoder: Transformer-based byte-level encoder
- LatentEncoder: Cross-attention compression
- ModelAdapter: Per-model projection layers
- BridgeModel: Complete encoder-adapter pipeline
- ElasticGPUConfig: Automatic batch size optimization

### Training Module (lines 6500-8000 in LATENTWIRE.py)
- Multi-objective training loop
- Gradient accumulation and mixed precision
- Checkpoint saving and recovery
- Wandb integration for experiment tracking

### Evaluation Module (lines 8000-10000 in LATENTWIRE.py)
- Comprehensive evaluation suite
- Multiple baseline comparisons
- Metric computation and aggregation
- Result visualization and reporting

## Appendix B: Configuration Examples

### Minimal Training
```json
{
  "dataset": "squad",
  "samples": 1000,
  "epochs": 1,
  "batch_size": 8,
  "latent_len": 32
}
```

### Full Experiment
```json
{
  "dataset": "squad",
  "samples": 87599,
  "epochs": 24,
  "batch_size": 64,
  "latent_len": 32,
  "d_z": 256,
  "encoder_type": "byte",
  "lr": 1e-4,
  "warmup_steps": 500,
  "first_token_ce_weight": 0.5,
  "k_token_ce": 4,
  "kd_tau": 1.0,
  "calibration": "embed_rms",
  "anchor_text": "Answer: ",
  "append_bos_after_prefix": true,
  "eval_samples": 200,
  "eval_seeds": [42, 123, 456]
}
```

### SLURM Configuration
```bash
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
```

## Appendix C: Performance Optimization

### GPU Memory Optimization
1. Use gradient checkpointing for large models
2. Enable CPU offloading for optimizer states
3. Reduce sequence length during training
4. Use mixed precision (fp16/bf16)

### Training Speed Optimization
1. Pre-tokenize datasets offline
2. Use DataLoader prefetching
3. Enable torch.compile for PyTorch 2.0+
4. Optimize num_workers for I/O

### Inference Optimization
1. Cache encoder outputs for fixed prefixes
2. Use batch inference for multiple samples
3. Quantize latent representations (int8/int4)
4. Enable KV-cache for generation

---

*Last Updated: January 2025*
*Version: 1.0.0 (Consolidated)*