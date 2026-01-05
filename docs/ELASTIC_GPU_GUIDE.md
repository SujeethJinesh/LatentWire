# Elastic GPU Configuration Guide

## Overview

The Elastic GPU Configuration system automatically adapts LatentWire training to work optimally with 1-4 GPUs, eliminating manual tuning when switching between different hardware configurations.

## Key Features

### Automatic Detection
- Detects number of available GPUs via `torch.cuda.device_count()`
- Identifies GPU type (H100, A100, etc.) and memory capacity
- Adjusts memory calculations based on GPU specifications

### Dynamic Configuration
The system automatically configures:
- **Batch size**: Maximized for available memory
- **Gradient accumulation**: Maintains effective batch size
- **Parallelism strategy**: Model-parallel vs data-parallel
- **Memory allocation**: Optimized for model size and activations

## Configuration Strategies by GPU Count

### 1 GPU Configuration
- **Strategy**: Single GPU with gradient accumulation
- **Batch size**: Memory-limited (typically 32-48)
- **Gradient accumulation**: 2-4 steps to reach effective batch size
- **Use case**: Development, small-scale experiments

```bash
python latentwire/train.py --elastic_gpu --elastic_base_batch 64
```

### 2 GPU Configuration
- **H100s (80GB each)**:
  - Strategy: Data-parallel (DDP)
  - Both models replicated on both GPUs
  - Maximum throughput via data parallelism

- **A100s or smaller**:
  - Strategy: Model splitting
  - Llama on GPU 0, Qwen on GPU 1
  - Optimizes memory usage

```bash
CUDA_VISIBLE_DEVICES=0,1 python latentwire/train.py --elastic_gpu
```

### 3 GPU Configuration
- **Strategy**: Hybrid approach
- **Layout**:
  - Llama distributed across GPUs 0-1
  - Qwen on GPU 2
  - Encoder shares GPU 0 with Llama
- **Benefits**: Balances memory and compute

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python latentwire/train.py --elastic_gpu
```

### 4 GPU Configuration
- **H100 Cluster**:
  - Strategy: Full data-parallel (DDP)
  - Batch size: 256 (64 per GPU)
  - Maximum throughput configuration

- **A100 Cluster**:
  - Strategy: Hybrid model + data parallel
  - Llama on GPUs 0-1, Qwen on GPUs 2-3
  - 2-way data parallelism

```bash
python latentwire/train.py --elastic_gpu  # Uses all visible GPUs
```

## Usage

### Basic Usage
Enable elastic GPU configuration with a single flag:

```bash
python latentwire/train.py \
    --elastic_gpu \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples 10000 \
    --epochs 3
```

### Advanced Options
Fine-tune the elastic behavior:

```bash
python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64      # Base batch size for single GPU (default: 64)
    --elastic_target_util 0.75   # Target GPU utilization (default: 0.75)
    ...
```

### HPC/SLURM Usage
Example SLURM script with elastic configuration:

```bash
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --mem=256GB

python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --samples 87599 \
    --epochs 24 \
    ...
```

## Memory Calculations

The system uses these memory estimates:

| Component | Memory Usage |
|-----------|-------------|
| Llama-8B model | ~14 GB |
| Qwen-7B model | ~13 GB |
| Activations per batch item | ~0.75 GB (seq_len=256) |
| Optimizer states | ~4 GB per model |

Target utilization is set to 75% by default for stability.

## Testing Configuration

### Test Script
Run the test script to see your configuration:

```bash
python scripts/test_elastic_gpu.py
```

### Demo Script
Run a quick training demo:

```bash
bash scripts/run_elastic_gpu_demo.sh
```

### Simulating Different GPU Counts
Test with different GPU visibility:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/test_elastic_gpu.py

# Dual GPU
CUDA_VISIBLE_DEVICES=0,1 python scripts/test_elastic_gpu.py

# All GPUs
python scripts/test_elastic_gpu.py
```

## Performance Expectations

| GPU Configuration | Expected Throughput | Effective Batch Size |
|-------------------|--------------------|--------------------|
| 1× A100-40GB | ~50 samples/sec | 64 (via accumulation) |
| 2× A100-40GB | ~80 samples/sec | 96 |
| 4× H100-80GB | ~320 samples/sec | 256 |
| 4× A100-40GB | ~160 samples/sec | 128 |

## Troubleshooting

### Out of Memory (OOM)
If you encounter OOM errors:
1. Reduce `--elastic_base_batch` (try 32 or 48)
2. Lower `--elastic_target_util` to 0.6
3. Enable `--sequential_models` to process models one at a time

### Suboptimal Performance
If throughput is lower than expected:
1. Increase `--elastic_target_util` to 0.85 (if memory allows)
2. Ensure you're using the correct dtype (`TORCH_DTYPE=bf16` for H100s)
3. Check GPU utilization with `nvidia-smi`

### Configuration Not Applied
Ensure:
1. `--elastic_gpu` flag is present
2. CUDA is available (`torch.cuda.is_available()`)
3. GPUs are visible (check `CUDA_VISIBLE_DEVICES`)

## Implementation Details

The elastic configuration is implemented in:
- `latentwire/train.py`: `ElasticGPUConfig` class
- Detects hardware at runtime
- Overrides batch size and device placement arguments
- Maintains backward compatibility with manual configuration

## Future Enhancements

Planned improvements:
- [ ] Full DDP support with torchrun
- [ ] FSDP for very large models
- [ ] Dynamic batch size adjustment during training
- [ ] Multi-node elastic scaling
- [ ] Automatic mixed precision selection

## Related Documentation

- [H100 Optimization Guide](H100_OPTIMIZATION.md)
- [CLAUDE.md](../CLAUDE.md) for workflow guidelines
- [LOG.md](../LOG.md) for implementation history