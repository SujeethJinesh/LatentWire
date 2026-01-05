# Memory-Safe Training Guide for LatentWire

## Critical Memory Requirements

When training large language models with gradient-based optimization, the memory requirements extend far beyond just the model parameters. This guide provides accurate memory calculations to prevent OOM (Out of Memory) errors.

## Memory Breakdown for Llama-3.1-8B

### Fixed Memory Components

| Component | Memory (GB) | Description |
|-----------|------------|-------------|
| **Model Parameters** | 14 GB | Base model weights (8B params × 2 bytes for bf16) |
| **Optimizer State (Adam)** | 28 GB | Momentum + variance buffers (2× model size) |
| **Gradients** | 14 GB | Gradient storage (same size as model) |
| **CUDA Overhead** | 2 GB | CUDA context, kernels, workspace |
| **Safety Margin** | 2 GB | Buffer for unexpected allocations |
| **Total Fixed** | **60 GB** | Minimum required before any batch processing |

### Variable Memory (Per Batch Sample)

| Component | Memory per Sample | Description |
|-----------|------------------|-------------|
| **Activations** | ~0.5-1 GB | Forward/backward pass intermediates |
| **Attention Maps** | ~0.3 GB | Multi-head attention storage |
| **Logits** | ~0.2 GB | Output vocabulary projections |
| **Total per Sample** | **~1 GB** | Conservative estimate |

## GPU Configuration Guidelines

### H100 80GB
- **Available Memory**: 80GB - 60GB = 20GB
- **Safe Batch Size**: 20 (using 20GB for activations)
- **Gradient Accumulation**: Not needed for single GPU
- **Configuration**:
  ```bash
  --batch_size 20
  --gradient_accumulation_steps 1
  ```

### A100 80GB
- **Available Memory**: 80GB - 60GB = 20GB
- **Safe Batch Size**: 20
- **Configuration**: Same as H100

### A100 40GB
- **Available Memory**: 40GB - 60GB = -20GB ❌
- **Issue**: Cannot fit model + Adam optimizer
- **Solutions**:
  1. Use gradient checkpointing to reduce activation memory
  2. Use Adafactor optimizer (only 0.5× model size overhead)
  3. Use model parallelism across multiple GPUs
  4. Use smaller batch size with heavy gradient accumulation

### V100 32GB
- **Available Memory**: 32GB - 60GB = -28GB ❌
- **Issue**: Severely memory constrained
- **Solution**: Not recommended for 8B models with Adam

## Memory Optimization Techniques

### 1. Gradient Checkpointing
Reduces activation memory by ~50% at the cost of ~30% slower training:
```python
model.gradient_checkpointing_enable()
```

### 2. Mixed Precision Training
Use bf16/fp16 instead of fp32 (already assumed in calculations above):
```python
--bf16 true
--fp16 false  # bf16 preferred for stability
```

### 3. Optimizer Choice

| Optimizer | Memory Overhead | Notes |
|-----------|----------------|-------|
| SGD | 0× model size | No momentum/variance |
| Adafactor | 0.5× model size | Memory efficient |
| Adam/AdamW | 2× model size | Standard but memory heavy |
| LAMB | 2× model size | Similar to Adam |

### 4. Gradient Accumulation
Simulate larger batch sizes without increasing memory:
```bash
# Effective batch size 80 with only 20 in memory
--batch_size 20
--gradient_accumulation_steps 4
```

## Common Mistakes to Avoid

### ❌ Incorrect Memory Calculation
```bash
# WRONG: Only accounts for model, not optimizer
available = gpu_memory - model_size - safety_margin
# This would suggest 80 - 14 - 4 = 62GB available
# Result: OOM when Adam allocates 28GB more
```

### ✅ Correct Memory Calculation
```bash
# RIGHT: Accounts for all components
available = gpu_memory - model_size - optimizer_state - gradients - overhead - margin
# This gives 80 - 14 - 28 - 14 - 2 - 2 = 20GB available
# Result: Safe training without OOM
```

## Testing Memory Configuration

Use the provided memory calculator to verify your configuration:

```bash
# Test your GPU configuration
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 80 \
    --num_gpus 4 \
    --optimizer adamw
```

## Monitoring Memory Usage

During training, monitor GPU memory with:
```bash
# Watch memory usage
watch -n 1 nvidia-smi

# Or use the monitoring script
bash finalization/monitor_experiment.sh
```

## Quick Reference Table

| GPU | Memory | Safe Batch Size | Gradient Accum | Effective Batch |
|-----|--------|----------------|----------------|-----------------|
| H100 | 80GB | 20 | 1 | 20 |
| A100 | 80GB | 20 | 1 | 20 |
| A100 | 40GB | 1* | 32 | 32 |
| V100 | 32GB | ❌ | - | - |

*Requires gradient checkpointing or alternative optimizer

## Key Takeaways

1. **Adam optimizer doubles memory requirements** - Always account for optimizer state
2. **60GB fixed overhead** for Llama-8B with Adam (model + optimizer + gradients)
3. **A100 40GB GPUs** need special handling (gradient checkpointing or different optimizer)
4. **Conservative batch sizes** prevent OOM and provide stable training
5. **Monitor actively** - Use nvidia-smi to verify memory usage matches predictions

## Update History

- 2024-01: Initial calculations assuming only model memory
- 2024-01: **FIXED** - Added optimizer state memory (28GB for Adam)
- 2024-01: Updated all batch size recommendations for safety