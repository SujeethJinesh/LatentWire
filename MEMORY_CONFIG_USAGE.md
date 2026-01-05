# Memory-Safe Configuration Usage Guide

## Overview
The `telepathy/memory_configs.py` module provides automatic memory-safe batch size and gradient accumulation configuration for training telepathy bridges on 80GB H100 GPUs.

## Quick Start

### 1. Local Testing (MacBook)
```bash
# Test configuration for your model pair
python3 telepathy/example_memory_safe_training.py \
    --source_model "meta-llama/Llama-3.2-3B-Instruct" \
    --target_model "mistralai/Mistral-7B-Instruct-v0.3"
```

### 2. HPC Training (SLURM)
```bash
# On HPC:
cd /projects/m000066/sujinesh/LatentWire
git pull
sbatch telepathy/submit_memory_safe_bridge.slurm
```

### 3. In Your Python Code
```python
from telepathy.memory_configs import get_memory_safe_config

config = get_memory_safe_config(
    source_model="meta-llama/Llama-3.2-3B-Instruct",
    target_model="mistralai/Mistral-7B-Instruct-v0.3"
)

batch_size = config['batch_size']
grad_accum = config['gradient_accumulation_steps']
print(f"Safe config: batch_size={batch_size}, grad_accum={grad_accum}")
print(f"Memory usage: {config['estimated_memory_gb']:.1f} GB / 80 GB")
```

## Supported Models

### Source Models (Llama Family)
- `meta-llama/Llama-3.2-1B-Instruct` (1.24B params)
- `meta-llama/Llama-3.2-3B-Instruct` (3.21B params)
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (8.03B params)

### Target Models
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3` (7.24B params)
- **Qwen**: `Qwen/Qwen2.5-1.5B-Instruct` (1.54B params)
- **Qwen**: `Qwen/Qwen2.5-3B-Instruct` (3.09B params)
- **Qwen**: `Qwen/Qwen2.5-7B-Instruct` (7.62B params)

## Recommended Configurations

| Source Model | Target Model | Batch Size | Grad Accum | Memory Usage |
|-------------|--------------|------------|------------|--------------|
| Llama-8B | Mistral-7B | 2 | 4 | ~46 GB |
| Llama-3B | Mistral-7B | 4 | 2 | ~43 GB |
| Llama-3B | Qwen-3B | 8 | 1 | ~47 GB |
| Llama-1B | Qwen-1.5B | 16 | 1 | ~55 GB |
| Llama-1B | Mistral-7B | 8 | 1 | ~58 GB |

## Memory Breakdown

The configuration accounts for:
1. **Model Parameters** (bfloat16 = 2 bytes/param)
2. **Bridge Architecture** (~1-4 GB depending on hidden dimensions)
3. **Activation Memory** (scales with batch size and sequence length)
4. **Safety Margin** (20% buffer to prevent OOM)

## Customization

### Adjust Safety Margin
```python
config = get_memory_safe_config(
    source_model="...",
    target_model="...",
    safety_margin=0.25  # 25% buffer instead of 20%
)
```

### Force Single-Model Mode
```python
config = get_memory_safe_config(
    source_model="meta-llama/Llama-3.2-3B-Instruct",
    target_model=None,  # Single-model bridge
    force_single_model=True
)
```

### Custom Sequence Length
```python
config = get_memory_safe_config(
    source_model="...",
    target_model="...",
    max_length=2048  # Longer sequences need more memory
)
```

## Files Created

1. **`telepathy/memory_configs.py`** - Core configuration module
2. **`telepathy/example_memory_safe_training.py`** - Example usage script
3. **`telepathy/submit_memory_safe_bridge.slurm`** - SLURM script with auto-configuration

## Troubleshooting

### "Models too large" Error
- Reduce sequence length: `max_length=1024`
- Increase safety margin tolerance: `safety_margin=0.15`
- Use smaller model combinations

### OOM During Training
- Configuration provides estimates; actual usage may vary
- Reduce batch size manually if needed
- Check for memory leaks in custom code

### Different GPU Memory
```python
config = get_memory_safe_config(
    source_model="...",
    target_model="...",
    gpu_memory_gb=40.0  # For A100 40GB
)
```

## Integration with Existing Scripts

To add memory safety to existing training scripts:

```python
# At the top of your training script
from telepathy.memory_configs import get_memory_safe_config

# In your argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...

    # Override with memory-safe config
    config = get_memory_safe_config(args.source_model, args.target_model)
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = config['gradient_accumulation_steps']

    return args
```

## Next Steps

1. Test configuration locally with `example_memory_safe_training.py`
2. Submit jobs using `submit_memory_safe_bridge.slurm`
3. Monitor memory usage in logs to refine estimates
4. Add new models to `MODEL_PARAMS` and `MODEL_HIDDEN_DIM` as needed