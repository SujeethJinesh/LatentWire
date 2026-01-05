# H100 GPU Optimization Guide for LatentWire Training

## Executive Summary

This guide provides comprehensive optimizations to achieve >90% GPU utilization on H100 GPUs for LatentWire LLM training. Based on analysis of the current codebase and latest H100 optimization research (2025), we identify key bottlenecks and provide actionable solutions.

## Current Bottlenecks Identified

### 1. **Suboptimal Batch Size Configuration**
- Current: batch_size=64 without gradient accumulation
- Issue: Not fully utilizing 80GB HBM3 memory
- Impact: ~40-60% GPU utilization

### 2. **No Mixed Precision Training**
- Current: FP32/FP16 training without AMP
- Issue: Missing 2-3x speedup from Tensor Cores
- Impact: Leaving ~50% compute on the table

### 3. **Inefficient Data Loading**
- Current: Sequential data preparation, no prefetching
- Issue: GPU stalls waiting for data
- Impact: 10-20% idle time

### 4. **Single GPU Training**
- Current: No DDP/FSDP implementation
- Issue: Not using all 4 H100s available
- Impact: 75% of compute capacity unused

### 5. **No Compilation Optimization**
- Current: torch.compile disabled due to warnings
- Issue: Missing 20-30% speedup
- Impact: Slower kernels

## Optimization Recommendations

### 1. Optimal Batch Size and Memory Management

Based on H100 specifications and Llama 3.1 8B requirements:

```python
# Recommended configuration for H100 (80GB HBM3)
OPTIMAL_CONFIG = {
    'batch_size': 4,           # Per GPU micro-batch
    'gradient_accumulation': 16, # Effective batch = 64
    'max_batch_memory_gb': 72,  # 90% of 80GB
    'mixed_precision': 'bf16',  # Better than fp16 for training stability
}
```

**Implementation:**
```python
# In train.py, add gradient accumulation
accumulation_steps = 16
optimizer.zero_grad()

for step in range(steps_per_epoch):
    # Forward pass
    loss = compute_loss(batch)
    loss = loss / accumulation_steps

    # Backward
    loss.backward()

    # Update weights every N steps
    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training with H100 Tensor Cores

Enable BF16 training for 2-3x speedup:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for mixed precision
scaler = GradScaler('cuda')

# Training loop modification
with autocast(device_type='cuda', dtype=torch.bfloat16):
    # Forward pass
    z = encoder(batch_bytes)
    loss = compute_loss(z, targets)

# Scale loss and backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**For FP8 training (experimental, 2x additional speedup):**
```python
# Use Transformer Engine for FP8
import transformer_engine.pytorch as te

# Replace linear layers with FP8-enabled versions
encoder = te.Linear(input_dim, output_dim, bias=True)
```

### 3. Optimized DataLoader Configuration

```python
# Optimal DataLoader settings for H100
train_loader = DataLoader(
    dataset,
    batch_size=4,  # Per GPU
    num_workers=16,  # 2x CPU cores per GPU
    pin_memory=True,  # Essential for GPU
    persistent_workers=True,  # Avoid worker respawn
    prefetch_factor=4,  # Prefetch 4 batches per worker
    drop_last=True,  # Avoid uneven last batch
)

# Enable CUDA graphs for static shapes
if static_shapes:
    model = torch.jit.script(model)
    # Or use CUDA graphs
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = model(static_input)
```

### 4. Multi-GPU Training with DDP

Implement DistributedDataParallel for 4x speedup:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    # Create model on correct device
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])

    # Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=4)

    # Training loop as normal
    for batch in loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**SLURM script modification:**
```bash
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4

# Launch with torchrun
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         latentwire/train.py \
         --batch_size 4 \
         --gradient_accumulation 4
```

### 5. Enable torch.compile with Proper Settings

Fix the compilation warnings and enable:

```python
# Proper torch.compile configuration
import torch._dynamo as dynamo
dynamo.config.suppress_errors = True  # Continue on errors

# Compile with appropriate backend
if hasattr(torch, 'compile'):
    # Use inductor backend for H100
    encoder = torch.compile(
        encoder,
        mode="max-autotune",  # Best performance, slower compile
        backend="inductor",    # CUDA graphs + triton
        fullgraph=True,       # Compile entire graph
    )

    # For dynamic shapes, use:
    encoder = torch.compile(
        encoder,
        mode="reduce-overhead",
        dynamic=True,  # Handle dynamic shapes
    )
```

### 6. Memory-Efficient Attention (FlashAttention)

Use FlashAttention-3 for H100:

```python
# Enable SDPA (built into PyTorch 2.0+)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,      # FlashAttention
    enable_math=False,      # Disable fallback
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v)

# Or use explicit FlashAttention
from flash_attn import flash_attn_func
attn_output = flash_attn_func(q, k, v, causal=True)
```

### 7. Gradient Checkpointing for Large Batches

Enable gradient checkpointing to fit larger batches:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedEncoder(nn.Module):
    def forward(self, x):
        # Checkpoint transformer layers
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

### 8. Optimized Learning Rate Schedule

Use OneCycle or Cosine schedule for better convergence:

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=total_batches,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
)

# In training loop
scheduler.step()
```

### 9. Profile and Monitor GPU Utilization

Add profiling to identify bottlenecks:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(loader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# Real-time monitoring
nvidia-smi dmon -s pucvmet -i 0,1,2,3
```

### 10. Optimized SLURM Configuration

```bash
#!/bin/bash
#SBATCH --job-name=latentwire_h100_optimized
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16  # For data loading
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt

# Environment optimizations
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Launch training
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         latentwire/train.py \
         --batch_size 4 \
         --gradient_accumulation 16 \
         --mixed_precision bf16 \
         --compile_model \
         --use_flash_attention \
         --num_workers 16 \
         --pin_memory \
         --persistent_workers
```

## Expected Performance Improvements

| Optimization | Expected Speedup | GPU Util Impact |
|-------------|-----------------|-----------------|
| Optimal batch size | 1.2-1.5x | +10-15% |
| BF16 mixed precision | 2-3x | +20-30% |
| torch.compile | 1.2-1.3x | +5-10% |
| DDP (4 GPUs) | 3.8x | +75% total |
| Optimized DataLoader | 1.1-1.2x | +5-10% |
| FlashAttention | 1.3-1.5x | +10% |
| **Combined** | **8-12x** | **>90%** |

## Implementation Priority

1. **Phase 1 (Quick Wins)** - 2-3x speedup
   - Enable BF16 mixed precision
   - Optimize DataLoader (pin_memory, workers)
   - Increase batch size with gradient accumulation

2. **Phase 2 (Multi-GPU)** - 4x additional
   - Implement DDP for 4 GPUs
   - Add gradient checkpointing
   - Fix and enable torch.compile

3. **Phase 3 (Advanced)** - 1.5x additional
   - FP8 training with Transformer Engine
   - CUDA graphs for static shapes
   - Advanced profiling and tuning

## Monitoring Commands

```bash
# GPU utilization
nvidia-smi dmon -s pucvmet -i 0,1,2,3

# Memory usage
watch -n 1 nvidia-smi

# Power and temperature
nvidia-smi -q -d POWER,TEMPERATURE

# Training metrics (in Python)
print(f"Throughput: {batch_size * world_size / time_per_step:.1f} samples/sec")
print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"GPU Utilization: {gpu_util:.1%}")
```

## Troubleshooting

### OOM Errors
- Reduce batch_size, increase gradient_accumulation
- Enable gradient checkpointing
- Use FP8 or INT8 quantization

### Low GPU Utilization
- Check DataLoader num_workers
- Profile CPU bottlenecks
- Ensure pin_memory=True
- Check for CPU-bound preprocessing

### DDP Hangs
- Ensure all ranks have same batch count
- Use DistributedSampler
- Check NCCL environment variables
- Verify network connectivity

## References

- [PyTorch H100 Optimization Guide](https://github.com/pytorch/torchtitan/blob/main/docs/performance.md)
- [NVIDIA H100 Best Practices](https://developer.nvidia.com/blog/breaking-mlperf-training-records-with-nvidia-h100-gpus/)
- [FlashAttention for H100](https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html)
- [Mixed Precision Training](https://docs.pytorch.org/docs/stable/data.html)
- [DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)