# KNOWN_ISSUES.md

Last Updated: 2026-01-04

This document tracks known issues, bugs, workarounds, and limitations discovered in the LatentWire codebase. Each issue includes description, impact, workaround/fix, and priority level.

## Critical Issues

### 1. CUDA Out of Memory (OOM) Errors
**Description**: Training frequently runs out of GPU memory, especially with knowledge distillation (KD) enabled or when using multiple models simultaneously.

**Impact**: Training crashes, requires restart with smaller batch sizes

**Workarounds/Fixes**:
- Reduce batch size (documented safe values: batch_size=2-4 for dual models)
- Use gradient accumulation to maintain effective batch size
- Set `KD_TEACHER_CHUNK=1` for per-example KD processing
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- Process evaluation in batches to avoid OOM
- Clear GPU cache between experiments with `torch.cuda.empty_cache()`

**Priority**: Critical

**Affected Files**:
- `latentwire/train.py` - KD teacher forward pass concatenation
- `paper_writing/cross_attention.py` - CUDA Graph OOM with dynamic shapes
- `experimental/learning/unified_cross_model_experiments.py` - Multi-model loading

---

### 2. PAD Token Gradient Contamination
**Description**: PAD tokens were not properly masked in loss computation, causing gradient contamination in left-padded sequences.

**Impact**: Incorrect gradient updates, poor model convergence

**Fix Applied**: PAD tokens now masked with -100 in labels, attention masks properly zeroed

**Priority**: Critical (FIXED)

**Affected Files**:
- `latentwire/losses.py` - Added proper PAD masking in k_token_ce_from_prefix

---

### 3. BOS Token Misalignment
**Description**: Inconsistent BOS token handling between training and evaluation causing token misalignment.

**Impact**: First-token prediction accuracy severely degraded

**Fix Applied**: Consistent BOS policy with `append_bos_after_prefix=yes`

**Priority**: Critical (FIXED)

**Affected Files**:
- `latentwire/prefix_utils.py`
- `latentwire/train.py`
- `latentwire/eval.py`

---

## High Priority Issues

### 4. Memory Fragmentation Over Time
**Description**: GPU memory becomes fragmented during long training runs, especially with KD enabled.

**Impact**: OOM errors after several epochs even with conservative batch sizes

**Workarounds**:
- Enable expandable segments: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Periodic checkpoint and restart
- Process KD in smaller chunks

**Priority**: High

---

### 5. Device Mismatch Errors
**Description**: Tensors occasionally end up on wrong devices in multi-GPU setups.

**Impact**: Runtime errors, training crashes

**Fix Applied**: Explicit `.to(device)` calls added, optimizer state alignment fixed

**Priority**: High (MOSTLY FIXED)

**Affected Files**:
- `latentwire/train.py` - `_align_optimizer_state_to_param_devices` function added

---

### 6. torch.compile() Issues
**Description**: torch.compile causes symbolic_shapes warnings and creates excessive CUDA graphs (51+ graphs, 23-25 GiB each).

**Impact**: Memory exhaustion, slow first steps

**Workaround**: Disabled by default with `--no_compile` flag

**Priority**: High

**Affected Files**:
- `paper_writing/cross_attention.py`
- `latentwire/train.py`

---

## Medium Priority Issues

### 7. Mode Collapse in Generation
**Description**: Models often collapse to generating single tokens repeatedly (e.g., "the" token).

**Impact**: Poor generation quality, low diversity

**Workarounds**:
- Adjust first-token CE weight carefully (sweet spot: 8-11)
- Add entropy regularization
- Use temperature sampling
- Monitor diversity metrics during training

**Priority**: Medium

---

### 8. Slow Dataset Loading
**Description**: Dataset loading can be slow, especially with trust_remote_code warnings.

**Impact**: Slower iteration during development

**Workaround**: Cache datasets locally, disable trust_remote_code where safe

**Priority**: Medium

---

### 9. Checkpoint Size
**Description**: Checkpoints can be large (>500MB with optimizer state).

**Impact**: Disk space usage, slow save/load

**Workarounds**:
- Save without optimizer state for evaluation checkpoints
- Implement checkpoint pruning to keep only best/recent

**Priority**: Medium

---

## Low Priority Issues

### 10. Incomplete Test Coverage
**Description**: Limited test coverage for checkpoint recovery and edge cases.

**Impact**: Bugs may not be caught before deployment

**Fix Needed**: Add comprehensive test suite

**Priority**: Low

---

### 11. Logging Verbosity
**Description**: Excessive debug output and warnings from transformers library.

**Impact**: Log files become very large, hard to find important information

**Workarounds**:
- Suppress specific warnings with warning filters
- Use log levels appropriately
- Rotate log files

**Priority**: Low

---

## Performance Limitations

### 12. Batch Size Constraints
**Description**: Limited to small batch sizes (2-4) when using multiple models.

**Impact**: Slower training, may affect convergence

**Safe Configurations**:
- Single model: batch_size=16-32
- Dual models (Llama + Mistral): batch_size=2-4
- With KD enabled: reduce by 50%

**Priority**: Medium

---

### 13. GPU Utilization
**Description**: GPU utilization often below 80% due to memory constraints.

**Impact**: Suboptimal training speed

**Potential Fixes**:
- Better memory management
- Gradient checkpointing
- Model parallelism improvements

**Priority**: Low

---

## Compatibility Issues

### 14. MPS Backend Issues
**Description**: MPS (Apple Silicon) backend has issues with model.generate() and other operations.

**Impact**: Cannot fully utilize Apple Silicon GPUs

**Workaround**: Use CPU fallback or CUDA-only features

**Priority**: Low (platform-specific)

---

### 15. SLURM Configuration
**Description**: Incorrect SLURM account/partition settings cause job failures.

**Impact**: Jobs fail to submit or run

**Required Settings**:
- Account: `marlowe-m000066` (NOT just `marlowe`)
- Partition: `preempt`
- Working dir: `/projects/m000066/sujinesh/LatentWire`

**Priority**: Critical (for HPC users)

---

## Known Workarounds in Code

### Temporary Fixes
1. **KD Teacher Chunking**: Process in chunks to avoid OOM (`KD_TEACHER_CHUNK=1`)
2. **Gradient Clipping**: Aggressive clipping (1.0) to prevent explosions
3. **Learning Rate Warmup**: Extended warmup to prevent early collapse
4. **Sequential Model Loading**: Load models one at a time to avoid OOM
5. **Manual Garbage Collection**: Explicit `torch.cuda.empty_cache()` calls

---

## Monitoring Recommendations

### Key Metrics to Watch
1. **Memory Usage**: Peak memory should stay below 75GB on 80GB GPUs
2. **Gradient Norms**: Should stay below 10.0, ideally 0.5-2.0
3. **First-Token Accuracy**: Target >10% for successful training
4. **Loss Convergence**: All losses should decrease, watch for plateaus
5. **Diversity Metrics**: Entropy should stay above 1.0

### Debug Flags
- `--debug`: Enable verbose logging
- `--debug_print_first N`: Print first N generations
- `--debug_topk N`: Debug top-k predictions

---

## Future Improvements Needed

1. **Memory Management**: Implement better memory pooling and management
2. **Checkpointing**: Add gradient checkpointing for larger batch sizes
3. **Mixed Precision**: Optimize mixed precision training
4. **Data Pipeline**: Improve data loading and preprocessing efficiency
5. **Testing**: Comprehensive test suite for all components
6. **Documentation**: Better inline documentation of workarounds

---

## How to Report New Issues

When discovering new issues, please document:
1. **Description**: Clear explanation of the problem
2. **Reproduction**: Steps to reproduce
3. **Error Messages**: Full stack traces
4. **Configuration**: Hardware, software versions, hyperparameters
5. **Workaround**: Any temporary fixes found
6. **Impact**: How it affects training/evaluation

Update this file when issues are discovered or resolved.