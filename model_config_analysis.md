# Model Configuration Analysis for LatentWire/Telepathy Experiments

## Models Currently Configured

### Llama Models
- **meta-llama/Llama-3.2-1B** (1.24B params, 2048 hidden dim)
- **meta-llama/Llama-3.2-1B-Instruct** (1.24B params, 2048 hidden dim)
- **meta-llama/Llama-3.2-3B** (3.21B params, 3072 hidden dim)
- **meta-llama/Llama-3.2-3B-Instruct** (3.21B params, 3072 hidden dim)
- **meta-llama/Meta-Llama-3.1-8B** (8.03B params, 4096 hidden dim)
- **meta-llama/Meta-Llama-3.1-8B-Instruct** (8.03B params, 4096 hidden dim)

### Mistral Models
- **mistralai/Mistral-7B-v0.3** (7.24B params, 4096 hidden dim)
- **mistralai/Mistral-7B-Instruct-v0.3** (7.24B params, 4096 hidden dim)

### Qwen Models
- **Qwen/Qwen2.5-1.5B** (1.54B params, 1536 hidden dim)
- **Qwen/Qwen2.5-1.5B-Instruct** (1.54B params, 1536 hidden dim)
- **Qwen/Qwen2.5-3B** (3.09B params, 2048 hidden dim)
- **Qwen/Qwen2.5-3B-Instruct** (3.09B params, 2048 hidden dim)
- **Qwen/Qwen2.5-7B** (7.62B params, 3584 hidden dim)
- **Qwen/Qwen2.5-7B-Instruct** (7.62B params, 3584 hidden dim)

## Configuration Issues Found

### 1. **Precision Settings**
✅ **Correctly Configured:**
- All models consistently use `torch_dtype=torch.bfloat16`
- This is optimal for H100 GPUs which have native BF16 support
- Provides better numerical stability than FP16 for training

### 2. **Device Map Settings**
⚠️ **Inconsistent Configuration:**
- Some files use `device_map="auto"` (good for multi-GPU)
- Others use `device_map={"": local_rank}` (for DDP)
- Mac experiments use `device_map="auto"` conditionally

**Recommendation:** Use consistent device mapping based on platform:
```python
# For HPC with DDP
device_map = {"": local_rank} if using_ddp else "auto"

# For Mac
device_map = "auto"
```

### 3. **Memory Configurations (from memory_configs.py)**

Memory-safe batch sizes for 80GB H100 GPUs:

| Model Combination | Batch Size | Grad Accum | Effective Batch |
|-------------------|------------|------------|-----------------|
| Llama-8B + Mistral-7B | 2 | 4 | 8 |
| Llama-3B + Mistral-7B | 3 | 3 | 9 |
| Llama-3B + Qwen-3B | 6 | 2 | 12 |
| Llama-1B + Qwen-1.5B | 12 | 1 | 12 |
| Single Llama-8B | 6 | 2 | 12 |
| Single Mistral-7B | 6 | 2 | 12 |
| Single Llama-3B | 12 | 1 | 12 |

### 4. **Model ID Verification**

✅ **All model IDs are correct for HuggingFace:**
- Llama models use correct "meta-llama/" prefix
- Mistral models use correct "mistralai/" prefix
- Qwen models use correct "Qwen/" prefix
- Version numbers are accurate (3.1, 3.2 for Llama; v0.3 for Mistral; 2.5 for Qwen)

### 5. **Missing Model Configurations**

The following models appear in memory_configs.py but are NOT used in experiments:
- **Qwen/Qwen2.5-7B** and **Qwen/Qwen2.5-7B-Instruct** (7.62B params)
  - These are configured but never loaded in actual experiments
  - Memory estimates exist but no training scripts use them

### 6. **Tokenizer Configuration**

✅ **Consistent Pattern:**
```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Standard for causal LMs
```

### 7. **Generation Configuration**

⚠️ **Not Explicitly Set:**
- Generation configs rely on model defaults
- No explicit `generation_config` parameters set
- This could lead to inconsistent behavior across models

**Recommendation:** Set explicit generation parameters:
```python
generation_config = {
    "max_new_tokens": 256,
    "do_sample": False,  # For reproducibility
    "temperature": 1.0,
    "top_p": 1.0,
    "pad_token_id": tokenizer.eos_token_id
}
```

## Memory Usage Analysis

Based on configurations, estimated memory usage per model (BF16):

| Model | Parameter Memory | Activation Memory (batch=1, seq=1536) | Total |
|-------|------------------|---------------------------------------|--------|
| Llama-3.2-1B | 2.48 GB | ~2.5 GB | ~5 GB |
| Llama-3.2-3B | 6.42 GB | ~5.2 GB | ~12 GB |
| Llama-3.1-8B | 16.06 GB | ~12.3 GB | ~28 GB |
| Mistral-7B | 14.48 GB | ~12.3 GB | ~27 GB |
| Qwen2.5-1.5B | 3.08 GB | ~2.3 GB | ~5.4 GB |
| Qwen2.5-3B | 6.18 GB | ~3.5 GB | ~10 GB |

## Recommendations

1. **Standardize device_map usage:**
   - Create a utility function that returns appropriate device_map based on platform and DDP usage
   - Avoid hardcoding device_map in individual scripts

2. **Add Qwen2.5-7B experiments:**
   - Memory configs exist but no experiments use this model
   - Could be valuable for cross-vocabulary comparisons

3. **Explicit generation configs:**
   - Create a standard generation config dictionary
   - Apply consistently across all evaluation scripts

4. **Memory monitoring:**
   - Add torch.cuda.max_memory_reserved() logging after model loading
   - Compare with memory_configs.py estimates to validate accuracy

5. **Flash Attention 2:**
   - Currently only used in unified_cross_model_experiments.py
   - Consider enabling for all HPC experiments to reduce memory usage

6. **Model caching:**
   - Consider using HuggingFace cache directory on HPC to avoid re-downloading
   - Set: `export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface`

## Validation Checklist

- [x] Model IDs are correct for HuggingFace
- [x] BFloat16 precision used consistently
- [x] Memory estimates align with H100 capacity (80GB)
- [x] Tokenizer configurations are appropriate
- [ ] Device map usage is inconsistent (needs standardization)
- [ ] Generation configs not explicitly set
- [ ] Qwen2.5-7B configured but unused
- [ ] Flash Attention 2 not consistently enabled

## Next Steps

1. Create `telepathy/model_utils.py` with standardized loading functions
2. Update all training scripts to use consistent model loading
3. Add explicit generation configurations
4. Enable Flash Attention 2 for all HPC experiments
5. Add memory usage tracking to training logs