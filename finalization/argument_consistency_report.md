# Argument Consistency Report for LatentWire

## Critical Finding

There is an inconsistency between the argument names used in scripts and the actual implementation:

### The Problem

1. **train.py** uses: `--warm_anchor_text`
2. **eval.py** uses: `--latent_anchor_text` and `--latent_anchor_mode`
3. **Scripts** are using: `--latent_anchor_text` (for train.py) which is WRONG

## Files Affected

### Scripts calling train.py with WRONG argument name (--latent_anchor_text instead of --warm_anchor_text):

1. `/Users/sujeethjinesh/Desktop/LatentWire/finalization/RUN.sh`
   - Line 251: Uses `--latent_anchor_text "Answer: "` (WRONG - should be `--warm_anchor_text`)

### Actual Arguments in Implementation

#### latentwire/train.py (Line 1528)
```python
ap.add_argument("--warm_anchor_text", type=str, default="",
                help="Anchor text to insert between prefix and answer during training")
```

#### latentwire/eval.py (Lines 2220-2221)
```python
ap.add_argument("--latent_anchor_mode", type=str, default="auto", choices=["auto","chat","text","none"])
ap.add_argument("--latent_anchor_text", type=str, default="Answer: ")
```

## All Scripts Using train.py (for reference)

The following scripts correctly use train.py but should be checked for argument consistency:

### Main Scripts
- `scripts/run_embedding_diagnostics.sh` - Line 93
- `scripts/quick_test.sh` - Line 57
- `scripts/run_optimized_training.sh` - Line 39
- `scripts/run_integration_test.sh` - Line 257
- `scripts/run_mixed_precision.sh` - Line 42
- `scripts/run_8epoch_training.sh` - Line 119
- `scripts/run_elastic_gpu_demo.sh` - Line 62
- `scripts/run_edge_case_tests.sh` - Line 77

### Archive Scripts
- `scripts/archive/run_optimized_h100.sh` - Line 48
- `scripts/archive/run_hero_h100.sh` - Line 47
- `scripts/archive/run_minimal_h100.sh` - Line 33
- `scripts/archive/run_embedding_smoke.sh` - Line 28

### Telepathy Scripts
- `telepathy/submit_production_readiness.slurm` - Line 160
- `telepathy/test_preemptible_hpc.slurm` - Lines 55, 137
- `telepathy/submit_dataloader_benchmark.slurm` - Lines 76, 102
- `telepathy/submit_mixed_precision.slurm` - Line 70
- `telepathy/submit_gpu_monitoring.slurm` - Lines 96, 113, 130
- `telepathy/test_preemptible_quick.sh` - Lines 75, 170
- `telepathy/submit_elastic_gpu_experiment.slurm` - Lines 72, 92, 113, 134
- `telepathy/submit_ddp_training.slurm` - Line 92

## Recommended Fix

### For finalization/RUN.sh

Change line 251 from:
```bash
--latent_anchor_text "Answer: " \
```

To:
```bash
--warm_anchor_text "Answer: " \
```

### Verification Steps

After fixing, verify that:
1. All train.py calls use `--warm_anchor_text`
2. All eval.py calls use `--latent_anchor_text` and `--latent_anchor_mode`
3. No scripts are using undefined arguments

## Summary

The main issue is in `/Users/sujeethjinesh/Desktop/LatentWire/finalization/RUN.sh` which uses the wrong argument name for train.py. This would cause the training script to fail or ignore the anchor text setting, potentially affecting model performance significantly.