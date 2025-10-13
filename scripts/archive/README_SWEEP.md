# Embedding Distribution Experiment Sweeps

Quick reference for running embedding transformation experiments.

## TL;DR - Quick Start

```bash
# Run the sweep (15-30 minutes)
git pull && rm -rf runs/embed_sweep_simple
PYTHONPATH=. bash scripts/run_embed_sweep_simple.sh

# Results appear automatically at the end
```

## What This Does

Tests whether embedding transformations can fix the **arbitrary embedding distribution problem**:
- Frozen LLMs expect embeddings from discrete token vocabulary
- Learned encoders produce arbitrary continuous embeddings
- This mismatch causes 0% F1 despite good reconstruction

**Key Insight**: Uses real text embeddings as input, applies transformations, tests generation. No training needed - immediate feedback on what helps.

## Experiments Included (35 total)

### 1. Baseline (1 experiment)
- No transformation - establishes upper bound

### 2. RMS Matching (3 experiments)
- Fixes known magnitude mismatch (115-120× too large)
- **Sweeps**: target_scale ∈ {0.8, 1.0, 1.2}
- **Likelihood**: VERY HIGH (70%+)

### 3. Batch Distribution (1 experiment)
- Normalizes mean+std across batch
- **Likelihood**: MEDIUM (30-50%)

### 4. K-Nearest Projection (10 experiments)
- Projects embeddings onto convex hull of real tokens
- **Sweeps**: k ∈ {3,5,7,10}, α ∈ {0.2,0.3,0.5,0.7,0.9}
- **Likelihood**: HIGH (50-70%)

### 5. Anchor + Offset (7 experiments)
- Each embedding = nearest token + small residual
- **Sweeps**: ε ∈ {0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3}
- **Likelihood**: VERY HIGH (70%+)

### 6. Soft Codebook (6 experiments)
- VQ-VAE style discrete bottleneck
- **Sweeps**: size ∈ {128,256,512,1024}, τ ∈ {0.7,1.0}
- **Likelihood**: MEDIUM (30-50%)

## Output

The script produces:

1. **Live progress** during run
2. **Automatic analysis** at end showing:
   - Top 5 performers ranked by F1
   - Best in each category
   - Hyperparameter insights (best k, α, ε, etc.)
   - Key insights and recommendations
3. **Saved results**:
   - `runs/embed_sweep_simple/summary.json` - Full results
   - `runs/embed_sweep_simple/<experiment>.json` - Individual results

## Interpreting Results

**What to look for:**

| Observation | Meaning | Action |
|-------------|---------|--------|
| Baseline F1 > 0.5 | LLM can use embeddings | Problem is encoder distribution ✓ |
| RMS matching wins | Magnitude mismatch | Integrate RMS calibration |
| Anchor+offset wins | Must stay near tokens | Add manifold regularization |
| K-nearest wins | Arbitrary embeddings fail | Force convex combinations |
| Everything fails | inputs_embeds broken | Debug generation setup |

## Re-running Analysis

```bash
python scripts/analyze_sweep_results.py --summary runs/embed_sweep_simple/summary.json
```

## Running Single Experiment

```bash
# Test just one configuration
PYTHONPATH=. python scripts/run_embed_sweep_simple.py \
    --experiment rms_scale1.0 \
    --samples 100

# See all available experiments
PYTHONPATH=. python scripts/run_embed_sweep_simple.py --experiment list
```

## After Finding Winner

1. **Integrate into training**:
   ```python
   # In latentwire/train.py, after adapter forward:
   reconstructed = adapter(Z)
   if args.embed_transform == 'rms_matching':
       reconstructed = calibrate_to_rms(reconstructed, target_rms)
   ```

2. **Run full training**:
   ```bash
   python latentwire/train.py \
       --embed_transform rms_matching \
       --target_rms_scale 1.0 \
       --samples 10000 \
       --epochs 10
   ```

## Files

- `run_embed_sweep_simple.sh` - Main entry point
- `run_embed_sweep_simple.py` - Python implementation
- `analyze_sweep_results.py` - Analysis and insights
- `../latentwire/embed_experiments.py` - Transformation implementations

## Multi-Epoch Question

**Q**: Do we need multiple epochs?
**A**: No, 1 epoch is sufficient for screening:

- Transformations without learnable params (RMS, K-nearest, anchor+offset) work immediately
- Transformations with params (codebook) show promise in 1 epoch if viable
- If something doesn't help in 1 epoch, more won't fix fundamental mismatch

**Strategy**: Fast screening (1 epoch) → Focused training (10+ epochs on winners)

## Expected Runtime

- **With CUDA**: ~15-20 minutes (35 experiments × ~0.5 min each)
- **CPU/MPS**: ~30-45 minutes (35 experiments × ~1 min each)

Each experiment processes 100 samples with max_new_tokens=12 generation.

## Troubleshooting

**"Model not found"**: Set MODEL_ID env var:
```bash
MODEL_ID="meta-llama/Meta-Llama-3.1-8B" bash scripts/run_embed_sweep_simple.sh
```

**Out of memory**: Reduce samples:
```bash
SAMPLES=50 bash scripts/run_embed_sweep_simple.sh
```

**Slow on CPU**: Use fewer experiments:
```bash
# Edit run_embed_sweep_simple.py:generate_experiment_configs()
# Comment out some sweep configurations
```

## See Also

- `../LOG.md` - Full experiment documentation
- `../latentwire/embed_experiments.py` - Implementation details
- `../PLAN.md` - Phase A improvements roadmap
