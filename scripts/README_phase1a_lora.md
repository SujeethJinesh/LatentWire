# Phase 1a + LoRA Experiment

## Hypothesis

**Question**: Can LoRA adaptation improve Phase 1a's pure reconstruction approach (24% F1) without triggering the mode collapse seen in Phase 1b (0% F1)?

## Background

From REPORT.md and LOG.md:

| Experiment | Supervision | LoRA | Result | Issue |
|------------|-------------|------|--------|-------|
| Phase 1a | Pure reconstruction (minimal CE) | ❌ No | **24% F1** | Answer present, wrong format |
| Phase 1b | Reconstruction + K-token CE (λ=0.5) | ❌ No | **0% F1** | Mode collapse to "the the the" |
| Section 2.8 | K-token CE | ✅ Yes (r=8-16) | **4% F1** | Still collapsed despite LoRA |
| **This experiment** | **Minimal CE (λ=0.01)** | ✅ **Yes** | **?% F1** | **Untested gap!** |

## Rationale

1. **Phase 1a works** (24% F1) - semantics preserved, just format issue
2. **LoRA helps adaptation** (LOG.md line 2380) - weights growing, losses decreasing
3. **K-token CE causes collapse** (LOG.md line 3625) - conflicts with reconstruction
4. **Gap identified**: Never tested reconstruction + LoRA (without strong K-token CE)

## Experimental Design

### Training Setup
- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Dataset**: SQuAD
- **Samples**: 5,000 (reduced for faster iteration)
- **Epochs**: 3
- **Batch size**: 32

### Minimal Supervision (Phase 1a style)
- **First-token CE weight**: 0.01 (vs 0.5 in Phase 1b, 0.0 in Phase 1a)
- **KD weight**: 0.0 (disabled)
- **K tokens**: 4 (minimal supervision window)

### LoRA Sweep (11 configurations)

1. **Baseline**: No LoRA (replicate Phase 1a)

2. **Small** (minimal capacity):
   - r=4, α=8, layers=8
   - r=4, α=16, layers=8

3. **Medium** (balanced):
   - r=8, α=8, layers=12
   - r=8, α=16, layers=12
   - r=8, α=32, layers=12

4. **Large** (more capacity):
   - r=16, α=16, layers=16
   - r=16, α=32, layers=16

5. **Full model**:
   - r=8, α=16, all 32 layers
   - r=16, α=32, all 32 layers

6. **High rank**:
   - r=32, α=32, layers=16

## Success Criteria

- **Baseline (no LoRA)**: Should match Phase 1a (~24% F1)
- **Marginal improvement**: LoRA gives +2-5% F1 (26-29%)
- **Significant improvement**: LoRA gives +5-10% F1 (29-34%)
- **Strong improvement**: LoRA gives +10%+ F1 (34%+)

## Running the Experiment

### On HPC cluster:
```bash
git pull && rm -rf runs && PYTHONPATH=. bash scripts/sweep_phase1a_lora.sh
```

### Customize parameters:
```bash
# Quick smoke test (300 samples, 1 epoch)
SAMPLES=300 EPOCHS=1 bash scripts/sweep_phase1a_lora.sh

# Full experiment (10k samples, 5 epochs)
SAMPLES=10000 EPOCHS=5 bash scripts/sweep_phase1a_lora.sh
```

## Analysis

### View summary:
```bash
cat runs/phase1a_lora_sweep/sweep_summary.txt
```

### Compare configurations:
```bash
python runs/phase1a_lora_sweep/compare_results.py
```

### Best configuration:
```bash
# Find config with highest F1
python -c "
import json, glob
from pathlib import Path
results = []
for f in glob.glob('runs/phase1a_lora_sweep/*/diagnostics.jsonl'):
    config = Path(f).parent.name
    best_f1 = 0
    with open(f) as fp:
        for line in fp:
            try:
                data = json.loads(line)
                best_f1 = max(best_f1, data.get('text_f1', 0))
            except: pass
    results.append((config, best_f1))
results.sort(key=lambda x: x[1], reverse=True)
print(f'Best: {results[0][0]} with F1={results[0][1]:.1%}')
"
```

## Expected Timeline

- **Per configuration**: ~15-20 minutes (5k samples, 3 epochs)
- **Full sweep** (11 configs): ~3-4 hours
- **Quick smoke test** (300 samples, 1 epoch): ~30 minutes total

## Next Steps Based on Results

### If LoRA improves F1 by +5% or more:
1. Add LoRA configuration to Phase 0 post-processing pipeline
2. Use best LoRA config as baseline for future experiments
3. Update REPORT.md with this finding

### If LoRA provides marginal improvement (+2-5%):
1. Document as minor optimization
2. Consider for production deployment (small parameter cost)
3. Continue with VQ-VAE (Phase 1) as planned

### If LoRA doesn't help (<+2%):
1. Confirms Phase 1a limitation is not about LLM adaptation
2. Focus on post-processing (Phase 0) and VQ-VAE (Phase 1)
3. LoRA is necessary but not sufficient (as REPORT.md suggests)

## Integration with REPORT.md

If successful, add to Section 2.8 or create new Section 2.9:

```markdown
### 2.9 Hypothesis 9: LoRA with Minimal Supervision

**Motivation**: Phase 1a (pure reconstruction) achieved 24% F1. Can LoRA adaptation
improve this without K-token CE collapse?

**Results**:
- Baseline (no LoRA): 24% F1 (matches Phase 1a)
- Best LoRA (r=X, α=Y, layers=Z): [RESULT]% F1 (Δ=[DELTA]%)

**Lesson Learned**: [Fill based on results]
```

## Files Generated

- `runs/phase1a_lora_sweep/[config]/training_*.log` - Full training logs
- `runs/phase1a_lora_sweep/[config]/diagnostics.jsonl` - Per-step metrics
- `runs/phase1a_lora_sweep/sweep_summary.txt` - Results summary
- `runs/phase1a_lora_sweep/compare_results.py` - Analysis script
