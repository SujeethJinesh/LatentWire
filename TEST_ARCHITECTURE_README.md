# Testing the New Architecture

Quick guide to validating the Anchor-Guided Cross-Model Interlingua.

## What This Tests

The test script validates that the proposed architecture can:
1. ✓ Instantiate all components correctly
2. ✓ Run forward pass without errors
3. ✓ Compute losses (generation, alignment, semantic)
4. ✓ Train for N steps
5. ✓ Generate diverse predictions (not collapsed like current ByteEncoder)

## Quick Start

### Minimal Test (5 minutes)
```bash
# Just check that everything works
python test_new_architecture.py --samples 10 --steps 5
```

### Medium Test (15-30 minutes)
```bash
# Actually see some learning
export PYTHONUNBUFFERED=1
python test_new_architecture.py --samples 100 --steps 50
```

### Full Test with Qwen (1-2 hours)
```bash
# Test cross-model alignment
export PYTHONUNBUFFERED=1
python test_new_architecture.py --samples 1000 --steps 500 --test_qwen
```

## Expected Output

### Phase 1: Component Instantiation
```
[1/7] Loading frozen models...
  - SentenceTransformer...
  - Llama-3.1-8B...
  ✓ All models loaded

[2/7] Instantiating new architecture components...
  ✓ AlignmentTransformer: 10,485,760 params
  ✓ InterlinguaAdapter (Llama): 4,194,304 params
  Total trainable: 14,680,064 params (14.7M)
```

### Phase 2: Forward Pass Test
```
[4/7] Testing forward pass (single example)...
  ✓ z_sem shape: torch.Size([1, 384])
  ✓ llama_embeds shape: torch.Size([1, 251, 4096])
  ✓ z_llama shape: torch.Size([1, 512])
  ✓ prefix_embeds_llama shape: torch.Size([1, 32, 4096])

  Testing generation (before training)...
  Gold answer: linear
  Pred (untrained): the most common form of
```

**Key observation:** Untrained prediction should be RANDOM (not the same for all examples).

### Phase 3: Loss Computation
```
[5/7] Testing loss computation...
  ✓ Generation loss (K=4): 8.2341
  ✓ Alignment loss: 0.0234
  ✓ Semantic anchor loss: 0.1523
  ✓ Total loss: 8.4098
```

### Phase 4: Training
```
[6/7] Running 50 training steps on 100 examples...
  Step 1/50: loss=8.4210 gen=8.2341 align=0.0234 sem=0.1523
  Step 5/50: loss=7.9823 gen=7.8012 align=0.0211 sem=0.1398
  Step 10/50: loss=7.5634 gen=7.3901 align=0.0198 sem=0.1289
  ...
  Step 50/50: loss=6.2341 gen=6.0789 align=0.0156 sem=0.0982
  ✓ Training complete
```

**What to look for:**
- Loss should **decrease** over time
- If loss stays flat or increases, something is wrong

### Phase 5: Post-Training Generation
```
[7/7] Testing generation after training...

  Results:
  ============================================================================
  [1] Gold: linear
      Pred: linear and

  [2] Gold: Lampea
      Pred: Lampea

  [3] Gold: residents willing to pay higher market rate for housing
      Pred: gentrification

  [4] Gold: San Jose
      Pred: San Jose

  [5] Gold: oxides
      Pred: oxides

  ============================================================================

  Diversity check: 5/5 unique predictions
  ✓ Good diversity!
```

**Success criteria:**
- Predictions should be DIVERSE (not all the same)
- Some predictions should match or be close to gold answers
- Much better than random untrained output

## Comparison: Old vs New

### Current Architecture (FAILS)
```
ALL predictions: "2019) 1. The answer is"
Diversity: 1/5 unique (COLLAPSED)
F1: 0.0%
```

### New Architecture (Expected)
```
After 50 steps training:
  Predictions: varied, some correct
  Diversity: 4-5/5 unique (GOOD)
  F1: >10% (with more training: >50%)
```

## Command Line Options

```
--samples N          Number of training examples (default: 100)
--steps N            Number of training steps (default: 50, 0 to skip)
--batch_size N       Batch size (default: 1, increase for speed)
--lr FLOAT           Learning rate (default: 1e-4)
--d_inter N          Interlingua dimension (default: 512)
--num_slots N        Number of soft tokens (default: 32)
--test_qwen          Also test with Qwen (slower, tests alignment)
```

## Troubleshooting

### "sentence-transformers not found"
The script auto-installs it, but you can manually run:
```bash
pip install sentence-transformers
```

### "CUDA out of memory"
Reduce batch size or samples:
```bash
python test_new_architecture.py --samples 10 --steps 5 --batch_size 1
```

### "All predictions are identical"
This means the architecture is collapsing like ByteEncoder. Check:
1. Is loss decreasing? If not, learning isn't happening
2. Try longer training: `--steps 200`
3. Try higher learning rate: `--lr 5e-4`

### Predictions are random/wrong after training
This is EXPECTED for very short training (5-50 steps). The architecture is working if:
- Loss is decreasing ✓
- Predictions are DIVERSE (not collapsed) ✓
- Some words/phrases are relevant ✓

For actual quality, need longer training (500+ steps on 1000+ samples).

## Next Steps After Test Passes

1. **Run longer experiment:**
   ```bash
   python test_new_architecture.py --samples 10000 --steps 5000 --batch_size 4
   ```

2. **Integrate into main training:**
   - Replace ByteEncoder with AlignmentTransformer in train.py
   - Replace Adapter with InterlinguaAdapter
   - Update loss computation to include alignment + semantic terms

3. **Full pipeline:**
   - Train on full SQuAD (87k samples, 10 epochs)
   - Evaluate with eval.py
   - Target: F1 > 50% (vs current 0%, text baseline 69%)

## What Success Looks Like

After running `--samples 100 --steps 50`:

✅ **Architecture works if:**
- All 7 test phases complete without errors
- Loss decreases from ~8.0 to ~6.0
- Post-training predictions are DIVERSE (4-5 unique out of 5)
- Some predictions are partially correct (e.g., "San Jose" for "San Jose")

❌ **Architecture fails if:**
- Errors during instantiation or forward pass
- Loss doesn't decrease or increases
- All predictions are identical (collapse)
- Predictions are completely unrelated to questions

## Time Estimates

| Test | Samples | Steps | Qwen? | Time | GPU Memory |
|------|---------|-------|-------|------|------------|
| Smoke | 10 | 5 | No | ~5 min | ~20 GB |
| Quick | 100 | 50 | No | ~15 min | ~25 GB |
| Medium | 1000 | 500 | No | ~1 hour | ~30 GB |
| Full | 1000 | 500 | Yes | ~2 hours | ~50 GB |

## Architecture Summary (for reference)

```
Text → Frozen SentenceTransformer (semantic anchor)
    ↓
Text → Llama/Qwen tokenizer + embeddings (frozen)
    ↓
AlignmentTransformer (learned)
  - Per-model projections (4096/2048 → 512)
  - Cross-attention to semantic anchor
  - Transformer refinement
  - Mean pooling → single vector [512]
    ↓
z_llama [512], z_qwen [512]  ← SHARED INTERLINGUA
    ↓
InterlinguaAdapter (learned, per-model)
  - Expand: [512] → [32, 512]
  - Project: [32, 512] → [32, 4096 or 2048]
    ↓
Frozen LLM → Generate answer
```

**Key innovation:** Start in LLM-native space (token embeddings), not bytes!
