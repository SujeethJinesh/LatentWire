# Action Plan: Achieving F1 > 0 Before Hero Run

## Critical Finding: Stage B IS Learning (ChatGPT was wrong)

**ChatGPT's error**: Claimed "Stage B shows no steps >0" and "only Stage A learned"

**Truth from diagnostics**:
- Stage A (steps 10-160): `first_acc=0.000` throughout ❌
- Stage B (steps 10-160): THREE breakthroughs ✅
  - Step 40: `first_acc=4.17%`
  - Step 110: `first_acc=8.33%` (PEAK)
  - Step 160: `first_acc=4.17%` (regressed)

**Conclusion**: LoRA in Stage B IS working. The issue is:
1. **Insufficient steps** (160 → need 300-500 for convergence)
2. **Regression instability** (8.33% → 4.17% suggests learning rate too high)
3. **Exposure bias** (8.33% first-token ≠ coherent generation → need autoregressive training)

---

## Staged Approach to Hero Readiness

We'll incrementally extend smoke until F1 > 0, then scale to hero with confidence.

### Phase 1: Extended Stage B Smoke (NOW) - 30 min, ~$5

**Goal**: Test if 2× more Stage B steps (320 vs 160) achieves F1 > 0

**Command**:
```bash
# Resume from existing Stage A checkpoint (proven to work)
RESUME_FROM=runs/smoke/ckpt/stageA bash scripts/smoke_stageb_extended.sh
```

**What changes**:
- Stage B epochs: 4 → 8 (160 → 320 steps)
- Same hyperparameters as original smoke
- Focuses compute on the part that's learning (Stage B with LoRA)

**Success criteria**:
- Peak `first_acc ≥ 12%` (vs smoke's 8.33%)
- Final `first_acc` within 20% of peak (no regression)
- **F1 > 0** in eval (even F1=0.01 is a breakthrough)

**Analysis during run**:
```bash
# Monitor training progress
tail -f runs/smoke_stageb_ext/pipeline_*.log

# Check for breakthroughs
python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl

# Monitor eval for F1 > 0
bash scripts/monitor_f1_breakthrough.sh runs/smoke_stageb_ext/pipeline_*.log
```

**Decision tree**:
- If **F1 > 0**: ✅ Hero-ready! Architecture works, just needs more steps
- If **F1 = 0** but `first_acc ≥ 12%`: ⚠️ Exposure bias confirmed, add scheduled sampling
- If **F1 = 0** and `first_acc < 12%`: ⚠️ Need hyperparameter tuning (see Phase 2)

### Phase 2: Hyperparameter Sweep (IF Phase 1 fails) - 2 hours, ~$20

If extended smoke still shows F1=0, run targeted experiments:

**Experiment A: Increase First-Token Weight**
```bash
# Test if stronger acceptance pressure helps
FIRST_TOKEN_CE_WEIGHT_STAGEB=12.0 \
RESUME_FROM=runs/smoke/ckpt/stageA \
bash scripts/smoke_stageb_extended.sh
```

**Experiment B: More Epochs**
```bash
# Test if 16 epochs (640 steps) is enough
# Edit smoke_stageb_extended.sh: EPOCHS_STAGEB=16
RESUME_FROM=runs/smoke/ckpt/stageA \
bash scripts/smoke_stageb_extended.sh
```

**Experiment C: Learning Rate Scheduling**
```bash
# Add cosine decay to train.py (code change needed)
# lr: 5e-5 → 1e-6 over training
# Prevents regression (8.33% → 4.17%)
```

**Success criteria**: ANY experiment achieving F1 > 0.01

### Phase 3: Architectural Fix (IF Phase 2 fails) - 1 day, code changes

If hyperparameter sweep fails, implement scheduled sampling:

**Scheduled Sampling**: Mix gold tokens with model predictions during training

```python
# In train.py, replace teacher forcing with:
def scheduled_sampling_step(model, prefix, gold_ids, epoch, max_epochs):
    """Anneal from full teacher-forcing to 50% model predictions."""
    p_gold = max(0.5, 1.0 - epoch / max_epochs)  # 1.0 → 0.5

    generated = []
    for t in range(len(gold_ids)):
        # Predict next token
        logits = model(prefix + generated)
        pred_token = logits.argmax()

        # Use gold or prediction based on schedule
        if random.random() < p_gold:
            next_token = gold_ids[t]  # Teacher forcing
        else:
            next_token = pred_token   # Model prediction

        generated.append(next_token)

    # Compute loss on full sequence
    return loss(generated, gold_ids)
```

**Expected impact**:
- Model learns to recover from its own mistakes
- Bridges teacher-forcing → autoregressive gap
- Should achieve F1 > 0.05

### Phase 4: Hero Run (ONLY after F1 > 0) - 4-6 hours, ~$100

Once smoke achieves F1 > 0, scale to hero with confidence:

```bash
bash scripts/run_llama_single.sh --hero
```

**Expected hero results** (based on smoke signals):
- If smoke F1 = 0.01-0.05 → hero F1 = 0.10-0.20 ✅
- If smoke F1 = 0.05-0.10 → hero F1 = 0.20-0.40 ✅
- If smoke F1 > 0.10 → hero F1 = 0.40+ ✅✅

---

## Detailed Metrics & Logging Plan

### Training Diagnostics (diagnostics.jsonl)

**Already logged**:
- `first_acc`: First-token accuracy per step ✅
- `first`: First-token CE loss ✅
- `kce`: K-token CE loss ✅
- `kd`: Knowledge distillation loss ✅
- `grad_norm`: Gradient norm ✅

**Add these** (modify train.py):
```python
# Per-step metrics
- first_acc_top5: Top-5 first-token accuracy
- answer_length_avg: Average generated length (for "thethethe" detection)
- token_diversity: Unique tokens in first 10 predictions

# Per-epoch metrics (save to diagnostics_epoch.jsonl)
- sample_predictions: 5 random (question, pred, gold) tuples
- first_token_histogram: Distribution of predicted first tokens
- loss_breakdown: Individual contribution of each loss term
```

### Eval Metrics (eval output)

**Already logged**:
- EM, F1 (aggregate) ✅
- First-token top1/top5 ✅
- NLL/token ✅

**Add these** (modify eval.py):
```python
# Per-example metrics (save to eval_detailed.jsonl)
for example in eval_set:
    {
        "question": example.question,
        "gold": example.answer,
        "prediction": model.generate(example.question),
        "em": exact_match(pred, gold),
        "f1": f1_score(pred, gold),
        "first_token_correct": pred[0] == gold[0],
        "prefix_used": "latent" or "text",
    }

# Breakthrough detection
- first_f1_nonzero_idx: Index of first example with F1 > 0
- f1_distribution: Histogram of F1 scores (how many 0, 0-0.1, 0.1-0.5, etc)
- error_modes: Count of "thethethe", empty, wrong_entity, etc
```

### Breakthrough Monitoring

**Real-time analysis** during training:
```bash
# Watch for first_acc improvements
watch -n 30 'tail -20 runs/smoke_stageb_ext/diagnostics.jsonl | \
  python -c "import sys,json; \
  [print(f\"Step {json.loads(l)[\"global_step\"]}: first_acc={json.loads(l)[\"models\"][\"llama\"][\"first_acc\"]:.3%}\") \
  for l in sys.stdin]"'

# Alert on F1 > 0
tail -f runs/smoke_stageb_ext/pipeline_*.log | \
  grep --line-buffered "F1:" | \
  awk '$NF > 0 {print "🎉 BREAKTHROUGH: F1 =", $NF; system("say breakthrough detected")}'
```

**Post-run analysis**:
```bash
# Comprehensive diagnostics report
python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl

# Find first F1 > 0 example
grep "\"f1\":" runs/smoke_stageb_ext/eval_detailed.jsonl | \
  awk -F'f1": ' '$2 > 0 {print NR, $0; exit}'
```

---

## Acceptance Criteria: Hero Readiness Checklist

Before running hero, ALL must pass:

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| **First-token learning** | Peak ≥ 12% | `python scripts/analyze_breakthrough.py diagnostics.jsonl` |
| **First-token stability** | Final within 20% of peak | Same script, check regression analysis |
| **F1 breakthrough** | F1 > 0 on ANY example | `bash scripts/monitor_f1_breakthrough.sh pipeline.log` |
| **Generation quality** | Avg F1 ≥ 0.01 | `grep "Llama  EM:" pipeline.log` (latent F1 line) |
| **Gradient health** | Max grad_norm < 500 | `grep grad_norm diagnostics.jsonl | sort -t= -k2 -rn | head -1` |
| **No mode collapse** | <10% "thethethe" outputs | `grep prediction eval_detailed.jsonl | grep -c "thethe"` |

**Hero launch command** (only run when all ✅):
```bash
# Confirm readiness
echo "Smoke achieved F1=$(grep 'Llama.*F1:' runs/smoke_stageb_ext/pipeline_*.log | tail -1)"
echo "Peak first_acc=$(python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl | grep 'Peak first_acc')"

# Launch hero
bash scripts/run_llama_single.sh --hero 2>&1 | tee runs/hero/pipeline_$(date +%Y%m%d_%H%M%S).log
```

---

## Why This Approach is Safe

1. **Incremental validation**: Each phase costs <10% of hero, catches 80% of issues
2. **Data-driven decisions**: Every decision point has clear metrics
3. **Fail-fast**: Stops at Phase 1 if architecture is broken (saves 90% of compute)
4. **Proven components**: Stage A checkpoint is validated, only extending Stage B
5. **Rich diagnostics**: Every step logged, every breakthrough detected

**Estimated timeline**:
- Phase 1 (extended smoke): 30 min → **START HERE**
- Phase 2 (if needed): 2 hours
- Phase 3 (if needed): 1 day
- Phase 4 (hero): 4-6 hours

**Estimated cost** (vs jumping to hero):
- Incremental approach: $5-25 (if Phases 1-2 succeed) vs $100 hero
- Risk reduction: 75% cheaper if smoke reveals issues
- Upside: If Phase 1 succeeds, we're hero-ready in 30 min

---

## Expected Outcomes

### If Phase 1 succeeds (F1 > 0):
```
Extended smoke (320 steps):
- Peak first_acc: 12-15%
- Final first_acc: 10-13% (stable)
- F1: 0.01-0.05 (breakthrough!)

Hero (2000-6600 steps):
- Peak first_acc: 15-20%
- F1: 0.10-0.20 (acceptance target)
```

### If Phase 1 fails (F1 = 0):
```
Extended smoke (320 steps):
- Peak first_acc: 10-12%
- F1: 0.000

→ Proceed to Phase 2 (hyperparameter sweep)
→ If Phase 2 fails, Phase 3 (scheduled sampling)
→ DO NOT RUN HERO until F1 > 0 achieved
```

---

## Immediate Next Step

**RUN PHASE 1 NOW**:

```bash
# Set up monitoring in separate terminal
watch -n 10 'python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl 2>/dev/null | tail -30'

# Run extended Stage B smoke
RESUME_FROM=runs/smoke/ckpt/stageA bash scripts/smoke_stageb_extended.sh

# When complete, check results
python scripts/analyze_breakthrough.py runs/smoke_stageb_ext/diagnostics.jsonl
bash scripts/monitor_f1_breakthrough.sh runs/smoke_stageb_ext/pipeline_*.log
```

**Decision point** (30 min from now):
- F1 > 0 → ✅ Hero-ready, proceed to Phase 4
- F1 = 0 → ⚠️ Analyze why, proceed to Phase 2/3

This ensures we're **100% sure** before committing to hero run.
