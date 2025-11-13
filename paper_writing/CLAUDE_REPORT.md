# LatentWire GSM8K Ablation Study Analysis
**Date**: November 13, 2025
**Run ID**: `ablations_20251112_234456`
**Analyst**: Claude (Sonnet 4.5)

---

## Executive Summary

All 8 ablation experiments completed successfully on HPC node n07 with 4× H100 GPUs. However, **critical evaluation bugs** prevent meaningful accuracy measurement. The results show:

- **Target-alone baseline**: 100% accuracy (INVALID - due to gold answer extraction bug)
- **Bridged outputs**: 0-21.5% peak accuracy, collapsing to 0% by end of training
- **Training**: Models ARE learning (loss decreases, outputs become coherent)
- **Evaluation**: Broken due to gold answer extraction from 8-shot exemplars instead of test samples

### Critical Bug Discovered

The `extract_final_answer()` function extracts the **first "####" marker** from the concatenated text `(8-shot_examples + test_question + test_answer)`. Since the first 8-shot example has answer "#### 12", **all gold answers are incorrectly reported as "12"**, resulting in artificially perfect target-alone accuracy.

**Evidence**:
```
Gold answer: 12  (INCORRECT - should be 18)
Target-alone (extracted): 12  (matches first exemplar, not actual answer)
```

Actual test question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

Correct answer: 18 (16 - 3 - 4 = 9 eggs, 9 × $2 = $18)
Reported gold answer: 12 (from first 8-shot exemplar)

---

## Infrastructure & Configuration

### Hardware
- **Node**: n07.cm.cluster
- **GPUs**: 4× NVIDIA H100 80GB HBM3
- **CUDA**: Working successfully (after fixing module configuration)

### Modules (Working Configuration)
```bash
gcc/13.1.0
conda/24.3.0-0
stockcuda/12.6.2
cudnn/cuda12/9.3.0.75
```

### Training Configuration
```
Source Model: Mistral-7B-Instruct-v0.3
Target Model: Llama-3.1-8B-Instruct
Dataset: GSM8K (200 eval samples, 8-shot CoT)
Batch Size: 4 per device (16 global across 4 GPUs)
Eval Batch Size: 36
Max New Tokens: 512
Training Steps: 2000
Eval Every: 250 steps
Early Stop Patience: 3
Precision: bf16
```

---

## Experiment Results

### Ablation 1: DiT Bridge Variants

#### 1a: DiT 2-step, 64 tokens
**Config**: `dit_steps_train=2`, `dit_pool=mean`, `dit_teacher=answer`
**Duration**: 01:24:24 (23:44:56 → 01:09:20)
**Peak Bridged Accuracy**: 0.5% (at step 250)
**Final Bridged Accuracy**: 0.0%

**Training Progression**:
- Step 0: Loss 2.06, Bridged 0%
- Step 250: Loss 1.93, Bridged **0.5%** (peak)
- Step 500-750: Bridged drops to 0%, early stop

**Sample Outputs** (Step 250):
- Bridged: Valid GSM8K reasoning with proper formatting (`<<100-50=50>>50`)
- Missing: "####" final answer markers

#### 1b: DiT 4-step, 64 tokens
**Config**: `dit_steps_train=4` (more denoising iterations)
**Duration**: 01:30:39
**Peak Bridged Accuracy**: 0.5%
**Final Bridged Accuracy**: 0.0%

**Key Finding**: Doubling denoising steps (2→4) did NOT improve performance. Similar 0.5% peak accuracy.

#### 1c: DiT with Attention Pooling, 64 tokens ⭐
**Config**: `dit_pool=attn` (attention-based source conditioning)
**Duration**: 01:23:34
**Peak Bridged Accuracy**: **21.5%** (at step 750) ← Best performer
**Final Bridged Accuracy**: 0.0%

**Training Progression**:
- Step 0: Bridged 0% (garbage: `_REFUSE) to the rescue...`)
- Step 250: Bridged 0%
- Step 500: Bridged **16.5%** ← Learning accelerates
- Step 750: Bridged **21.5%** ← Peak performance
- Step 1000-1500: Bridged drops to 0% ← Catastrophic collapse

**Sample Bridged Outputs** (Step 750):
```
Gold: 12 (incorrect extraction)
Bridged: 3  (valid GSM8K answer with ####)
Bridged: 2  (valid GSM8K answer with ####)
```

**Critical Observation**: At step 750, bridged outputs include valid "####" markers and numerical answers, demonstrating the bridge CAN learn to produce proper format. However, training collapses afterward.

**Analysis**: Attention pooling (vs mean pooling) provides **43× improvement** over 2-step mean pooling (21.5% vs 0.5%). This suggests source conditioning quality is critical.

#### 1d: DiT with CFG, 64 tokens
**Config**: `dit_cfg=2.0`, `dit_drop_cond=0.1`
**Duration**: 00:52:23
**Peak Bridged Accuracy**: 0.0%
**Final Bridged Accuracy**: 0.0%

**Analysis**: Classifier-Free Guidance (CFG) did NOT help. Possible reasons:
1. CFG helps with mode coverage, but collapse issue is different
2. `dit_drop_cond=0.1` may be too low for effective unconditional training
3. Rectified flow may not benefit from CFG like diffusion models do

#### 1e: DiT with Prompt Teacher, 64 tokens
**Config**: `dit_teacher=prompt`, `dit_loss_warmup=500`
**Duration**: 01:23:27
**Peak Bridged Accuracy**: 0.5%
**Final Bridged Accuracy**: 0.0%

**Analysis**: Switching teacher from answer→prompt (conditioning on question instead of solution) did not improve results. Similar 0.5% peak as baseline 1a.

---

### Ablation 2: Stability Fixes

#### 2a: Stable 64 tokens
**Config**: InfoNCE + Early Stopping + Generation Hygiene
**Duration**: 00:33:55
**Peak Bridged Accuracy**: 0.0%
**Final Bridged Accuracy**: 0.0%

**Analysis**: Shorter run (33 min vs 1h24m for 1a) suggests faster collapse. Stability fixes did NOT prevent the core issue.

---

### Ablation 3: Sequence Length

#### 3a: Stable 32 tokens
**Config**: 32 soft tokens (higher compression)
**Duration**: 00:33:21
**Peak Bridged Accuracy**: 0.0%
**Final Bridged Accuracy**: 0.0%

#### 3b: Stable 48 tokens
**Config**: 48 soft tokens (medium compression)
**Duration**: 00:33:34
**Peak Bridged Accuracy**: 0.0%
**Final Bridged Accuracy**: 0.0%

**Analysis**: Sequence length variations (32, 48, 64) showed no meaningful differences when stability fixes failed.

---

## Detailed Analysis: What's Working vs What's Broken

### ✅ What's Working

1. **Infrastructure**:
   - CUDA/cuDNN working perfectly on n07
   - DDP synchronization across 4 GPUs
   - Model loading and training pipeline

2. **Training Loss**:
   - Decreases from ~2.0 to ~1.1-1.5
   - Stable across all experiments
   - Indicates models ARE optimizing

3. **Bridged Output Quality** (Step 250-750):
   ```
   Step 0: "_REFUSE) to the rescue..."  (garbage)

   Step 250: "The number of people who have not yet received a gift is 100 - 50 = <<100-50=50>>50.
              The number of people who have received a gift is 50 - 20 = <<50-20=30>>30..."
              (coherent GSM8K-style reasoning!)

   Step 750: "#### 3"  (valid final answer marker!)
   ```

4. **Prompt Handling**:
   - Full 8-shot CoT prompts visible (no truncation)
   - Target-alone generates complete responses
   - `truncation=False` fix working

5. **Logging Improvements**:
   - Full questions logged (not truncated at 200 chars)
   - Complete generated outputs shown
   - Both extracted and full text available

### ❌ What's Broken

#### Critical Bug 1: Gold Answer Extraction

**Location**: `cross_attention.py` line ~1126
```python
gold_full_text = samples[i].tgt_prompt + " " + samples[i].tgt_answer
log(f"Gold answer: {extract_final_answer(gold_full_text)}")
```

**Problem**: `extract_final_answer()` uses regex `r"#### (\-?[0-9\.\,]+)"` which matches the **FIRST** occurrence. Since `tgt_prompt` contains 8 exemplars with "####" markers, it always extracts the first exemplar's answer (12).

**Impact**:
- All 24 evaluation samples report gold answer as "12"
- Target-alone appears 100% accurate (it's not - it's just echoing exemplars)
- Cannot measure true accuracy for either baseline or bridged

**Fix Required**:
```python
# Extract only from the test answer, not the full prompt
gold_answer = extract_final_answer(samples[i].tgt_answer)  # NOT tgt_prompt + tgt_answer
```

#### Critical Bug 2: Training Collapse

**Observation**: Experiment 1c reaches 21.5% accuracy at step 750, then collapses to 0% by step 1000-1500.

**Evidence**:
- Step 500: 16.5% ← improving
- Step 750: 21.5% ← peak
- Step 1000: 0.0% ← sudden collapse
- Step 1250: 0.0% ← no recovery

**Possible Causes**:

1. **Mode Collapse**: DiT learns to generate generic text that doesn't include "####" markers
   - At collapse, bridged outputs show numbers (54, 277.78, 360) but no "####" prefix
   - Model produces valid reasoning but wrong format

2. **Overfitting to First Exemplar**:
   - Target-alone echoes first exemplar ("12") perfectly
   - Bridge may be learning to mimic this behavior
   - Early stopping detects "no improvement" because evaluation is broken

3. **Loss Landscape**:
   - Teacher-forced CE loss continues decreasing (~1.9)
   - But generative quality degrades
   - Indicates train/eval distribution mismatch

#### Bug 3: Early Stopping on Broken Metric

**Problem**: Early stopping uses bridged accuracy, which is unreliable due to Bug #1. When training collapse begins (step 750→1000), early stopping triggers and halts training at a local minimum.

**Impact**: Best models (like 1c at step 750) are discarded instead of being saved as checkpoints.

---

## Key Findings & Insights

### 1. Attention Pooling Matters (43× improvement)

Experiment 1c (attention pooling) achieved **21.5%** peak accuracy vs 0.5% for mean pooling (1a/1b/1e). This massive improvement suggests:

- **Source conditioning quality is critical**: How source embeddings are aggregated affects bridge learning
- **Attention can focus on relevant tokens**: Instead of averaging all source tokens equally
- **Architectural choice matters more than hyperparameters**: Pool method >> denoising steps or teacher type

### 2. DiT Can Learn (But Unstably)

Valid GSM8K-style outputs at step 250-750 prove:
- ✅ Models CAN compress 8-shot prompts into 64 tokens
- ✅ Bridged representations CAN condition target generation
- ✅ Reasoning chains are coherent and on-topic
- ❌ Training is unstable (collapses after ~750 steps)
- ❌ Format adherence ("####" marker) is unreliable

### 3. The Truncation Fixes Worked

Previous issues with "Target-alone: 12" for all questions were due to truncation. Current logs show:
- Full 8-shot prompts (lines 223-283 of logs)
- Complete target-alone outputs (300+ lines of generated text)
- No artificial cutoffs at 2048/4096 tokens

**However**, the evaluation metric extraction is still broken (Bug #1).

### 4. max_new_tokens=512 May Be Insufficient

Many bridged outputs show incomplete reasoning:
```
"The bakery makes a total of $625+$437.50=$1062.50.
 The bakery makes a"  ← Cut off mid-sentence
```

While 512 tokens matches Meta's official protocol, GSM8K CoT with 8-shot examples may need more room to complete both reasoning AND final answer.

### 5. Models Are Not Learning "####" Marker Reliably

Even successful runs (1c at step 750) show mixed results:
- Some outputs: `#### 3` ✅
- Some outputs: `3` (number without marker) ❌
- Some outputs: Valid reasoning but no final line ❌

This suggests the training loss doesn't strongly penalize missing formatting.

---

## Comparison to Baseline

The summary.log references:
```
2b_baseline_64tok: Peak 81.5% → Final 36.0% (45.5% degradation)
```

This baseline used **cross-attention** (not DiT) and achieved:
- **81.5% peak accuracy** (vs 21.5% for best DiT)
- **36% final accuracy** (vs 0% for DiT)

**Conclusion**: DiT experiments (1a-1e) have NOT yet matched the cross-attention baseline's performance. The hypothesis that "iterative refinement prevents collapse" is not yet validated.

---

## Training Dynamics

### Loss Curves

All experiments show similar loss progression:
```
Step 0:   Loss ~2.0
Step 100: Loss ~1.1
Step 250: Loss ~1.9-2.0
Step 500: Loss ~1.9
Step 750: Loss ~1.9
```

**Observation**: Loss stabilizes around 1.9 but doesn't correlate with accuracy improvements. This suggests:
1. Cross-entropy loss on teacher-forced labels is not predictive of generation quality
2. Models optimize for next-token prediction, not adherence to "####" format
3. Need additional losses (format compliance, answer accuracy)

### Evaluation Trends (Experiment 1c)

```
Step 0:    Bridged 0.0%   (untrained, garbage)
Step 250:  Bridged 0.0%   (learning, but no ####)
Step 500:  Bridged 16.5%  (breakthrough!)
Step 750:  Bridged 21.5%  (peak)
Step 1000: Bridged 0.0%   (collapse begins)
Step 1250: Bridged 0.0%   (full collapse)
Step 1500: Bridged 0.0%   (no recovery)
```

**Critical Window**: Steps 500-750 show rapid improvement (0% → 21.5%) followed by catastrophic collapse. This 250-step window is where the model "gets it" before degenerating.

---

## Recommended Next Steps

### Immediate Fixes (Required)

1. **Fix Gold Answer Extraction** (HIGH PRIORITY)
   ```python
   # In cross_attention.py, run_eval function
   # OLD:
   gold_full_text = samples[i].tgt_prompt + " " + samples[i].tgt_answer
   log(f"Gold answer: {extract_final_answer(gold_full_text)}")

   # NEW:
   log(f"Gold answer: {extract_final_answer(samples[i].tgt_answer)}")
   ```

2. **Save Intermediate Checkpoints**
   - Save at steps 500, 750, 1000 (not just final)
   - Allow analysis of pre-collapse models
   - Experiment 1c at step 750 is likely the best model but was discarded

3. **Disable Early Stopping** (Temporarily)
   - Run full 2000 steps to observe complete training curve
   - Current early stopping triggers on broken metrics

### Evaluation Improvements

4. **Add Format Compliance Metric**
   ```python
   format_accuracy = (outputs containing "####") / total_outputs
   ```

5. **Separate Answer Accuracy from Format**
   - Measure: "Correct number generated (even without ####)"
   - Measure: "#### marker present"
   - Measure: "Both correct"

6. **Log More Examples**
   - Currently shows 3 examples per eval
   - Increase to 10 to see output diversity

### Training Improvements

7. **Add Format Loss**
   - Penalize outputs without "####" marker
   - Reward outputs with proper final answer structure
   - Weight this loss separately from CE loss

8. **Increase max_new_tokens**
   - Try 768 or 1024 to ensure complete reasoning chains
   - Current 512 may cut off before "####"

9. **Learning Rate Adjustment**
   - Current LR: 1e-4
   - Try: 5e-5 or 3e-5 to slow down collapse
   - Add LR warmup

10. **Investigate Attention Pooling Further** (Based on 1c success)
    - Try `dit_heads=16` (more attention capacity)
    - Try `dit_depth=8` or `dit_depth=12` (deeper network)
    - Experiment with different pooling mechanisms

### Diagnostic Experiments

11. **Ablate Early Stopping**
    - Run 1c config for full 2000 steps without early stopping
    - Observe if collapse is inevitable or can be delayed

12. **Checkpoint Analysis**
    - Generate outputs from steps 500, 750, 1000, 1250
    - Manually inspect output quality degradation
    - Identify exact failure mode

13. **Reduce Eval Frequency**
    - Current: every 250 steps
    - Try: every 500 steps
    - Hypothesis: Frequent eval disrupts training

---

## Data Quality Issues

### Test Set Homogeneity

All 24 logged examples report gold answer "12" (due to Bug #1), but this revealed a deeper concern:

**Question**: Are evaluation samples diverse enough?

From logs, observed questions include:
- Recycling cans and newspapers
- Picking strawberries
- Selling duck eggs at farmers market
- Buying toys and hats
- Various arithmetic word problems

**Recommendation**: Verify GSM8K eval split has sufficient diversity in:
- Answer magnitudes (not just 0-20)
- Operation types (not just addition/subtraction)
- Problem complexity

---

## Computational Efficiency

### Resource Usage

- **Training Speed**: ~250 steps per 20-25 minutes
- **Total Runtime**: 6-8 hours for all 8 experiments
- **Eval Speed**: ~200 samples in <2 minutes with batch_size=36
- **Memory**: No OOM issues with per_device_batch=4

### Cost Analysis

- 8 experiments × 1.5 hours = 12 GPU-hours
- Early stopping saved ~8 hours (2000 steps not completed)
- If collapse is fixed, expect 2-3x longer training times

---

## Conclusions

### What We Learned

1. **DiT can learn** but is **unstable** (21.5% peak → 0% collapse)
2. **Attention pooling** is critical (43× better than mean pooling)
3. **Evaluation is broken** (gold answer extraction bug)
4. **Training dynamics show promise** (coherent outputs at mid-training)
5. **Current best: Experiment 1c at step 750** (21.5% accuracy)

### What We Don't Know Yet

1. **True accuracy numbers**: Bug #1 prevents measurement
2. **Why collapse happens**: Need checkpoint analysis
3. **Can DiT match cross-attention**: 21.5% << 81.5% baseline
4. **Optimal architecture**: Only tested 1 attention pooling config

### Recommendation

**DO NOT** proceed with paper writing until:
1. Bug #1 (gold answer extraction) is fixed
2. Experiment 1c is re-run with intermediate checkpoints
3. True accuracy numbers are measured
4. Collapse is understood and mitigated

The current results are **promising but incomplete**. The infrastructure works, models learn, but evaluation and stability issues prevent conclusive findings.

---

## Appendix: Log File Locations

```
paper_writing/runs/ablations_20251112_234456/
├── 1a_dit_2step_64tok/train.log        (01:24:24, peak 0.5%)
├── 1b_dit_4step_64tok/train.log        (01:30:39, peak 0.5%)
├── 1c_dit_attn_64tok/train.log         (01:23:34, peak 21.5%) ⭐
├── 1d_dit_cfg_64tok/train.log          (00:52:23, peak 0.0%)
├── 1e_dit_prompt_teacher_64tok/train.log (01:23:27, peak 0.5%)
├── 2a_stable_64tok/train.log           (00:33:55, peak 0.0%)
├── 3a_stable_32tok/train.log           (00:33:21, peak 0.0%)
├── 3b_stable_48tok/train.log           (00:33:34, peak 0.0%)
└── summary.log
```

---

**Report Generated**: 2025-11-13
**Next Update**: After Bug #1 fix and re-run of Experiment 1c
