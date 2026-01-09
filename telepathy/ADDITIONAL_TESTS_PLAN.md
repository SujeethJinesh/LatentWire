# Additional Tests to Bolster the Telepathy Paper

*Priority-ranked experiments that can be completed within 1 week to strengthen the MLSys 2025 submission*

## 🚨 Critical Gaps to Address (Must Do)

### 1. **Statistical Rigor** (1-2 days)
Current paper lacks basic statistical validation that reviewers will expect:

```python
# Run all existing experiments with:
- 5 random seeds (currently single-run)
- Full test sets (not subsamples)
- Bootstrap confidence intervals (95% CI)
- McNemar's test for paired comparisons
- Report mean ± std for all metrics
```

**Implementation**:
```bash
for seed in 42 123 456 789 2024; do
    python telepathy/eval_classification.py \
        --checkpoint $CKPT \
        --seed $seed \
        --full_test \
        --bootstrap_ci
done
```

**Addresses**: "Are results cherry-picked?" concern
**Time**: 1-2 days on 4 H100s
**Impact**: ESSENTIAL - paper will be rejected without this

### 2. **Linear Probe Baseline** (1 day)
Reviewers will ask: "How much better is Perceiver than simple linear projection?"

```python
# For each dataset, train sklearn LogisticRegression on:
- Llama layer 24 embeddings → Mistral predictions
- Compare with Perceiver bridge accuracy
```

**Expected Results**:
- Linear probe: ~75-80% accuracy
- Perceiver bridge: 94.7% accuracy
- Shows Perceiver's value beyond simple projection

**Time**: 4-6 hours
**Impact**: HIGH - validates architectural choices

### 3. **Production Metrics** (1 day)
MLSys cares about systems efficiency:

```python
# Measure and report:
- Throughput (samples/second) vs batch size [1, 8, 32, 128]
- Latency breakdown (encode, bridge, decode)
- Memory usage scaling
- Quantization impact (fp16, int8, int4)
```

**Expected Results**:
- Linear throughput scaling to batch 32
- 22× speedup breakdown: 41% encoding, 5% bridge, 54% saved generation
- Int8 maintains >92% accuracy

**Time**: 4-5 hours
**Impact**: HIGH - MLSys is a systems conference

## 📊 High-Value Additional Benchmarks (Should Do)

### 4. **IMDB Movie Reviews** (3 hours)
Tests longer text handling (231 words vs SST-2's 19 words)

```bash
python telepathy/train_bridge.py --dataset imdb --epochs 5
python telepathy/eval_classification.py --dataset imdb
```

**Expected**: 90-92% accuracy (bridge advantage on longer texts)
**Addresses**: "Only short text" concern

### 5. **MNLI Natural Language Inference** (8 hours)
Standard GLUE benchmark, 3-way entailment classification

```bash
python telepathy/train_bridge.py --dataset mnli --epochs 10
```

**Expected**: 75-80% accuracy (shows reasoning limitations)
**Addresses**: "Beyond simple sentiment" concern

### 6. **20 Newsgroups** (4 hours)
Tests scaling to 20 classes (vs 4 in AG News)

**Expected**: Bridge advantage grows with more classes
**Addresses**: Multi-class scaling question

### 7. **Model Size Scaling** (12 hours)
Test with Phi-3-mini (3.8B) and Llama-2-13B

```python
model_pairs = [
    ("microsoft/phi-3-mini", "mistralai/Mistral-7B"),  # 3.8B → 7B
    ("meta-llama/Llama-2-13b", "mistralai/Mistral-7B") # 13B → 7B
]
```

**Expected**:
- Smaller source: -5% accuracy but still >85%
- Larger source: +2-3% accuracy ceiling

**Addresses**: "Only one model size" concern

## 🔬 Interpretability Suite (Nice to Have)

### 8. **Attention Visualization** (8 hours)
Show what Perceiver queries focus on:

```python
# Hook into cross-attention layers
# Generate heatmaps showing:
- Which soft tokens attend to entities/sentiment/syntax
- Attention entropy (focused vs diffuse)
- Layer-wise attention evolution
```

**Deliverable**: Figure showing specialized attention patterns
**Impact**: MEDIUM - helps explain "why it works"

### 9. **Soft Token Probing** (6 hours)
What information do soft tokens encode?

```python
# Train linear probes to extract:
- Sentiment (expected: 85% accuracy)
- Topic (expected: 70% accuracy)
- Length (expected: 90% accuracy)
- Named entities (expected: 75% accuracy)
```

**Deliverable**: Table of probe accuracies
**Impact**: MEDIUM - shows interpretable features

### 10. **Causal Interventions** (8 hours)
Can we control outputs by editing soft tokens?

```python
# Experiments:
- Swap soft tokens between examples
- Interpolate between positive/negative
- Add directional vectors to flip sentiment
```

**Expected**: 60-70% successful manipulations
**Impact**: MEDIUM - proves causal relationship

## 🛡️ Robustness Tests (Important for Production Claims)

### 11. **Adversarial Robustness** (4 hours)
Test with TextFooler and BERT-Attack:

```python
# Generate adversarial examples
# Measure accuracy degradation:
- Text baseline: -40% accuracy
- Bridge: -15% accuracy (more robust!)
```

**Impact**: MEDIUM - security-conscious reviewers

### 12. **Noise Injection** (3 hours)
Add varying levels of noise:

```python
noise_levels = [0.1, 0.2, 0.3]  # Gaussian noise std
# Measure graceful degradation
```

**Expected**: <10% degradation at 0.1 noise
**Impact**: MEDIUM - real-world robustness

### 13. **Cross-Dataset Generalization** (6 hours)
Train on SST-2, test on IMDB/Rotten Tomatoes:

**Expected**: 70-75% zero-shot transfer
**Impact**: MEDIUM - generalization claims

## 🌍 Multilingual Evaluation (Stretch Goal)

### 14. **XNLI Subset** (8 hours)
Test on 3 languages (Spanish, French, German):

```python
languages = ['es', 'fr', 'de']
# Use same English-trained bridge
```

**Expected**: 70-80% accuracy (15% drop from English)
**Impact**: LOW-MEDIUM - shows language independence

## 📋 Implementation Priority Order

### Week 1 Schedule (Recommended)

**Day 1-2**: Statistical validation (MUST DO)
- Re-run existing experiments with 5 seeds
- Add confidence intervals and significance tests

**Day 3**: Baselines and production metrics
- Linear probe baseline
- Throughput/latency measurements

**Day 4-5**: Key additional benchmarks
- IMDB (long text)
- MNLI (reasoning)
- 20 Newsgroups (multi-class)

**Day 6**: Model scaling experiments
- Phi-3-mini → Mistral
- Llama-13B → Mistral

**Day 7**: Analysis and writing
- Generate tables/figures
- Update paper with new results

## 🎯 Success Metrics

The additional tests succeed if they show:

1. **Statistical rigor**: All results with 95% CI, p < 0.01
2. **Baseline comparison**: Bridge > linear probe by >15%
3. **Production ready**: Linear scaling, <50ms latency
4. **Generalization**: Consistent advantage across datasets
5. **Interpretability**: Attention patterns align with intuition

## 💻 Quick Start Commands

```bash
# Clone and setup
cd /projects/m000066/sujinesh/LatentWire
git pull

# Run statistical validation
bash telepathy/scripts/statistical_validation.sh

# Run additional benchmarks
bash telepathy/scripts/additional_benchmarks.sh

# Generate paper tables
python telepathy/scripts/generate_tables.py

# Run full suite (48 hours on 4 H100s)
sbatch telepathy/submit_full_evaluation.slurm
```

## 📊 Expected Impact on Paper

With these additional tests, the paper transforms from:
- **Current**: "Interesting idea with promising results"
- **Enhanced**: "Rigorously validated system ready for production"

Key improvements:
- Statistical significance for all claims
- Comparison with simpler baselines
- Production metrics for MLSys audience
- Broader benchmark coverage
- Interpretability analysis
- Robustness validation

## 🚀 Bottom Line

**Minimum viable additions** (2-3 days):
1. Statistical validation with 5 seeds
2. Linear probe baseline
3. Production metrics

**Recommended additions** (5-7 days):
4. IMDB, MNLI, 20 Newsgroups
5. Model size scaling
6. Basic interpretability

**Full suite** (10-14 days):
7. All interpretability experiments
8. Robustness tests
9. Multilingual evaluation
10. Cross-architecture experiments

The minimum viable additions are ESSENTIAL for acceptance. The recommended additions would significantly strengthen the paper. The full suite would make it a strong MLSys paper with comprehensive validation.