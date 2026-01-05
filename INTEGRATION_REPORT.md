# Integration Analysis Report

## Executive Summary
Analysis of critical integrations between components in the LatentWire/telepathy codebase reveals several missing integrations and potential interface issues.

## Integration Status

### 1. linear_probe_baseline.py ← run_unified_comparison.py
**Status: NOT INTEGRATED** ❌

**Finding:** The LinearProbeBaseline class exists but is not imported or used in run_unified_comparison.py

**Issue:**
- linear_probe_baseline.py provides a sklearn-based LogisticRegression baseline
- run_unified_comparison.py implements its own baselines but does not include the linear probe
- This is a critical missing baseline that reviewers expect

**Required Integration:**
```python
# In run_unified_comparison.py, add:
from telepathy.linear_probe_baseline import LinearProbeBaseline

# Add to baseline evaluation section:
probe = LinearProbeBaseline(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device=device,
    layer_indices=[16, 24, 31],  # Multiple layers
    pooling_strategy="mean",
    standardize=True
)
probe_results = probe.evaluate(eval_ds, dataset_name)
```

### 2. rouge_metrics.py ← run_comprehensive_revision.py
**Status: NOT INTEGRATED** ❌

**Finding:** RougeMetrics class exists but is not imported in run_comprehensive_revision.py

**Issue:**
- rouge_metrics.py provides ROUGE score computation for generation tasks
- run_comprehensive_revision.py mentions generation tasks (Phase 5: XSUM) but doesn't use RougeMetrics
- This means generation quality metrics are not being computed

**Required Integration:**
```python
# In run_comprehensive_revision.py, add:
from telepathy.rouge_metrics import RougeMetrics

# In Phase 5 (generation tasks):
rouge_scorer = RougeMetrics()
rouge_results = rouge_scorer.compute_rouge(predictions, references)
```

### 3. memory_configs.py ← training scripts
**Status: PARTIALLY INTEGRATED** ⚠️

**Finding:** memory_configs.py exists but only used in example scripts, not main training

**Issue:**
- memory_configs.py provides safe batch size calculations to avoid OOM
- Main training scripts (run_unified_comparison.py, run_comprehensive_revision.py) use hardcoded batch sizes
- This risks OOM errors on different GPU configurations

**Required Integration:**
```python
# In training scripts, add:
from telepathy.memory_configs import get_memory_safe_config

# Replace hardcoded batch_size with:
config = get_memory_safe_config(
    gpu_memory_gb=80,  # H100
    num_models=2,
    model_size_gb=16,  # 8B models
    soft_tokens=args.soft_tokens
)
batch_size = config['batch_size']
```

### 4. statistical_tests.py ← aggregate_results.py
**Status: PARTIALLY INTEGRATED** ⚠️

**Finding:** aggregate_results.py uses scipy.stats directly but not the advanced methods from statistical_tests.py

**Issue:**
- statistical_tests.py provides bootstrap_ci, mcnemar_test, effect_size calculations
- aggregate_results.py only uses basic scipy.stats.ttest_ind
- Missing proper confidence intervals and paired statistical tests

**Required Integration:**
```python
# In aggregate_results.py, add:
from telepathy.statistical_tests import bootstrap_ci, mcnemar_test, cohens_d

# Replace basic t-test with:
mean, ci = bootstrap_ci(scores, n_bootstrap=10000)
p_value = mcnemar_test(bridge_predictions, baseline_predictions)
effect = cohens_d(bridge_scores, baseline_scores)
```

### 5. SLURM scripts → Python modules
**Status: INTEGRATED** ✅

**Finding:** SLURM scripts properly call Python modules with correct paths

**Verified:**
- submit_comprehensive_revision.slurm calls run_comprehensive_revision.py
- Correct working directory: /projects/m000066/sujinesh/LatentWire
- Proper environment setup (PYTHONPATH, HF_HOME)
- Git pull/push integration working

## Critical Missing Integrations

### 1. Linear Probe Baseline Not Used
**Impact:** Missing critical baseline for reviewers
**Fix Priority:** HIGH
**Action:** Add LinearProbeBaseline to run_unified_comparison.py

### 2. ROUGE Metrics Not Used for Generation
**Impact:** Cannot measure generation quality
**Fix Priority:** HIGH
**Action:** Add RougeMetrics to generation evaluation

### 3. Memory Safety Not Enforced
**Impact:** Risk of OOM errors
**Fix Priority:** MEDIUM
**Action:** Use get_memory_safe_config() for batch sizing

### 4. Statistical Tests Underutilized
**Impact:** Weaker statistical claims in paper
**Fix Priority:** MEDIUM
**Action:** Use advanced statistical tests from statistical_tests.py

## Interface Compatibility Issues

### 1. Data Format Mismatch
**Location:** linear_probe_baseline.py expects different dataset format than run_unified_comparison.py provides

**Issue:**
- LinearProbeBaseline.evaluate() expects dataset with 'text' and 'label' fields
- run_unified_comparison uses different field names per dataset

**Fix:**
```python
# Add adapter in run_unified_comparison.py:
def adapt_dataset_for_probe(dataset, config):
    return dataset.map(lambda x: {
        'text': x[config['text_field']],
        'label': x[config['label_field']]
    })
```

### 2. Return Value Inconsistency
**Location:** Different modules return results in different formats

**Issue:**
- linear_probe_baseline returns dict with 'accuracy', 'f1_score', 'cv_scores'
- run_unified_comparison expects dict with 'accuracy', 'latency_ms', 'memory_mb'

**Fix:** Standardize return format across all baselines

## Circular Dependencies
**Status:** NONE FOUND ✅

No circular import dependencies detected between modules.

## Recommendations

### Immediate Actions (Priority 1)
1. **Integrate LinearProbeBaseline into run_unified_comparison.py**
   - Add import and initialization
   - Add evaluation call in baseline section
   - Save results to JSON

2. **Add RougeMetrics to generation tasks**
   - Import in run_comprehensive_revision.py
   - Compute ROUGE scores for XSUM task
   - Include in results output

### Short-term Actions (Priority 2)
1. **Enforce memory safety**
   - Replace hardcoded batch sizes with memory_configs
   - Add GPU memory detection
   - Implement adaptive batching

2. **Enhance statistical testing**
   - Replace t-tests with bootstrap CIs
   - Add McNemar tests for paired comparisons
   - Include effect size calculations

### Long-term Actions (Priority 3)
1. **Standardize interfaces**
   - Create base class for all baselines
   - Standardize return formats
   - Add type hints throughout

2. **Add integration tests**
   - Test data flow between modules
   - Verify interface compatibility
   - Add CI/CD pipeline

## Testing Recommendations

### Integration Test Suite
```python
# test_integrations.py
def test_linear_probe_integration():
    """Test LinearProbeBaseline works with run_unified_comparison data"""
    pass

def test_rouge_metrics_integration():
    """Test RougeMetrics works with generation outputs"""
    pass

def test_memory_config_integration():
    """Test memory configs prevent OOM"""
    pass

def test_statistical_integration():
    """Test statistical tests work with aggregated results"""
    pass
```

## Conclusion

The codebase has several critical missing integrations that prevent full functionality:
1. Linear probe baseline not used (reviewer requirement)
2. ROUGE metrics not computed for generation tasks
3. Memory safety not enforced
4. Statistical tests underutilized

These issues should be addressed before running final experiments for the paper.