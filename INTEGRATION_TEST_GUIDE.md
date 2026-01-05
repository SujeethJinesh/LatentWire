# Integration Test Guide

## Overview

The LatentWire/Telepathy project now includes comprehensive integration tests that validate the entire pipeline works correctly before running expensive experiments on HPC.

## Test Files

### 1. Minimal Integration Test (`telepathy/test_integration.py`)

**Purpose**: Quick validation that all components work together
**Runtime**: <5 minutes
**Coverage**:
- Bridge training with dummy data
- Linear probe baseline
- Statistical testing utilities
- Result aggregation

**Usage**:
```bash
# Run with bash wrapper
bash telepathy/run_integration_test.sh

# Or run directly
python telepathy/test_integration.py
```

### 2. CI/CD Integration Test (`telepathy/test_integration_ci.py`)

**Purpose**: Comprehensive testing with real models and data
**Runtime**:
- Quick mode: <2 minutes
- Full mode: <10 minutes

**Features**:
- Tests with actual transformer models (GPT-2, DistilGPT-2)
- Uses real datasets (SST-2, TREC)
- Measures memory usage
- Validates reproducibility with seeds
- Checks all output formats
- Generates detailed test report

**Usage**:
```bash
# Quick mode (for rapid iteration)
python telepathy/test_integration_ci.py --quick

# Full mode (before HPC submission)
python telepathy/test_integration_ci.py

# Keep temp files for debugging
python telepathy/test_integration_ci.py --keep-temp
```

## When to Run Integration Tests

1. **Before HPC Submission**: Always run `test_integration_ci.py` in full mode
2. **After Major Changes**: Run minimal test to catch obvious breaks
3. **CI/CD Pipeline**: Use quick mode for automated testing
4. **Debugging**: Use `--keep-temp` to preserve test artifacts

## Test Components Validated

### Core Pipeline
- [x] Data loading and preprocessing
- [x] Model loading and compatibility
- [x] Bridge training loop
- [x] Evaluation pipeline
- [x] Statistical testing
- [x] Result aggregation

### Quality Checks
- [x] Reproducibility with fixed seeds
- [x] Memory usage tracking
- [x] Output format validation
- [x] Error handling
- [x] Performance timing

## Expected Output

### Successful Run
```
================================================================================
 CI/CD INTEGRATION TEST SUITE
================================================================================
Mode: Quick
Temp directory: /tmp/ci_telepathy_XXXXX
Started: 2025-01-04T21:00:00

Testing Data Loading...
  ✓ sst2: 100 samples, 2 classes
  ✓ trec: 100 samples, 6 classes

Testing Model Loading...
  ✓ gpt2: 124.4M params, 12 layers

...

TEST SUMMARY
================================================================================
Total Tests: 7
Passed: 7 (100.0%)
Failed: 0

Test Results:
  ✓ Data Loading
  ✓ Model Loading
  ✓ Bridge Training
  ✓ Evaluation Pipeline
  ✓ Statistical Validation
  ✓ Reproducibility
  ✓ Output Formats

Full report saved to: /tmp/ci_telepathy_XXXXX/ci_test_report.json
Total time: 95.3s
```

### Failed Run
The test will:
1. Show which component failed
2. Preserve temp directory for debugging
3. Generate error report with stack traces
4. Exit with non-zero code for CI/CD

## Integration with Existing Experiments

The integration tests use the same components as production experiments:
- `LatentBridgeV15` for cross-model communication
- `LinearProbeBaseline` for sklearn-based classification
- `statistical_testing` for rigorous metrics
- Standard data loading pipelines

## Troubleshooting

### Common Issues

1. **Import Errors**: The tests include fallback imports for missing components
2. **Memory Issues**: Use `--quick` mode or reduce sample sizes
3. **GPU Availability**: Tests automatically fall back to CPU
4. **Model Download**: First run may be slow due to model downloads

### Debug Mode

To debug failures:
```bash
# Run with preserved artifacts
python telepathy/test_integration_ci.py --keep-temp

# Check the test report
cat /tmp/ci_telepathy_*/ci_test_report.json

# Examine specific test outputs
ls -la /tmp/ci_telepathy_*/
```

## Continuous Integration

For GitHub Actions or other CI systems:
```yaml
- name: Run Integration Tests
  run: |
    python telepathy/test_integration_ci.py --quick
  timeout-minutes: 5
```

## Next Steps

1. Run integration tests before any HPC submission
2. Add new test cases as pipeline evolves
3. Monitor test execution times
4. Use test results to catch regressions early

The integration tests provide confidence that the entire pipeline works correctly before investing GPU hours in full experiments.