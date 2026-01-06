# LatentWire Integration Test Suite

## Overview

This integration test suite validates the complete LatentWire training and evaluation pipeline end-to-end. It ensures all components work together correctly, from initial training through checkpoint resumption to final evaluation.

## Test Components

### 1. `test_integration.py`
Main integration test that validates:
- **Import Integrity**: All required modules can be imported
- **Initial Training**: Training runs successfully with checkpoint saving
- **Checkpoint Resume**: Training can resume from saved checkpoints with proper state restoration
- **Evaluation**: Evaluation works on saved checkpoints and produces metrics

### 2. `validate_integration_test.py`
Pre-flight validation script that checks:
- Python version (3.8+ required)
- Required packages installed (PyTorch, Transformers, etc.)
- Project structure and files exist
- GPU/compute resources available
- Disk space sufficient (>10GB recommended)

### 3. `run_integration_test.sh`
Bash wrapper that:
- Sets up proper environment variables
- Creates output directories and logs
- Runs the full test suite
- Generates summary report

## Usage

### Step 1: Validate Environment
First, ensure your environment is properly configured:

```bash
python validate_integration_test.py
```

This will check all prerequisites and report any issues that need fixing.

### Step 2: Run Integration Tests
Once validation passes, run the full integration test:

```bash
bash run_integration_test.sh
```

Or run the Python test directly with options:

```bash
python test_integration.py --verbose --output results.json
```

## Test Workflow

The integration test executes the following workflow:

1. **Setup Phase**
   - Create temporary test directory
   - Set environment variables
   - Validate imports

2. **Training Phase** (Test 1)
   - Train for 50 samples with small model configuration
   - Save checkpoints every 5 samples
   - Verify checkpoint creation

3. **Resume Phase** (Test 2)
   - Load checkpoint from Test 1
   - Continue training for 100 samples
   - Verify state restoration and continued progress

4. **Evaluation Phase** (Test 3)
   - Load final checkpoint
   - Run evaluation on 10 samples
   - Verify metrics generation

5. **Cleanup Phase**
   - Remove temporary files
   - Generate test report

## Output Files

The test suite generates several output files:

```
runs/integration_test/
├── integration_test_YYYYMMDD_HHMMSS.log  # Full test log
├── test_results.json                      # Structured test results
├── train_initial/                         # Initial training artifacts
│   ├── checkpoint_*/                      # Saved checkpoints
│   └── training_stats.json               # Training statistics
├── train_resume/                          # Resume training artifacts
│   ├── checkpoint_*/                      # New checkpoints
│   └── training_stats.json               # Continued statistics
└── eval_results/                          # Evaluation outputs
    └── *.json                             # Evaluation metrics
```

## Test Configuration

The test uses minimal configuration for fast execution:

- **Samples**: 50-100 (small dataset subset)
- **Batch Size**: 4
- **Latent Length**: 8
- **Latent Dimension**: 128
- **Models**: Llama-3.1-8B + Qwen2.5-7B (sequential mode)

These settings ensure tests complete quickly while still exercising all components.

## Success Criteria

The integration test is considered successful when:

1. All imports work without errors
2. Initial training completes and saves checkpoints
3. Resume correctly loads state and continues training
4. Evaluation produces valid metrics (F1, EM, NLL)
5. No unexpected errors or crashes occur

## Troubleshooting

### Common Issues and Solutions

**Issue**: Import errors
- **Solution**: Ensure PYTHONPATH includes project root
- Run: `export PYTHONPATH=.`

**Issue**: CUDA/GPU errors
- **Solution**: Test runs on CPU by default, GPU optional
- For GPU: ensure PyTorch CUDA version matches system

**Issue**: Out of memory
- **Solution**: Test uses small configurations
- Reduce batch_size or latent_len if needed

**Issue**: Checkpoint not found
- **Solution**: Check test_dir path in logs
- Ensure write permissions in output directory

**Issue**: Timeout errors
- **Solution**: Default timeout is 600s for training
- Increase timeout in test_integration.py if needed

## Integration with CI/CD

The test suite returns appropriate exit codes:
- **0**: All tests passed
- **1**: One or more tests failed

This makes it suitable for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    python validate_integration_test.py
    bash run_integration_test.sh
```

## Performance Expectations

On typical hardware:
- **Validation**: ~5 seconds
- **Full Test Suite**: ~5-15 minutes
  - Initial training: 2-5 minutes
  - Resume training: 2-5 minutes
  - Evaluation: 1-2 minutes

## Extending the Tests

To add new test cases:

1. Add method to `IntegrationTest` class
2. Follow naming convention: `test_<feature_name>`
3. Return dict with `success` key and details
4. Update test counting in results

Example:
```python
def test_new_feature(self) -> Dict[str, Any]:
    self.log("Testing new feature...")
    # Test implementation
    success = True  # Based on test outcome

    if success:
        self.results["tests_passed"] += 1
    else:
        self.results["tests_failed"] += 1
        self.results["failures"].append("new_feature")

    return {"success": success, ...}
```

## Related Documentation

- `CHECKPOINT_RESUME_GUIDE.md`: Detailed checkpoint system documentation
- `FAILURE_RECOVERY_GUIDE.md`: Recovery procedures for failures
- `test_checkpoint_integration.py`: Lower-level checkpoint tests
- `test_preemption_resume.py`: Preemption-specific tests