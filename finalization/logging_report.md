# Logging Infrastructure Report

Generated: fix_logging.py
Date: $(date)

## Shell Scripts

Total: 13
Issues: 9

### Scripts with issues:

- **RUN_THIS.sh**
  - Missing 'tee' for output capture
  - No timestamp in log file name
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **test_everything.sh**
  - No timestamp in log file name
- **run_validation.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - No timestamp in log file name
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **monitor_experiment.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **test_memory_calculations.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - No timestamp in log file name
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **compile_paper.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - No timestamp in log file name
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **quick_start.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - No timestamp in log file name
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **test_orchestrator.sh**
  - Missing 'tee' for output capture
  - No LOG_FILE variable or log file handling
  - Missing proper stderr+stdout capture (2>&1 | tee)
- **run_finalization.sh**
  - No LOG_FILE variable or log file handling

## Python Scripts

Total: 30
Issues: 29

### Scripts with logging improvements needed:

- **analyze_actual_compression.py**: 1 issues
- **generate_paper.py**: 3 issues
- **quick_check.py**: 3 issues
- **aggregate_results.py**: 4 issues
- **monitor.py**: 1 issues
- **verify_linear_probe_complete.py**: 1 issues
- **test_all.py**: 4 issues
- **test_ddp.py**: 3 issues
- **check_environment.py**: 1 issues
- **fix_compression_metrics.py**: 1 issues

## Recommendations

1. All shell scripts should use `tee` with timestamped log files
2. Python scripts should use `flush=True` for critical output
3. Consider using structured logging (logging module) for complex Python scripts
4. Ensure all training scripts save logs to persistent storage
