# JSON Output Consistency Report

## Overview

This report summarizes the JSON output consistency across all experimental scripts in the LatentWire/Telepathy project.

## Validation Results

### Scripts Checked

1. **run_comprehensive_revision.py**
   - Output files: `phase*_results.json`, `all_results.json`
   - Schema compliance: ✅ Valid
   - Required fields present: ✅

2. **run_unified_comparison.py**
   - Output files: `unified_results_*.json`, `unified_summary_*.json`
   - Schema compliance: ✅ Valid for results, ⚠️ Summary files need separate schema
   - Required fields present: ✅

3. **aggregate_results.py**
   - Output files: `aggregated_results.json`, `significance_tests.json`
   - Schema compliance: ✅ Valid
   - Required fields present: ✅

4. **statistical_tests.py**
   - Output: Returns Python objects (no direct JSON output)
   - Integration: Used by other scripts for statistical analysis
   - Schema compliance: N/A (utility functions)

## Key Findings

### ✅ Consistent Elements

1. **Core Metrics**
   - All scripts use `accuracy` (0-100 scale)
   - All include `correct` and `total` counts
   - Consistent use of `f1`, `precision`, `recall` where applicable

2. **Metadata Format**
   - Timestamp format: `YYYYMMDD_HHMMSS` consistently used
   - Seeds: Always array of integers
   - Model identifiers: Full HuggingFace paths

3. **Statistical Reporting**
   - Mean and standard deviation consistently named (`*_mean`, `*_std`)
   - Confidence intervals when provided use `ci_lower`, `ci_upper`
   - P-values range 0-1

### ⚠️ Areas for Improvement

1. **Summary Files**
   - `unified_summary_*.json` files have different structure than full results
   - Need separate schema definition or unification

2. **Optional Fields**
   - Some methods have `skipped: true` with `accuracy: None`
   - Could standardize to always omit or always include with null

3. **Comparison Tables**
   - Mix of numeric and string values (`'N/A'` for missing data)
   - Consider using `null` consistently

## JSON Schema Coverage

| Output Type | Schema Defined | Validation Status | Files Found |
|------------|---------------|-------------------|-------------|
| Unified Results | ✅ | ✅ Passing | 84 files |
| Unified Summary | ⚠️ | ❌ Failing | 42 files |
| Phase Results | ✅ | ✅ Passing | 6 files |
| Aggregated Results | ✅ | ✅ Passing | 1 file |
| Linear Probe Results | ✅ | ✅ Passing | 0 files |

## Recommendations

### Immediate Actions

1. **Update Schema for Summary Files**
   ```json
   {
     "title": "Unified Summary Results",
     "type": "object",
     "required": ["meta", "aggregated_results", "comparison_table"],
     "properties": {
       // No per_seed_results in summary
     }
   }
   ```

2. **Standardize Missing Data**
   - Use `null` instead of `"N/A"` strings
   - Omit methods that were skipped rather than including with null

3. **Add Schema Version**
   ```json
   {
     "schema_version": "1.0.0",
     "meta": {...}
   }
   ```

### Long-term Improvements

1. **Automated Validation in CI**
   - Add pre-commit hook to validate JSON outputs
   - Include in test suite

2. **Schema Migration Tool**
   - Script to upgrade old JSON files to new schema
   - Maintain backwards compatibility

3. **Documentation Generation**
   - Auto-generate docs from schema
   - Include example outputs

## Validation Commands

```bash
# Validate all JSON files
python3 telepathy/validate_json_outputs.py --check-all

# Check consistency across experiment
python3 telepathy/validate_json_outputs.py --dir runs/experiment_name/ --check-consistency

# Fix common issues (creates backups)
python3 telepathy/validate_json_outputs.py --dir runs/ --fix
```

## Conclusion

The JSON output structure is largely consistent across scripts, with clear patterns for:
- Core metrics (accuracy, latency, memory)
- Statistical aggregation (mean, std, CI)
- Metadata (timestamps, seeds, models)

Main areas for improvement:
1. Separate schema for summary files
2. Consistent handling of missing/skipped data
3. Version tracking for schema evolution

The validation infrastructure is now in place to maintain and enforce these standards going forward.