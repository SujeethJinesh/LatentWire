#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate all file paths in the LatentWire codebase.
Ensures paths work on both local Mac and HPC cluster environments.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

def check_file_for_path_issues(filepath):
    """Check a single file for path-related issues."""
    issues = []

    try:
        content = filepath.read_text()
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Check for hardcoded /home/ paths (will fail on HPC)
            if '/home/' in line and not line.strip().startswith('#'):
                issues.append({
                    'file': str(filepath),
                    'line': line_num,
                    'issue': 'Hardcoded /home/ path - will fail on HPC',
                    'content': line.strip()
                })

            # Check for hardcoded absolute paths without using Path()
            if filepath.suffix == '.py':
                # Check for string literals with absolute paths
                abs_path_pattern = r'["\']/(projects|Users|tmp|var|usr|etc)/[^"\']*["\']'
                if re.search(abs_path_pattern, line) and 'Path(' not in line and not line.strip().startswith('#'):
                    if '/projects/m000066/sujinesh/LatentWire' not in line:  # This is our valid HPC path
                        issues.append({
                            'file': str(filepath),
                            'line': line_num,
                            'issue': 'Hardcoded absolute path without Path() - may not be portable',
                            'content': line.strip()
                        })

            # Check for missing directory creation before use
            if filepath.suffix == '.py':
                if 'open(' in line and 'exist_ok' not in line and 'makedirs' not in content[:content.find(line)]:
                    # Check if this might write to a directory that doesn't exist
                    if any(mode in line for mode in ['w', 'a', 'x']):
                        issues.append({
                            'file': str(filepath),
                            'line': line_num,
                            'issue': 'File write without checking directory exists',
                            'content': line.strip()
                        })

            # Check SLURM files for correct paths
            if filepath.suffix == '.slurm':
                # Check for incorrect account
                if '--account=' in line and '--account=marlowe-m000066' not in line:
                    issues.append({
                        'file': str(filepath),
                        'line': line_num,
                        'issue': 'Incorrect SLURM account (should be marlowe-m000066)',
                        'content': line.strip()
                    })

                # Check for incorrect partition
                if '--partition=' in line and '--partition=preempt' not in line:
                    issues.append({
                        'file': str(filepath),
                        'line': line_num,
                        'issue': 'Incorrect SLURM partition (should be preempt)',
                        'content': line.strip()
                    })

                # Check for /home paths in SLURM (should use /projects)
                if '/home/sjinesh' in line:
                    issues.append({
                        'file': str(filepath),
                        'line': line_num,
                        'issue': 'SLURM using /home path instead of /projects',
                        'content': line.strip()
                    })

    except Exception as e:
        issues.append({
            'file': str(filepath),
            'line': 0,
            'issue': 'Error reading file: {}'.format(e),
            'content': ''
        })

    return issues

def validate_directory_creation(filepath):
    """Check if directories are properly created before use."""
    issues = []

    if filepath.suffix in ['.py', '.sh']:
        try:
            content = filepath.read_text()

            # Check Python files
            if filepath.suffix == '.py':
                # Look for patterns that suggest directory usage without creation
                if 'save_dir' in content or 'output_dir' in content:
                    if 'makedirs' not in content and 'mkdir' not in content and 'Path(' not in content:
                        issues.append({
                            'file': str(filepath),
                            'line': 0,
                            'issue': 'Uses save_dir/output_dir but no directory creation found',
                            'content': 'Missing os.makedirs() or Path().mkdir()'
                        })

            # Check shell scripts
            elif filepath.suffix == '.sh':
                if 'OUTPUT_DIR=' in content or 'output_dir=' in content:
                    if 'mkdir -p' not in content:
                        issues.append({
                            'file': str(filepath),
                            'line': 0,
                            'issue': 'Sets OUTPUT_DIR but no mkdir -p found',
                            'content': 'Missing mkdir -p command'
                        })

        except Exception as e:
            pass

    return issues

def main():
    """Main validation function."""
    print("=" * 80)
    print("PATH VALIDATION REPORT FOR LATENTWIRE")
    print("=" * 80)
    print()

    # Find all relevant files
    root = Path('.')
    py_files = list(root.glob('**/*.py'))
    sh_files = list(root.glob('**/*.sh'))
    slurm_files = list(root.glob('**/*.slurm'))

    # Filter out unwanted directories
    all_files = []
    for f in py_files + sh_files + slurm_files:
        if not any(skip in str(f) for skip in ['.git', 'runs/', '__pycache__', '.pytest_cache']):
            all_files.append(f)

    print("Checking {} files...".format(len(all_files)))
    print()

    all_issues = []
    critical_issues = []
    warnings = []

    # Check each file
    for filepath in all_files:
        issues = check_file_for_path_issues(filepath)
        issues.extend(validate_directory_creation(filepath))

        for issue in issues:
            if any(word in issue['issue'] for word in ['SLURM', '/home/', 'will fail']):
                critical_issues.append(issue)
            else:
                warnings.append(issue)

        all_issues.extend(issues)

    # Report findings
    if critical_issues:
        print("🔴 CRITICAL ISSUES (will cause failures):")
        print("-" * 40)
        for issue in critical_issues[:10]:  # Show first 10
            print("  File: {}:{}".format(issue['file'], issue['line']))
            print("  Issue: {}".format(issue['issue']))
            if issue['content']:
                print("  Line: {}".format(issue['content'][:80]))
            print()
        if len(critical_issues) > 10:
            print("  ... and {} more critical issues".format(len(critical_issues) - 10))
        print()

    if warnings:
        print("⚠️  WARNINGS (potential issues):")
        print("-" * 40)
        for issue in warnings[:5]:  # Show first 5
            print("  File: {}:{}".format(issue['file'], issue['line']))
            print("  Issue: {}".format(issue['issue']))
            print()
        if len(warnings) > 5:
            print("  ... and {} more warnings".format(len(warnings) - 5))
        print()

    # Check specific important patterns
    print("📋 PATH PATTERNS SUMMARY:")
    print("-" * 40)

    # Count Path() usage
    path_usage_count = 0
    os_path_usage_count = 0
    for f in py_files:
        if any(skip in str(f) for skip in ['.git', 'runs/', '__pycache__']):
            continue
        try:
            content = f.read_text()
            if 'from pathlib import Path' in content or 'import pathlib' in content:
                path_usage_count += 1
            if 'os.path.' in content:
                os_path_usage_count += 1
        except:
            pass

    print("  Python files using pathlib.Path: {}".format(path_usage_count))
    print("  Python files using os.path: {}".format(os_path_usage_count))
    print()

    # Check for proper runs/ directory usage
    runs_usage = []
    for f in all_files:
        try:
            content = f.read_text()
            if 'runs/' in content:
                runs_usage.append(str(f))
        except:
            pass

    print("  Files referencing 'runs/' directory: {}".format(len(runs_usage)))
    print()

    # Final summary
    print("=" * 80)
    if critical_issues:
        print("❌ FAILED: Found {} critical path issues that must be fixed!".format(len(critical_issues)))
        print("   These will cause failures when running on HPC.")

        # Save report
        report = {
            'critical_issues': critical_issues,
            'warnings': warnings,
            'summary': {
                'total_files_checked': len(all_files),
                'critical_count': len(critical_issues),
                'warning_count': len(warnings)
            }
        }

        report_path = Path('runs/path_validation_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print("\n   Full report saved to: {}".format(report_path))

        return 1
    else:
        print("✅ PASSED: No critical path issues found!")
        print("   Found {} minor warnings that should be reviewed.".format(len(warnings)))
        return 0

if __name__ == '__main__':
    sys.exit(main())