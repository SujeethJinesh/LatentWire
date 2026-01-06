#!/usr/bin/env python3
"""
Fix logging buffering issues to prevent log loss during SLURM preemption.

This script:
1. Adds flush=True to critical print statements in Python files
2. Ensures PYTHONUNBUFFERED=1 is set in shell scripts
3. Verifies tee usage for proper log capture
"""

import os
import re
import sys
from pathlib import Path

def find_critical_print_statements(file_path):
    """Find print statements that should have flush=True."""
    critical_patterns = [
        r'epoch.*\d+',  # Epoch progress
        r'step.*\d+',   # Step progress
        r'loss.*:',     # Loss values
        r'accuracy',    # Accuracy metrics
        r'f1.*score',   # F1 scores
        r'checkpoint',  # Checkpoint saves
        r'saved',       # Save operations
        r'complete',    # Completion messages
        r'error',       # Error messages
        r'warning',     # Warning messages
        r'starting',    # Start messages
        r'finished',    # Finish messages
        r'results?',    # Results
        r'evaluation',  # Evaluation progress
        r'training',    # Training progress
    ]

    critical_lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Check if line contains print statement
        if 'print(' in line:
            # Skip if already has flush=True
            if 'flush=True' in line:
                continue

            # Check if it's a critical print
            line_lower = line.lower()
            for pattern in critical_patterns:
                if re.search(pattern, line_lower):
                    critical_lines.append((i + 1, line.strip()))
                    break

    return critical_lines

def add_flush_to_prints(file_path, dry_run=False):
    """Add flush=True to critical print statements."""
    critical_lines = find_critical_print_statements(file_path)

    if not critical_lines:
        return 0

    if dry_run:
        print(f"\n{file_path}:")
        for line_num, line in critical_lines:
            print(f"  Line {line_num}: {line[:80]}...")
        return len(critical_lines)

    # Read file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Modify critical lines
    modified = 0
    for line_num, _ in critical_lines:
        idx = line_num - 1
        line = lines[idx]

        # Find the closing parenthesis of the print statement
        # Handle multi-line prints
        if line.rstrip().endswith(')'):
            # Simple case: print ends on same line
            lines[idx] = line.rstrip()[:-1] + ', flush=True)\n'
            modified += 1
        else:
            # Multi-line print - find the closing parenthesis
            for j in range(idx + 1, min(idx + 10, len(lines))):
                if ')' in lines[j]:
                    # Add flush=True before the closing parenthesis
                    lines[j] = lines[j].replace(')', ', flush=True)', 1)
                    modified += 1
                    break

    # Write back
    if modified > 0:
        with open(file_path, 'w') as f:
            f.writelines(lines)

    return modified

def check_shell_script(file_path):
    """Check shell script for proper logging setup."""
    issues = {
        'no_pythonunbuffered': False,
        'no_tee': False,
        'no_pipefail': False,
    }

    with open(file_path, 'r') as f:
        content = f.read()

    # Check for PYTHONUNBUFFERED
    if 'PYTHONUNBUFFERED' not in content:
        issues['no_pythonunbuffered'] = True

    # Check for tee usage
    if '| tee' not in content and file_path.name != 'test_*.sh':
        issues['no_tee'] = True

    # Check for pipefail
    if 'pipefail' not in content:
        issues['no_pipefail'] = True

    return issues

def fix_shell_script(file_path, dry_run=False):
    """Fix shell script logging issues."""
    issues = check_shell_script(file_path)

    if not any(issues.values()):
        return False

    if dry_run:
        print(f"\n{file_path}:")
        if issues['no_pythonunbuffered']:
            print("  - Missing PYTHONUNBUFFERED=1")
        if issues['no_tee']:
            print("  - Missing tee for logging")
        if issues['no_pipefail']:
            print("  - Missing set -o pipefail")
        return True

    with open(file_path, 'r') as f:
        lines = f.readlines()

    modified = False

    # Add PYTHONUNBUFFERED after shebang and initial comments
    if issues['no_pythonunbuffered']:
        # Find where to insert (after initial comments)
        insert_idx = 1
        for i, line in enumerate(lines[1:], 1):
            if not line.startswith('#') and line.strip():
                insert_idx = i
                break

        # Check if there's already an export section
        export_found = False
        for i in range(insert_idx, min(insert_idx + 10, len(lines))):
            if 'export' in lines[i]:
                # Add after existing exports
                lines.insert(i + 1, 'export PYTHONUNBUFFERED=1\n')
                modified = True
                export_found = True
                break

        if not export_found:
            lines.insert(insert_idx, '\n# Ensure unbuffered output for logging\nexport PYTHONUNBUFFERED=1\n')
            modified = True

    # Add pipefail after set commands
    if issues['no_pipefail']:
        for i, line in enumerate(lines):
            if line.startswith('set '):
                if 'pipefail' not in line:
                    lines.insert(i + 1, 'set -o pipefail  # Ensure pipe failures are detected\n')
                    modified = True
                    break

    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)

    return modified

def main():
    """Main function to fix logging issues."""
    # Parse arguments
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    # Find relevant files
    project_root = Path(__file__).parent.parent.parent

    # Python files to check
    python_files = [
        'latentwire/train.py',
        'latentwire/eval.py',
        'latentwire/eval_sst2.py',
        'latentwire/eval_agnews.py',
        'telepathy/eval_telepathy_trec.py',
        'scripts/statistical_testing.py',
    ]

    # Shell scripts to check (exclude test scripts)
    shell_scripts = []
    for pattern in ['*.sh', 'scripts/*.sh', 'telepathy/*.sh']:
        for file in project_root.glob(pattern):
            if not file.name.startswith('test_'):
                shell_scripts.append(file.relative_to(project_root))

    print("=" * 60)
    print("CHECKING PYTHON FILES FOR MISSING flush=True")
    print("=" * 60)

    total_python_fixes = 0
    for py_file in python_files:
        file_path = project_root / py_file
        if file_path.exists():
            fixes = add_flush_to_prints(file_path, dry_run)
            if fixes > 0:
                total_python_fixes += fixes
                if not dry_run:
                    print(f"Fixed {fixes} print statements in {py_file}")

    if total_python_fixes == 0:
        print("No critical print statements need flush=True")
    elif not dry_run:
        print(f"\nTotal: Fixed {total_python_fixes} print statements")

    print("\n" + "=" * 60)
    print("CHECKING SHELL SCRIPTS FOR LOGGING ISSUES")
    print("=" * 60)

    total_shell_fixes = 0
    for sh_file in shell_scripts:
        file_path = project_root / sh_file
        if file_path.exists():
            if fix_shell_script(file_path, dry_run):
                total_shell_fixes += 1
                if not dry_run:
                    print(f"Fixed {sh_file}")

    if total_shell_fixes == 0:
        print("No shell scripts need fixes")
    elif not dry_run:
        print(f"\nTotal: Fixed {total_shell_fixes} shell scripts")

    # Special check for RUN_ALL.sh
    print("\n" + "=" * 60)
    print("VERIFYING RUN_ALL.sh")
    print("=" * 60)

    run_all = project_root / 'RUN_ALL.sh'
    if run_all.exists():
        with open(run_all, 'r') as f:
            content = f.read()

        checks = {
            'PYTHONUNBUFFERED=1': 'PYTHONUNBUFFERED=1' in content,
            'tee usage': '| tee' in content,
            'pipefail': 'pipefail' in content,
            'SLURM section has PYTHONUNBUFFERED': 'export PYTHONUNBUFFERED=1' in content[content.find('create_slurm_script'):] if 'create_slurm_script' in content else False
        }

        all_good = all(checks.values())
        if all_good:
            print("✅ RUN_ALL.sh has proper logging configuration")
        else:
            print("Issues found in RUN_ALL.sh:")
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if dry_run:
        print(f"Would fix {total_python_fixes} Python print statements")
        print(f"Would fix {total_shell_fixes} shell scripts")
        print("\nRun without --dry-run to apply fixes")
    else:
        print(f"Fixed {total_python_fixes} Python print statements")
        print(f"Fixed {total_shell_fixes} shell scripts")
        print("\n✅ Logging configuration updated to prevent buffer loss")
        print("\nRecommendations:")
        print("1. Always use flush=True for critical progress messages")
        print("2. Set PYTHONUNBUFFERED=1 in all execution scripts")
        print("3. Use tee to capture both stdout and stderr")
        print("4. Set pipefail to catch errors in pipelines")

if __name__ == '__main__':
    main()