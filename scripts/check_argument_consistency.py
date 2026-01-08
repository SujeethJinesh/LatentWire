#!/usr/bin/env python3
"""
Check argument consistency between train.py, eval.py and all scripts.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

def extract_arguments_from_py(file_path: Path) -> Set[str]:
    """Extract all --arguments defined in a Python file."""
    arguments = set()
    if not file_path.exists():
        return arguments

    content = file_path.read_text()
    # Find all add_argument calls with --
    pattern = r'add_argument\s*\(\s*["\'](--.+?)["\']'
    matches = re.findall(pattern, content)
    arguments.update(matches)

    return arguments

def extract_arguments_from_script(file_path: Path) -> Dict[str, List[int]]:
    """Extract all --arguments used in a shell script with line numbers."""
    arguments = {}
    if not file_path.exists():
        return arguments

    lines = file_path.read_text().split('\n')
    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        # Find all --arguments
        pattern = r'--[a-zA-Z_][a-zA-Z0-9_\-]*'
        matches = re.findall(pattern, line)
        for match in matches:
            if match not in arguments:
                arguments[match] = []
            arguments[match].append(i)

    return arguments

def check_script_arguments(script_path: Path, train_args: Set[str], eval_args: Set[str],
                          linear_probe_args: Set[str]) -> List[str]:
    """Check if a script uses valid arguments."""
    issues = []
    script_args = extract_arguments_from_script(script_path)

    if not script_args:
        return issues

    # Try to determine which Python script is being called
    content = script_path.read_text()
    lines = content.split('\n')

    for arg, line_nums in script_args.items():
        # Skip shell-specific arguments
        if arg in ['--help', '--version', '--account', '--gpus', '--job-name',
                   '--error', '--output', '--nodes', '--time', '--partition',
                   '--checkpoint']:  # --checkpoint might be a directory arg
            continue

        for line_num in line_nums:
            if line_num > 0 and line_num <= len(lines):
                line = lines[line_num - 1]

                # Determine which script is being called
                if 'train.py' in line:
                    if arg not in train_args:
                        issues.append(f"Line {line_num}: {arg} not defined in train.py")
                elif 'eval.py' in line:
                    if arg not in eval_args:
                        issues.append(f"Line {line_num}: {arg} not defined in eval.py")
                elif 'linear_probe_baseline.py' in line:
                    if arg not in linear_probe_args:
                        issues.append(f"Line {line_num}: {arg} not defined in linear_probe_baseline.py")

    return issues

def main():
    """Main consistency check."""
    base_dir = Path('/Users/sujeethjinesh/Desktop/LatentWire')

    # Extract arguments from Python files
    print("Extracting arguments from Python files...")
    train_args = extract_arguments_from_py(base_dir / 'latentwire' / 'train.py')
    eval_args = extract_arguments_from_py(base_dir / 'latentwire' / 'eval.py')
    linear_probe_args = extract_arguments_from_py(base_dir / 'latentwire' / 'linear_probe_baseline.py')

    print(f"Found {len(train_args)} arguments in train.py")
    print(f"Found {len(eval_args)} arguments in eval.py")
    print(f"Found {len(linear_probe_args)} arguments in linear_probe_baseline.py")

    # Check all shell scripts
    print("\nChecking shell scripts for consistency...")

    all_issues = []

    # Check main directories
    for dir_path in [base_dir / 'scripts', base_dir / 'telepathy', base_dir / 'finalization']:
        if not dir_path.exists():
            continue

        for script in dir_path.glob('**/*.sh'):
            issues = check_script_arguments(script, train_args, eval_args, linear_probe_args)
            if issues:
                all_issues.append((script, issues))

        for script in dir_path.glob('**/*.slurm'):
            issues = check_script_arguments(script, train_args, eval_args, linear_probe_args)
            if issues:
                all_issues.append((script, issues))

    # Report findings
    if not all_issues:
        print("\n✅ All scripts use valid arguments!")
    else:
        print(f"\n❌ Found issues in {len(all_issues)} scripts:\n")
        for script_path, issues in all_issues:
            print(f"\n{script_path.relative_to(base_dir)}:")
            for issue in issues[:5]:  # Show first 5 issues per script
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more issues")

    # Check for specific known issues
    print("\n" + "="*60)
    print("Checking specific argument patterns...")
    print("="*60)

    # Check anchor text usage
    print("\n1. Anchor text arguments:")
    print("   train.py should use: --warm_anchor_text")
    print("   eval.py should use: --latent_anchor_text")

    for dir_path in [base_dir / 'scripts', base_dir / 'telepathy', base_dir / 'finalization']:
        if not dir_path.exists():
            continue

        for script in list(dir_path.glob('**/*.sh')) + list(dir_path.glob('**/*.slurm')):
            content = script.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                if 'train.py' in line:
                    # Check next few lines for anchor arguments
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if '--latent_anchor_text' in lines[j] and 'eval.py' not in lines[j]:
                            print(f"   ⚠️  {script.relative_to(base_dir)}:{j+1} uses --latent_anchor_text with train.py (should be --warm_anchor_text)")
                        elif '--warm_anchor_text' in lines[j]:
                            pass  # Correct usage

                elif 'eval.py' in line:
                    # Check next few lines for anchor arguments
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if '--warm_anchor_text' in lines[j]:
                            print(f"   ⚠️  {script.relative_to(base_dir)}:{j+1} uses --warm_anchor_text with eval.py (should be --latent_anchor_text)")

    print("\n✅ Consistency check complete!")

if __name__ == "__main__":
    main()