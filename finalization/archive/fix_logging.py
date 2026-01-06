#!/usr/bin/env python3
"""
Fix logging in all scripts to ensure proper output capture with tee and flush=True.

This script:
1. Checks all shell scripts for proper tee logging with timestamps
2. Adds flush=True to critical Python print statements
3. Ensures all scripts follow best practices for logging
"""

import os
import re
import sys
from pathlib import Path

def check_shell_script_logging(file_path):
    """Check if a shell script has proper logging with tee."""
    issues = []

    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    # Check for tee usage
    has_tee = 'tee' in content
    has_log_file = 'LOG_FILE' in content or 'log_file' in content.lower()
    has_timestamp = 'date +' in content or 'timestamp' in content.lower()

    if not has_tee:
        issues.append(f"Missing 'tee' for output capture")

    if not has_log_file:
        issues.append(f"No LOG_FILE variable or log file handling")

    if not has_timestamp:
        issues.append(f"No timestamp in log file name")

    # Check for proper tee pattern: { command } 2>&1 | tee "$LOG_FILE"
    proper_tee_pattern = r'\{[^}]+\}\s*2>&1\s*\|\s*tee'
    if not re.search(proper_tee_pattern, content):
        # Also check for alternative patterns
        alt_pattern1 = r'2>&1\s*\|\s*tee'
        alt_pattern2 = r'\|\s*tee\s+-a'
        if not (re.search(alt_pattern1, content) or re.search(alt_pattern2, content)):
            issues.append(f"Missing proper stderr+stdout capture (2>&1 | tee)")

    return issues

def fix_shell_script_logging(file_path):
    """Fix logging in a shell script by adding proper tee setup."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if already has proper logging
    content = ''.join(lines)
    if 'tee' in content and 'LOG_FILE' in content:
        return False  # Already has logging

    # Find the right place to insert logging setup (after shebang and initial comments)
    insert_index = 0
    for i, line in enumerate(lines):
        if line.startswith('#!'):
            insert_index = i + 1
        elif not line.startswith('#') and line.strip():
            insert_index = i
            break

    # Prepare logging setup
    script_name = file_path.stem
    logging_setup = f'''
# =============================================================================
# LOGGING SETUP
# =============================================================================

# Ensure output directory exists
OUTPUT_DIR="${{OUTPUT_DIR:-runs/{script_name}}}"
mkdir -p "$OUTPUT_DIR"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/{script_name}_${{TIMESTAMP}}.log"

echo "Starting {script_name} at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wrapper function for logging commands
run_with_logging() {{
    echo "Running: $*" | tee -a "$LOG_FILE"
    {{ "$@"; }} 2>&1 | tee -a "$LOG_FILE"
    return ${{PIPESTATUS[0]}}
}}

'''

    # Insert logging setup
    lines.insert(insert_index, logging_setup)

    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)

    return True

def check_python_script_logging(file_path):
    """Check if Python script has proper logging with flush=True."""
    issues = []

    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    # Check for print statements without flush=True
    critical_patterns = [
        r'print\s*\([^)]*["\']Starting',
        r'print\s*\([^)]*["\']Error',
        r'print\s*\([^)]*["\']Warning',
        r'print\s*\([^)]*["\']Complete',
        r'print\s*\([^)]*["\']Training',
        r'print\s*\([^)]*["\']Epoch',
        r'print\s*\([^)]*["\']Loss',
        r'print\s*\([^)]*["\']Checkpoint',
    ]

    for i, line in enumerate(lines, 1):
        for pattern in critical_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if 'flush=True' not in line:
                    issues.append(f"Line {i}: Critical print without flush=True: {line.strip()[:50]}...")

    # Check if using logging module
    if 'import logging' not in content:
        issues.append("Not using logging module (consider using structured logging)")

    return issues

def fix_python_script_logging(file_path):
    """Add flush=True to critical print statements in Python script."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    modified = False
    critical_keywords = [
        'Starting', 'Error', 'Warning', 'Complete', 'Finished',
        'Training', 'Epoch', 'Loss', 'Checkpoint', 'Saving',
        'Loading', 'Resuming', 'Batch', 'Step', 'Iteration'
    ]

    for i, line in enumerate(lines):
        # Check if line has a print statement
        if 'print(' in line:
            # Check if it contains critical keywords
            has_critical = any(keyword.lower() in line.lower() for keyword in critical_keywords)

            if has_critical and 'flush=True' not in line:
                # Add flush=True to the print statement
                # Handle various cases
                if line.rstrip().endswith(')'):
                    # Simple case: print(...)
                    lines[i] = line.rstrip()[:-1] + ', flush=True)\n'
                    modified = True
                elif line.rstrip().endswith(','):
                    # Case: print(...,
                    lines[i] = line.rstrip() + ' flush=True)\n'
                    modified = True

    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)

    return modified

def add_logging_module_to_python(file_path):
    """Add proper logging setup to Python script."""
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    # Check if already has logging
    if 'import logging' in content:
        return False

    # Find where to insert imports (after other imports)
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
        elif import_end > 0 and line.strip() and not line.startswith('#'):
            break

    # Add logging setup
    logging_setup = '''import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
'''

    lines.insert(import_end, logging_setup)

    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

    return True

def main():
    """Main function to check and fix logging in all scripts."""

    # Get the finalization directory
    finalization_dir = Path(__file__).parent

    print("=" * 60, flush=True)
    print("CHECKING AND FIXING LOGGING IN ALL SCRIPTS", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Check shell scripts
    shell_scripts = list(finalization_dir.glob("**/*.sh"))
    print(f"Found {len(shell_scripts)} shell scripts", flush=True)
    print("-" * 60, flush=True)

    shell_issues = []
    for script in shell_scripts:
        issues = check_shell_script_logging(script)
        if issues:
            shell_issues.append((script, issues))
            print(f"❌ {script.name}:", flush=True)
            for issue in issues:
                print(f"   - {issue}", flush=True)
        else:
            print(f"✓ {script.name}: OK", flush=True)

    print(flush=True)

    # Check Python scripts
    python_scripts = list(finalization_dir.glob("**/*.py"))
    # Exclude this script
    python_scripts = [s for s in python_scripts if s.name != 'fix_logging.py']

    print(f"Found {len(python_scripts)} Python scripts", flush=True)
    print("-" * 60, flush=True)

    python_issues = []
    for script in python_scripts:
        issues = check_python_script_logging(script)
        if issues:
            python_issues.append((script, issues))
            if len(issues) > 3:
                print(f"⚠️ {script.name}: {len(issues)} issues", flush=True)
            else:
                print(f"⚠️ {script.name}:", flush=True)
                for issue in issues[:3]:
                    print(f"   - {issue}", flush=True)
        else:
            print(f"✓ {script.name}: OK", flush=True)

    print(flush=True)
    print("=" * 60, flush=True)
    print("FIX SUMMARY", flush=True)
    print("=" * 60, flush=True)

    # Ask to fix issues
    if shell_issues:
        print(f"\nFound {len(shell_issues)} shell scripts with logging issues.", flush=True)
        response = input("Fix shell script logging? (y/n): ")
        if response.lower() == 'y':
            for script, _ in shell_issues:
                if fix_shell_script_logging(script):
                    print(f"  Fixed: {script.name}", flush=True)
                else:
                    print(f"  Already OK: {script.name}", flush=True)

    if python_issues:
        print(f"\nFound {len(python_issues)} Python scripts with potential logging issues.", flush=True)
        response = input("Add flush=True to critical prints? (y/n): ")
        if response.lower() == 'y':
            fixed_count = 0
            for script, _ in python_issues:
                if fix_python_script_logging(script):
                    fixed_count += 1
                    print(f"  Fixed: {script.name}", flush=True)
            print(f"Fixed {fixed_count} Python scripts", flush=True)

    print(flush=True)
    print("✅ Logging check complete!", flush=True)

    # Generate report
    report_file = finalization_dir / "logging_report.md"
    with open(report_file, 'w') as f:
        f.write("# Logging Infrastructure Report\n\n")
        f.write(f"Generated: {Path(__file__).name}\n")
        f.write(f"Date: $(date)\n\n")

        f.write("## Shell Scripts\n\n")
        f.write(f"Total: {len(shell_scripts)}\n")
        f.write(f"Issues: {len(shell_issues)}\n\n")

        if shell_issues:
            f.write("### Scripts with issues:\n\n")
            for script, issues in shell_issues:
                f.write(f"- **{script.name}**\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")

        f.write("\n## Python Scripts\n\n")
        f.write(f"Total: {len(python_scripts)}\n")
        f.write(f"Issues: {len(python_issues)}\n\n")

        if python_issues:
            f.write("### Scripts with logging improvements needed:\n\n")
            for script, issues in python_issues[:10]:  # Limit to first 10
                f.write(f"- **{script.name}**: {len(issues)} issues\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. All shell scripts should use `tee` with timestamped log files\n")
        f.write("2. Python scripts should use `flush=True` for critical output\n")
        f.write("3. Consider using structured logging (logging module) for complex Python scripts\n")
        f.write("4. Ensure all training scripts save logs to persistent storage\n")

    print(f"\n📄 Report saved to: {report_file}", flush=True)

if __name__ == "__main__":
    main()