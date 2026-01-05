#!/usr/bin/env python3
"""
Verify that logging infrastructure is properly configured across all scripts.
This script checks that fixes have been applied and are working correctly.
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime

def verify_shell_script(file_path):
    """Verify a shell script has proper logging."""
    with open(file_path, 'r') as f:
        content = f.read()

    checks = {
        'has_tee': 'tee' in content,
        'has_log_file': 'LOG_FILE' in content or 'log_file' in content.lower(),
        'has_timestamp': 'TIMESTAMP' in content or 'date +' in content,
        'has_stderr_redirect': '2>&1' in content,
        'has_proper_tee_pattern': bool(re.search(r'2>&1\s*\|\s*tee', content)),
    }

    return checks

def verify_python_script(file_path):
    """Verify Python script has proper logging."""
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.splitlines()

    critical_prints_with_flush = 0
    critical_prints_without_flush = 0

    critical_patterns = [
        r'print\s*\([^)]*["\'](?:Starting|Error|Warning|Complete|Training|Epoch|Loss|Checkpoint)',
    ]

    for line in lines:
        for pattern in critical_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if 'flush=True' in line:
                    critical_prints_with_flush += 1
                else:
                    critical_prints_without_flush += 1

    checks = {
        'has_logging_import': 'import logging' in content,
        'critical_prints_with_flush': critical_prints_with_flush,
        'critical_prints_without_flush': critical_prints_without_flush,
        'uses_print': 'print(' in content,
    }

    return checks

def main():
    """Main verification function."""

    print("=" * 70, flush=True)
    print("VERIFYING LOGGING INFRASTRUCTURE", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    finalization_dir = Path(__file__).parent

    # Verify shell scripts
    shell_scripts = list(finalization_dir.glob("**/*.sh"))
    shell_results = []

    print(f"Checking {len(shell_scripts)} shell scripts...", flush=True)
    print("-" * 70, flush=True)

    for script in shell_scripts:
        checks = verify_shell_script(script)
        shell_results.append({
            'script': script.name,
            'checks': checks,
            'passed': all([
                checks['has_tee'],
                checks['has_log_file'] or checks['has_timestamp'],
                checks['has_stderr_redirect']
            ])
        })

        status = "✅" if shell_results[-1]['passed'] else "❌"
        print(f"{status} {script.name:30s} ", end="", flush=True)

        if not shell_results[-1]['passed']:
            missing = []
            if not checks['has_tee']:
                missing.append("tee")
            if not (checks['has_log_file'] or checks['has_timestamp']):
                missing.append("logging setup")
            if not checks['has_stderr_redirect']:
                missing.append("stderr capture")
            print(f"Missing: {', '.join(missing)}", flush=True)
        else:
            print("OK", flush=True)

    print(flush=True)

    # Verify Python scripts
    python_scripts = list(finalization_dir.glob("**/*.py"))
    # Exclude this script and the fix script
    python_scripts = [s for s in python_scripts if s.name not in ['verify_logging.py', 'fix_logging.py']]

    python_results = []

    print(f"Checking {len(python_scripts)} Python scripts...", flush=True)
    print("-" * 70, flush=True)

    critical_scripts = []
    for script in python_scripts:
        checks = verify_python_script(script)

        # Determine if this is a critical script based on name
        is_critical = any(keyword in script.name.lower() for keyword in
                         ['train', 'eval', 'experiment', 'run', 'test', 'checkpoint'])

        if is_critical and checks['critical_prints_without_flush'] > 0:
            critical_scripts.append({
                'script': script.name,
                'without_flush': checks['critical_prints_without_flush'],
                'with_flush': checks['critical_prints_with_flush']
            })

        python_results.append({
            'script': script.name,
            'checks': checks,
            'is_critical': is_critical
        })

    # Show only critical scripts with issues
    if critical_scripts:
        print("Critical scripts needing attention:", flush=True)
        for item in critical_scripts[:10]:  # Show first 10
            print(f"  ⚠️ {item['script']}: {item['without_flush']} prints without flush", flush=True)
    else:
        print("✅ All critical scripts have proper flush=True", flush=True)

    print(flush=True)

    # Summary
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    shell_passed = sum(1 for r in shell_results if r['passed'])
    print(f"Shell Scripts:  {shell_passed}/{len(shell_results)} passed", flush=True)

    critical_ok = len(critical_scripts) == 0
    status = "✅" if critical_ok else "⚠️"
    print(f"Python Scripts: {status} {len(critical_scripts)} critical scripts need attention", flush=True)

    print(flush=True)

    # Save verification results
    results = {
        'timestamp': datetime.now().isoformat(),
        'shell_scripts': {
            'total': len(shell_results),
            'passed': shell_passed,
            'details': shell_results
        },
        'python_scripts': {
            'total': len(python_results),
            'critical_issues': len(critical_scripts),
            'critical_details': critical_scripts
        }
    }

    results_file = finalization_dir / 'logging_verification.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📊 Detailed results saved to: {results_file}", flush=True)

    # Overall status
    overall_good = shell_passed == len(shell_results) and len(critical_scripts) == 0

    if overall_good:
        print(flush=True)
        print("✅ VERIFICATION PASSED - All scripts have proper logging!", flush=True)
        return 0
    else:
        print(flush=True)
        print("⚠️ VERIFICATION INCOMPLETE - Some scripts need additional fixes", flush=True)
        print("  Run 'python3 fix_logging.py' to apply additional fixes", flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())