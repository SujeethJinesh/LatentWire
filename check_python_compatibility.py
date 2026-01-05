#!/usr/bin/env python3
"""
Check Python files for compatibility with Python 3.8+
Reports any usage of Python 3.10+ features that would fail on older versions.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def check_file(filepath: Path) -> List[str]:
    """Check a single Python file for compatibility issues."""
    issues = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for walrus operator := (3.8+)
        if ':=' in content:
            issues.append(f"Uses walrus operator := (requires Python 3.8+)")

        # Check for union types with | (3.10+)
        # Look for patterns like ": str | int" or "-> str | None"
        import re
        union_pattern = r':\s*[A-Za-z_]\w*\s*\|\s*[A-Za-z_]'
        if re.search(union_pattern, content):
            # Double check it's not in a string or comment
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(union_pattern, line):
                    # Simple check: not in a string (crude but effective for most cases)
                    if not (line.strip().startswith('#') or
                           line.strip().startswith('"""') or
                           line.strip().startswith("'''")):
                        issues.append(f"Line {i}: Possible union type with | (requires Python 3.10+)")

        # Parse AST to check for match/case statements (3.10+)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if sys.version_info >= (3, 10):
                    # Only check on Python 3.10+ where ast.Match exists
                    if hasattr(ast, 'Match') and isinstance(node, ast.Match):
                        issues.append(f"Uses match/case statement (requires Python 3.10+)")

        except SyntaxError as e:
            issues.append(f"Syntax error (might be version-related): {e}")

        # Check for parenthesized context managers (3.10+)
        if re.search(r'with\s*\(.*\):', content):
            issues.append("Uses parenthesized context managers (requires Python 3.10+)")

    except Exception as e:
        issues.append(f"Error reading file: {e}")

    return issues


def check_directory(directory: Path) -> Dict[str, List[str]]:
    """Check all Python files in a directory."""
    issues_by_file = {}

    for py_file in directory.rglob('*.py'):
        # Skip virtual environments and cache
        if any(part in py_file.parts for part in ['venv', '__pycache__', '.git']):
            continue

        issues = check_file(py_file)
        if issues:
            issues_by_file[str(py_file.relative_to(directory))] = issues

    return issues_by_file


def main():
    """Check specific directories for Python compatibility."""
    base_dir = Path('/Users/sujeethjinesh/Desktop/LatentWire')

    # Directories to check
    dirs_to_check = [
        'telepathy',
        'latentwire',
        'scripts'
    ]

    print("Python Compatibility Check")
    print("==========================")
    print("Checking for Python 3.10+ features that would fail on Python 3.8...")
    print()

    total_issues = 0

    for dir_name in dirs_to_check:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            continue

        print(f"\nChecking {dir_name}/...")
        issues = check_directory(dir_path)

        if issues:
            print(f"  Found {len(issues)} files with potential issues:")
            for filepath, file_issues in issues.items():
                print(f"\n  {filepath}:")
                for issue in file_issues:
                    print(f"    - {issue}")
                total_issues += len(file_issues)
        else:
            print("  ✓ No compatibility issues found")

    print(f"\n{'='*60}")
    if total_issues == 0:
        print("✓ All files are compatible with Python 3.8+")
    else:
        print(f"⚠ Found {total_issues} potential compatibility issues")
        print("  These would need to be fixed before running on Python 3.8")

    return total_issues


if __name__ == '__main__':
    sys.exit(main())