#!/usr/bin/env python3
"""
Validate JSON outputs against the standardized schema.

This script checks all JSON outputs from experimental scripts to ensure
they conform to the standardized schema defined in json_schema.json.

Usage:
    python telepathy/validate_json_outputs.py --file path/to/results.json
    python telepathy/validate_json_outputs.py --dir runs/
    python telepathy/validate_json_outputs.py --check-all
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    print("jsonschema package not installed. Install with: pip install jsonschema")
    sys.exit(1)


def load_schema(schema_path="telepathy/json_schema.json"):
    """Load the JSON schema from file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_json_file(file_path, schema):
    """
    Validate a single JSON file against the schema.

    Returns:
        (is_valid, error_message)
    """
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Validate against schema
        validate(instance=data, schema=schema)
        return True, None

    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except ValidationError as e:
        # Get the best matching schema error
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: len(e.path))
        if errors:
            error = errors[0]
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            return False, f"Validation error at {path}: {error.message}"
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def find_json_files(directory, patterns=None):
    """Find all JSON result files in a directory."""
    if patterns is None:
        patterns = [
            "**/unified_results*.json",
            "**/unified_summary*.json",
            "**/aggregated_results.json",
            "**/phase*_results.json",
            "**/all_results.json",
            "**/linear_probe_results.json",
            "**/significance_tests.json"
        ]

    json_files = []
    for pattern in patterns:
        json_files.extend(directory.glob(pattern))

    return sorted(set(json_files))


def check_consistency(json_files):
    """
    Check for consistency issues across JSON files.

    Returns dict of issues found.
    """
    issues = {
        "missing_fields": [],
        "type_mismatches": [],
        "inconsistent_structures": []
    }

    # Load all files
    data_by_file = {}
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data_by_file[str(file_path)] = json.load(f)
        except Exception as e:
            issues["missing_fields"].append(f"{file_path}: Failed to load - {e}")

    # Check for common fields and types
    if len(data_by_file) > 1:
        # Get common top-level keys
        all_keys = [set(data.keys()) for data in data_by_file.values()]
        common_keys = set.intersection(*all_keys) if all_keys else set()

        # Check for files missing common keys
        for file_path, data in data_by_file.items():
            missing = common_keys - set(data.keys())
            if missing:
                issues["missing_fields"].append(
                    f"{Path(file_path).name}: Missing common fields {missing}"
                )

        # Check for type consistency
        for key in common_keys:
            types = set()
            for file_path, data in data_by_file.items():
                if key in data:
                    types.add(type(data[key]).__name__)

            if len(types) > 1:
                issues["type_mismatches"].append(
                    f"Field '{key}' has inconsistent types across files: {types}"
                )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate JSON outputs against schema")
    parser.add_argument("--file", type=str, help="Validate a single JSON file")
    parser.add_argument("--dir", type=str, help="Validate all JSON files in directory")
    parser.add_argument("--check-all", action="store_true",
                       help="Check all JSON files in runs/")
    parser.add_argument("--schema", type=str, default="telepathy/json_schema.json",
                       help="Path to JSON schema file")
    parser.add_argument("--check-consistency", action="store_true",
                       help="Check for consistency across files")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to fix common issues (experimental)")

    args = parser.parse_args()

    # Load schema
    try:
        schema = load_schema(args.schema)
    except Exception as e:
        print(f"Error loading schema: {e}")
        sys.exit(1)

    # Determine files to validate
    json_files = []

    if args.file:
        json_files = [Path(args.file)]
    elif args.dir:
        json_files = find_json_files(Path(args.dir))
    elif args.check_all:
        json_files = find_json_files(Path("runs"))
    else:
        parser.print_help()
        sys.exit(1)

    if not json_files:
        print("No JSON files found to validate")
        sys.exit(0)

    print(f"Found {len(json_files)} JSON files to validate")
    print("=" * 70)

    # Validate each file
    valid_count = 0
    invalid_count = 0
    errors = []

    for file_path in json_files:
        is_valid, error = validate_json_file(file_path, schema)

        try:
            display_path = file_path.relative_to(Path.cwd())
        except ValueError:
            display_path = file_path

        if is_valid:
            print(f"✓ {display_path}")
            valid_count += 1
        else:
            print(f"✗ {display_path}")
            print(f"  Error: {error}")
            invalid_count += 1
            errors.append((file_path, error))

    print("=" * 70)
    print(f"Summary: {valid_count} valid, {invalid_count} invalid")

    # Check consistency if requested
    if args.check_consistency and len(json_files) > 1:
        print("\nChecking consistency across files...")
        issues = check_consistency(json_files)

        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for issue in issue_list:
                    print(f"  - {issue}")

    # Provide fix suggestions
    if invalid_count > 0 and not args.fix:
        print("\nCommon fixes:")
        print("1. Ensure all percentage values are 0-100 (not 0-1)")
        print("2. Include all required fields (accuracy, correct, total)")
        print("3. Use correct timestamp format (YYYYMMDD_HHMMSS)")
        print("4. Ensure proper nesting of results by dataset and seed")
        print("\nRun with --fix to attempt automatic fixes (make backups first!)")

    # Attempt fixes if requested
    if args.fix and invalid_count > 0:
        print("\nAttempting to fix common issues...")
        fixed = 0

        for file_path, error in errors:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                modified = False

                # Fix percentage values (if they're 0-1, convert to 0-100)
                def fix_percentages(obj):
                    nonlocal modified
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if 'accuracy' in key and isinstance(value, (int, float)):
                                if 0 <= value <= 1:
                                    obj[key] = value * 100
                                    modified = True
                            elif isinstance(value, (dict, list)):
                                fix_percentages(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix_percentages(item)

                fix_percentages(data)

                # Add missing required fields
                if 'meta' in data and 'timestamp' not in data['meta']:
                    from datetime import datetime
                    data['meta']['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    modified = True

                if modified:
                    # Create backup
                    backup_path = file_path.with_suffix('.json.bak')
                    with open(backup_path, 'w') as f:
                        with open(file_path, 'r') as orig:
                            f.write(orig.read())

                    # Write fixed version
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)

                    print(f"Fixed: {file_path} (backup: {backup_path})")
                    fixed += 1

            except Exception as e:
                print(f"Could not fix {file_path}: {e}")

        print(f"\nFixed {fixed} files")

    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()