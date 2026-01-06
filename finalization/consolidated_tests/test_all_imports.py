#!/usr/bin/env python3
"""
Test script to verify all Python files in finalization can import correctly.
This will identify any import errors and help fix them.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

# Add the finalization directory to the Python path
FINALIZATION_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(FINALIZATION_DIR))
sys.path.insert(0, str(FINALIZATION_DIR.parent))  # For latentwire imports

def test_import(file_path, base_dir):
    """Test importing a Python file and return any errors."""
    try:
        # Get the relative path from base_dir
        rel_path = Path(file_path).relative_to(base_dir)

        # Convert path to module name (e.g., latentwire/models.py -> latentwire.models)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_name = '.'.join(module_parts)

        # Skip __pycache__ files
        if '__pycache__' in module_name:
            return None, None

        # Skip test files themselves to avoid circular imports
        if module_name.startswith('test_'):
            return None, None

        print(f"Testing import: {module_name} from {file_path}")

        # Try to import the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module_name, None
        else:
            return module_name, f"Could not create spec for {file_path}"

    except Exception as e:
        error_msg = f"Error importing {file_path}:\n{str(e)}\n{traceback.format_exc()}"
        return module_name if 'module_name' in locals() else str(file_path), error_msg

def find_python_files(directory):
    """Recursively find all Python files in a directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)

def main():
    print(f"Testing imports for all Python files in: {FINALIZATION_DIR}")
    print("=" * 60)

    # Find all Python files
    python_files = find_python_files(FINALIZATION_DIR)

    print(f"Found {len(python_files)} Python files to test\n")

    success_count = 0
    error_count = 0
    skip_count = 0
    errors = []

    for file_path in python_files:
        module_name, error = test_import(file_path, FINALIZATION_DIR)

        if module_name is None:
            skip_count += 1
        elif error:
            error_count += 1
            errors.append((file_path, error))
            print(f"  ❌ FAILED: {file_path}")
        else:
            success_count += 1
            print(f"  ✅ SUCCESS: {module_name}")

    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Skipped: {skip_count}")

    if errors:
        print("\n" + "=" * 60)
        print("DETAILED ERRORS:")
        for file_path, error in errors:
            print(f"\n{file_path}:")
            print("-" * 40)
            # Only print the first few lines of the error for clarity
            error_lines = error.split('\n')
            for line in error_lines[:10]:
                print(f"  {line}")
            if len(error_lines) > 10:
                print(f"  ... ({len(error_lines) - 10} more lines)")

    return error_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)