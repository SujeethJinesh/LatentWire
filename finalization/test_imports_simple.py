#!/usr/bin/env python3
"""
Simple test to check internal imports between files in finalization.
Ignores missing external packages like torch, transformers, etc.
"""

import os
import sys
import ast
from pathlib import Path

FINALIZATION_DIR = Path(__file__).parent.absolute()

def get_imports_from_file(file_path):
    """Extract import statements from a Python file using AST."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)
                if node.module:
                    imports.append(node.module)

        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def is_internal_import(import_name, base_dir):
    """Check if an import is internal to the finalization directory."""
    # Check if it's a relative import or starts with known internal modules
    internal_prefixes = ['latentwire', 'telepathy', 'features', 'scripts', 'models', 'losses', 'config']

    for prefix in internal_prefixes:
        if import_name.startswith(prefix):
            return True

    # Check if it could be a local file import
    if '.' not in import_name:
        # Could be a direct module import like 'models' or 'config'
        potential_files = [
            base_dir / f"{import_name}.py",
            base_dir / import_name / "__init__.py"
        ]
        for path in potential_files:
            if path.exists():
                return True

    return False

def check_internal_imports():
    """Check all internal imports in the finalization directory."""
    python_files = []
    for root, dirs, files in os.walk(FINALIZATION_DIR):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(Path(root) / file)

    print(f"Checking {len(python_files)} Python files for internal imports...\n")

    issues = []

    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(FINALIZATION_DIR)
        imports = get_imports_from_file(file_path)

        internal_imports = [imp for imp in imports if is_internal_import(imp, FINALIZATION_DIR)]

        if internal_imports:
            print(f"\n{rel_path}:")
            for imp in internal_imports:
                # Check if the import can be resolved
                import_parts = imp.split('.')

                # Check different possible paths
                possible_paths = [
                    FINALIZATION_DIR / f"{imp.replace('.', '/')}.py",
                    FINALIZATION_DIR / imp.replace('.', '/') / "__init__.py",
                ]

                # For imports like 'latentwire.models', check in latentwire subdirectory
                if import_parts[0] == 'latentwire':
                    possible_paths.append(
                        FINALIZATION_DIR / 'latentwire' / f"{'.'.join(import_parts[1:]).replace('.', '/')}.py"
                    )
                    possible_paths.append(
                        FINALIZATION_DIR / 'latentwire' / '.'.join(import_parts[1:]).replace('.', '/') / "__init__.py"
                    )

                found = False
                for path in possible_paths:
                    if path.exists():
                        print(f"  ✓ {imp} -> {path.relative_to(FINALIZATION_DIR)}")
                        found = True
                        break

                if not found:
                    issue = f"  ✗ {imp} - CANNOT RESOLVE"
                    print(issue)
                    issues.append((str(rel_path), imp))

    print("\n" + "=" * 60)
    if issues:
        print(f"Found {len(issues)} unresolvable internal imports:")
        for file_path, imp in issues:
            print(f"  {file_path}: {imp}")
    else:
        print("All internal imports resolved successfully!")

    return issues

if __name__ == "__main__":
    issues = check_internal_imports()
    sys.exit(0 if not issues else 1)