#!/usr/bin/env python3
"""
Script to fix import issues in finalization directory.
"""

import os
import re
from pathlib import Path

FINALIZATION_DIR = Path(__file__).parent.absolute()

# Mapping of incorrect imports to correct ones
IMPORT_FIXES = {
    # Models module fixes
    'from latentwire.models import InterlinguaInterlinguaEncoder': 'from latentwire.models import InterlinguaEncoder',
    'from latentwire.data import load_examples': 'from latentwire.data import load_examples',
    # Commented entries (removed due to syntax errors)
    # '# from latentwire.prefix_utils  # Module doesn't exist': '# from latentwire.prefix_utils',
    # '# import latentwire.prefix_utils  # Module doesn't exist': '# import latentwire.prefix_utils',

    # For files in finalization root that import from telepathy
    'from linear_probe_baseline import': 'from linear_probe_baseline import',

    # Fix import names that don't match actual function names
    'from latentwire.models import InterlinguaInterlinguaEncoder,': 'from latentwire.models import InterlinguaEncoder,',
}

def fix_file_imports(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Apply fixes
    for old_import, new_import in IMPORT_FIXES.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"  Fixed: {old_import}")

    # Special case: Fix Encoder when imported alone or with other items
    content = re.sub(
        r'from latentwire\.models import (.*?)Encoder(.*?)(?=\n)',
        r'from latentwire.models import \1InterlinguaInterlinguaEncoder\2',
        content
    )

    # Save if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix imports in all Python files."""
    print(f"Fixing imports in {FINALIZATION_DIR}")
    print("=" * 60)

    fixed_count = 0

    for root, dirs, files in os.walk(FINALIZATION_DIR):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(FINALIZATION_DIR)

                if fix_file_imports(file_path):
                    print(f"✓ Fixed: {rel_path}")
                    fixed_count += 1

    print("\n" + "=" * 60)
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main()