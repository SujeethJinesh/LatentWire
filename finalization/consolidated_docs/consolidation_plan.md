# Consolidation Plan

## Target: 10 or fewer files in root directory

### Files to keep in root:
1. **config.yaml** - Main configuration
2. **requirements.txt** - Python dependencies
3. **RUN_ALL.sh** - Main execution script
4. **README.md** - Project documentation
5. **MANIFEST.txt** - File inventory

### Directories to create and move files into:
- **tests/** - All test files
- **benchmarks/** - All benchmark scripts
- **docs/** - All documentation (.md files)
- **shell_scripts/** - All .sh files except RUN_ALL.sh
- **evaluation/** - All eval scripts