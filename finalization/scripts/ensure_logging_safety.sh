#!/usr/bin/env bash
# ensure_logging_safety.sh - Ensure logs are not lost during SLURM preemption
#
# This script adds critical safeguards to prevent log buffering issues:
# 1. Sets PYTHONUNBUFFERED=1 in all execution contexts
# 2. Adds flush=True to critical Python print statements
# 3. Ensures tee is used properly with pipefail

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "ENSURING LOGGING SAFETY FOR SLURM"
echo "=========================================="
echo ""

# 1. Fix RUN_ALL.sh to ensure PYTHONUNBUFFERED is set in SLURM context
echo "Checking RUN_ALL.sh for SLURM logging safety..."
if grep -q "export PYTHONUNBUFFERED=1" RUN_ALL.sh; then
    echo -e "${GREEN}✓${NC} RUN_ALL.sh already has PYTHONUNBUFFERED=1"
else
    echo -e "${YELLOW}⚠${NC} RUN_ALL.sh missing PYTHONUNBUFFERED=1 - fixing..."
    # Add it after the PYTHONPATH export in SLURM section
    sed -i.bak '/export PYTHONPATH=./a\
export PYTHONUNBUFFERED=1' RUN_ALL.sh
    echo -e "${GREEN}✓${NC} Added PYTHONUNBUFFERED=1 to RUN_ALL.sh"
fi

# 2. Add flush=True to critical print statements in training script
echo ""
echo "Adding flush=True to critical prints in latentwire/train.py..."
python3 << 'EOF'
import re

file_path = 'latentwire/train.py'
with open(file_path, 'r') as f:
    content = f.read()

# Critical patterns that need immediate flushing
critical_patterns = [
    (r'print\(f?".*[Ee]poch.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*[Ss]tep.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*[Ss]aved.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*[Cc]omplete.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*[Tt]raining.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*PEAK.*"(?!, flush=True)', ', flush=True'),
    (r'print\(f?".*checkpoint.*"(?!, flush=True)', ', flush=True'),
]

modified = False
for pattern, replacement in critical_patterns:
    # Find lines matching pattern
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.search(pattern, line) and 'flush=True' not in line:
            # Add flush=True before the closing parenthesis
            if line.rstrip().endswith(')'):
                lines[i] = line.rstrip()[:-1] + ', flush=True)'
                modified = True

if modified:
    content = '\n'.join(lines)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Modified {file_path} to add flush=True")
else:
    print(f"No changes needed in {file_path}")
EOF

# 3. Create a wrapper script for SLURM jobs that ensures proper buffering
echo ""
echo "Creating SLURM wrapper script..."
cat > scripts/slurm_safe_wrapper.sh << 'EOF'
#!/bin/bash
# SLURM-safe wrapper that ensures logs are not lost

# Critical: Disable all buffering
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Ensure output is line-buffered
stdbuf -oL -eL "$@" 2>&1 | tee -a "${SLURM_LOG:-slurm_output.log}"
EOF
chmod +x scripts/slurm_safe_wrapper.sh
echo -e "${GREEN}✓${NC} Created scripts/slurm_safe_wrapper.sh"

# 4. Add a preemption handler that flushes logs
echo ""
echo "Creating preemption handler..."
cat > scripts/handle_preemption.py << 'EOF'
#!/usr/bin/env python3
"""
Handle SLURM preemption signals to ensure logs are flushed.
"""
import signal
import sys
import time

def handle_preemption(signum, frame):
    """Handle preemption signal by flushing all outputs."""
    print("\n[PREEMPTION] Received signal {}, flushing logs...".format(signum), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(1)  # Give time for logs to write
    print("[PREEMPTION] Logs flushed, exiting gracefully.", flush=True)
    sys.exit(0)

# Register handlers for common SLURM signals
signal.signal(signal.SIGTERM, handle_preemption)  # SLURM preemption
signal.signal(signal.SIGUSR1, handle_preemption)  # SLURM warning

if __name__ == "__main__":
    print("Preemption handler registered", flush=True)
    # Keep the script running
    while True:
        time.sleep(60)
EOF
chmod +x scripts/handle_preemption.py
echo -e "${GREEN}✓${NC} Created scripts/handle_preemption.py"

# 5. Update key training scripts to use unbuffered output
echo ""
echo "Updating critical shell scripts..."
for script in run_with_resume.sh run_integration_test.sh run_example_eval.sh; do
    if [ -f "$script" ]; then
        if grep -q "PYTHONUNBUFFERED" "$script"; then
            echo -e "${GREEN}✓${NC} $script already has PYTHONUNBUFFERED"
        else
            # Add after shebang
            sed -i.bak '2i\
export PYTHONUNBUFFERED=1\
' "$script"
            echo -e "${GREEN}✓${NC} Updated $script"
        fi
    fi
done

# 6. Create a verification script
echo ""
echo "Creating verification script..."
cat > scripts/verify_logging_safety.sh << 'EOF'
#!/bin/bash
# Verify that logging safety measures are in place

echo "Verifying logging safety measures..."
echo ""

# Check for PYTHONUNBUFFERED in key files
echo "Checking for PYTHONUNBUFFERED=1:"
for file in RUN_ALL.sh run_with_resume.sh run_integration_test.sh; do
    if [ -f "$file" ]; then
        if grep -q "PYTHONUNBUFFERED=1" "$file"; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file - MISSING!"
        fi
    fi
done

echo ""
echo "Checking for flush=True in train.py:"
critical_count=$(grep -c "flush=True" latentwire/train.py 2>/dev/null || echo "0")
echo "  Found $critical_count print statements with flush=True"

echo ""
echo "Checking for tee usage in key scripts:"
for file in RUN_ALL.sh run_with_resume.sh; do
    if [ -f "$file" ]; then
        if grep -q "| tee" "$file"; then
            echo "  ✓ $file uses tee"
        else
            echo "  ✗ $file - NO TEE!"
        fi
    fi
done

echo ""
echo "Verification complete."
EOF
chmod +x scripts/verify_logging_safety.sh
echo -e "${GREEN}✓${NC} Created scripts/verify_logging_safety.sh"

# Run verification
echo ""
echo "=========================================="
echo "RUNNING VERIFICATION"
echo "=========================================="
bash scripts/verify_logging_safety.sh

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo -e "${GREEN}Logging safety measures have been applied:${NC}"
echo "1. PYTHONUNBUFFERED=1 set in critical scripts"
echo "2. flush=True added to critical print statements"
echo "3. Created SLURM-safe wrapper script"
echo "4. Created preemption handler"
echo "5. Verification script available"
echo ""
echo -e "${YELLOW}IMPORTANT for SLURM jobs:${NC}"
echo "1. Always use PYTHONUNBUFFERED=1 in SLURM scripts"
echo "2. Use tee to capture logs: command 2>&1 | tee logfile.log"
echo "3. Add flush=True to critical Python prints"
echo "4. Use the slurm_safe_wrapper.sh for critical jobs"
echo ""
echo "To use the safe wrapper in SLURM:"
echo "  scripts/slurm_safe_wrapper.sh python train.py [args]"
echo ""
echo -e "${GREEN}✓ Logging safety configuration complete${NC}"