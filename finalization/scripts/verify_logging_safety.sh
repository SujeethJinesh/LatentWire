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
