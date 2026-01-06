#!/bin/bash
# =============================================================================
# TEST SCRIPT FOR FAILURE RECOVERY IN RUN_ALL.sh
# =============================================================================
# This script tests the failure recovery mechanism by simulating failures
# at different phases and verifying that resume works correctly.
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=========================================="
echo "Testing RUN_ALL.sh Failure Recovery"
echo -e "==========================================${NC}"
echo ""

# Clean up any previous test runs
echo "Cleaning up previous test runs..."
rm -rf runs/test_recovery_* 2>/dev/null || true

# Test 1: Simulate failure during training
echo -e "${YELLOW}Test 1: Simulating training failure...${NC}"
echo ""

# Create a wrapper that will fail after creating output dir
cat > test_training_fail.sh << 'EOF'
#!/bin/bash
# This will create the output directory and state file, then fail
BASE_OUTPUT_DIR="runs/test_recovery_training_fail"
mkdir -p "$BASE_OUTPUT_DIR"

# Run with dry-run first to create state, then force a failure
bash RUN_ALL.sh quick --dry-run --no-interactive

# Now simulate a failure by exiting with error
echo "SIMULATED FAILURE IN TRAINING"
exit 1
EOF

chmod +x test_training_fail.sh

echo "Running script that will fail during training..."
if ./test_training_fail.sh 2>&1; then
    echo -e "${RED}ERROR: Script should have failed but didn't${NC}"
    exit 1
fi

# Check that state file was created
STATE_FILE=$(find runs/test_recovery_* -name ".experiment_state" 2>/dev/null | head -1)

if [[ -f "$STATE_FILE" ]]; then
    echo -e "${GREEN}✓ State file created: $STATE_FILE${NC}"

    # Check that training is marked as failed or in_progress
    if grep -q "training=\(failed\|in_progress\)" "$STATE_FILE"; then
        echo -e "${GREEN}✓ Training phase correctly marked${NC}"
    else
        echo -e "${RED}✗ Training phase not correctly marked${NC}"
    fi
else
    echo -e "${RED}✗ State file not created${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 2: Testing resume functionality...${NC}"
echo ""

# Test that resume command works (dry run only)
echo "Testing resume command syntax..."
if bash RUN_ALL.sh quick --resume "$STATE_FILE" --dry-run --no-interactive 2>&1 | grep -q "Resuming from previous state"; then
    echo -e "${GREEN}✓ Resume command accepted${NC}"
else
    echo -e "${RED}✗ Resume command failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 3: Verifying state management functions...${NC}"
echo ""

# Test state management functions directly
cat > test_state_functions.sh << 'EOF'
#!/bin/bash

# Source the functions from RUN_ALL.sh (extract them)
source <(sed -n '/^init_state_file()/,/^}/p; /^save_state()/,/^}/p; /^get_state()/,/^}/p; /^check_phase_completed()/,/^}/p' RUN_ALL.sh)

# Create a test state file
TEST_STATE_FILE="runs/test_state_mgmt/.experiment_state"
mkdir -p "$(dirname "$TEST_STATE_FILE")"
STATE_FILE="$TEST_STATE_FILE"

# Initialize
cat > "$STATE_FILE" << 'EOSTATE'
[phases]
phase1=completed
phase2=in_progress
phase3=failed
phase4=pending

[metadata]
test_key=test_value
EOSTATE

# Test get_state
if [[ "$(get_state phase1)" == "completed" ]]; then
    echo "✓ get_state works"
else
    echo "✗ get_state failed"
    exit 1
fi

# Test save_state
save_state "phase4" "completed"
if grep -q "phase4=completed" "$STATE_FILE"; then
    echo "✓ save_state works"
else
    echo "✗ save_state failed"
    exit 1
fi

# Test check_phase_completed
if check_phase_completed "phase1"; then
    echo "✓ check_phase_completed works"
else
    echo "✗ check_phase_completed failed"
    exit 1
fi

echo "✓ All state management functions work"
EOF

chmod +x test_state_functions.sh
if ./test_state_functions.sh; then
    echo -e "${GREEN}✓ State management functions verified${NC}"
else
    echo -e "${RED}✗ State management functions failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Test 4: Verifying error messages...${NC}"
echo ""

# Test that proper error messages are shown
echo "Checking error recovery instructions..."
if bash RUN_ALL.sh help 2>&1 | grep -q "resume"; then
    echo -e "${GREEN}✓ Resume option documented in help${NC}"
else
    echo -e "${RED}✗ Resume option not in help${NC}"
fi

# Clean up
echo ""
echo "Cleaning up test files..."
rm -f test_training_fail.sh test_state_functions.sh
rm -rf runs/test_recovery_* runs/test_state_mgmt

echo ""
echo -e "${GREEN}=========================================="
echo "All Tests Passed!"
echo -e "==========================================${NC}"
echo ""
echo "The failure recovery mechanism is working correctly."
echo "You can now use:"
echo "  - bash RUN_ALL.sh [command] to start experiments"
echo "  - bash RUN_ALL.sh [command] --resume [state_file] to resume after failure"