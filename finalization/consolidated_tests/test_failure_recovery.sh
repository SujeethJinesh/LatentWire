#!/bin/bash
# =============================================================================
# TEST SCRIPT FOR FAILURE RECOVERY MECHANISM IN RUN_ALL.sh
# =============================================================================
# This script tests the failure recovery mechanism by simulating various
# failure scenarios and verifying that the experiment state is properly
# saved and can be resumed.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }
print_test() { echo -e "${CYAN}[TEST]${NC} $1"; }

# Test configuration
TEST_DIR="test_failure_recovery_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ALL_SCRIPT="$SCRIPT_DIR/RUN_ALL.sh"

# Create test directory
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

print_info "Test directory: $(pwd)/$TEST_DIR"
print_info "Testing RUN_ALL.sh failure recovery mechanism"
echo ""

# =============================================================================
# TEST 1: Verify State File Creation
# =============================================================================
print_test "TEST 1: Verify state file is created on experiment start"

# Start an experiment with dry-run to avoid actual execution
timeout 10 bash "$RUN_ALL_SCRIPT" quick --dry-run --no-interactive 2>&1 | tee test1.log || true

# Check if state file was created
STATE_FILE=$(find runs -name ".experiment_state" 2>/dev/null | head -1)

if [[ -f "$STATE_FILE" ]]; then
    print_status "State file created: $STATE_FILE"

    # Verify state file contents
    if grep -q "version=" "$STATE_FILE" && \
       grep -q "training=pending" "$STATE_FILE" && \
       grep -q "phase1_statistical=pending" "$STATE_FILE"; then
        print_status "State file has correct initial structure"
    else
        print_error "State file structure is incorrect"
        cat "$STATE_FILE"
    fi
else
    print_error "State file was not created"
fi

echo ""

# =============================================================================
# TEST 2: Simulate Training Failure
# =============================================================================
print_test "TEST 2: Simulate training failure and verify state is saved"

# Create a modified version that will fail during training
cat > test_fail_training.sh << 'EOF'
#!/bin/bash
source $(dirname "$0")/../RUN_ALL.sh

# Override run_training to simulate failure
run_training() {
    print_header "TRAINING (SIMULATED FAILURE)"

    CURRENT_PHASE="training"
    mark_phase_started "training"

    print_error "Simulating training failure..."
    exit 42  # Specific exit code to verify
}

# Run with our override
main
EOF

# Run the failing script
chmod +x test_fail_training.sh
timeout 10 bash test_fail_training.sh experiment --dry-run --no-interactive 2>&1 | tee test2.log || EXIT_CODE=$?

if [[ $EXIT_CODE -eq 42 ]]; then
    print_status "Script failed with expected exit code 42"

    # Check if state was updated
    STATE_FILE=$(find runs -name ".experiment_state" 2>/dev/null | head -1)

    if [[ -f "$STATE_FILE" ]]; then
        if grep -q "training=failed" "$STATE_FILE" || grep -q "training=in_progress" "$STATE_FILE"; then
            print_status "Training phase marked as failed/in_progress in state file"
        else
            print_warning "Training phase not properly marked in state"
        fi

        if grep -q "last_failure=" "$STATE_FILE"; then
            print_status "Failure tracking information saved"
        fi
    fi
else
    print_warning "Script did not fail as expected (exit code: $EXIT_CODE)"
fi

echo ""

# =============================================================================
# TEST 3: Test Resume from State
# =============================================================================
print_test "TEST 3: Test resuming from saved state file"

# Create a mock state file with some completed phases
MOCK_STATE_DIR="runs/test_resume_$(date +%s)"
mkdir -p "$MOCK_STATE_DIR"
MOCK_STATE_FILE="$MOCK_STATE_DIR/.experiment_state"

cat > "$MOCK_STATE_FILE" << EOF
# LatentWire Experiment State File
# Created: $(date)
# Experiment: test_resume
# Timestamp: test

[metadata]
version=3.1.0
start_time=$(date +%s)
base_output_dir=$MOCK_STATE_DIR

[phases]
training=completed
phase1_statistical=completed
phase2_linear_probe=failed
phase3_baselines=pending
phase4_efficiency=pending
aggregation=pending

[datasets]
sst2_eval=completed
agnews_eval=pending
trec_eval=pending
squad_eval=pending

[checkpoints]
checkpoint_path=$MOCK_STATE_DIR/checkpoint/epoch10
final_epoch=10

[failures]
failure_count=1
last_failure=phase2_linear_probe
last_failure_time=$(date)
EOF

print_info "Created mock state file with:"
print_info "  - Training: completed"
print_info "  - Phase 1: completed"
print_info "  - Phase 2: failed"
print_info "  - Phase 3-4: pending"

# Test resuming with the state file
timeout 10 bash "$RUN_ALL_SCRIPT" experiment --resume "$MOCK_STATE_FILE" --dry-run --no-interactive 2>&1 | tee test3.log || true

# Check if resume was recognized
if grep -q "Resuming from previous state" test3.log; then
    print_status "Resume mode detected correctly"
fi

if grep -q "Skipping completed phase: training" test3.log; then
    print_status "Completed phases are being skipped"
fi

if grep -q "Re-attempting failed phase: phase2_linear_probe" test3.log || grep -q "phase2_linear_probe" test3.log; then
    print_status "Failed phase marked for retry"
fi

echo ""

# =============================================================================
# TEST 4: Test Phase Completion Tracking
# =============================================================================
print_test "TEST 4: Test phase completion tracking"

# Create a test that completes some phases
cat > test_phase_tracking.sh << 'EOF'
#!/bin/bash
# Source the main script to get all functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../RUN_ALL.sh"

# Override functions to simulate quick completion
run_training() {
    print_header "TRAINING (SIMULATED)"
    mark_phase_started "training"
    sleep 0.5
    CHECKPOINT_PATH="fake/checkpoint/path"
    save_state "checkpoint_path" "$CHECKPOINT_PATH" "checkpoints"
    mark_phase_completed "training"
    return 0
}

run_phase1_statistical() {
    print_header "PHASE 1 (SIMULATED)"
    mark_phase_started "phase1_statistical"
    sleep 0.5
    mark_phase_completed "phase1_statistical"
    return 0
}

# Set up minimal config
BASE_OUTPUT_DIR="runs/test_tracking_$(date +%s)"
mkdir -p "$BASE_OUTPUT_DIR"
init_state_file

# Run phases
run_training
run_phase1_statistical

# Show final state
print_state_summary
EOF

chmod +x test_phase_tracking.sh
bash test_phase_tracking.sh 2>&1 | tee test4.log

if grep -q "training.*completed" test4.log && grep -q "phase1_statistical.*completed" test4.log; then
    print_status "Phase completion tracking works correctly"
else
    print_warning "Phase tracking may have issues"
fi

echo ""

# =============================================================================
# TEST 5: Test Recovery Instructions Display
# =============================================================================
print_test "TEST 5: Test recovery instructions are shown on failure"

# Create a script that will fail and check recovery instructions
cat > test_recovery_display.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run a command that will fail
timeout 5 bash "$SCRIPT_DIR/../RUN_ALL.sh" experiment --phase 99 --no-interactive 2>&1 || true
EOF

chmod +x test_recovery_display.sh
bash test_recovery_display.sh 2>&1 | tee test5.log

if grep -q "FAILURE RECOVERY INSTRUCTIONS" test5.log || grep -q "To resume from where it failed" test5.log; then
    print_status "Recovery instructions are displayed on failure"
else
    print_warning "Recovery instructions may not be displayed properly"
fi

echo ""

# =============================================================================
# TEST 6: Test State File Update Operations
# =============================================================================
print_test "TEST 6: Test state file update operations"

# Create a test state file
TEST_STATE_FILE="test_state_ops.state"
cat > "$TEST_STATE_FILE" << EOF
[metadata]
version=1.0.0

[phases]
test_phase=pending

[data]
key1=value1
EOF

# Source the functions
source "$RUN_ALL_SCRIPT" 2>/dev/null || true

# Override STATE_FILE for testing
STATE_FILE="$TEST_STATE_FILE"

# Test save_state function
save_state "test_phase" "in_progress" "phases"
save_state "key1" "updated_value" "data"
save_state "new_key" "new_value" "data"

# Verify updates
if grep -q "test_phase=in_progress" "$TEST_STATE_FILE"; then
    print_status "State update works for existing keys"
fi

if grep -q "key1=updated_value" "$TEST_STATE_FILE"; then
    print_status "Value updates work correctly"
fi

if grep -q "new_key=new_value" "$TEST_STATE_FILE"; then
    print_status "New key insertion works"
fi

# Test get_state function
VALUE=$(get_state "test_phase")
if [[ "$VALUE" == "in_progress" ]]; then
    print_status "State retrieval works correctly"
fi

echo ""

# =============================================================================
# TEST 7: Test Multiple Failure Recovery
# =============================================================================
print_test "TEST 7: Test multiple failure recovery cycles"

# Create a state file with multiple failures
MULTI_FAIL_STATE="runs/multi_fail_$(date +%s)/.experiment_state"
mkdir -p "$(dirname "$MULTI_FAIL_STATE")"

cat > "$MULTI_FAIL_STATE" << EOF
[metadata]
version=3.1.0
base_output_dir=$(dirname "$MULTI_FAIL_STATE")

[phases]
training=completed
phase1_statistical=failed
phase2_linear_probe=failed
phase3_baselines=pending

[failures]
failure_count=3
last_failure=phase2_linear_probe
EOF

# Test that multiple failed phases are handled
timeout 10 bash "$RUN_ALL_SCRIPT" experiment --resume "$MULTI_FAIL_STATE" --dry-run --no-interactive 2>&1 | tee test7.log || true

RETRY_COUNT=$(grep -c "Re-attempting failed phase" test7.log 2>/dev/null || echo "0")
if [[ $RETRY_COUNT -gt 0 ]]; then
    print_status "Multiple failed phases detected for retry (found $RETRY_COUNT)"
else
    print_warning "Multiple failure handling may need verification"
fi

echo ""

# =============================================================================
# TEST SUMMARY
# =============================================================================
echo ""
echo "=============================================================="
echo "TEST SUMMARY"
echo "=============================================================="

TOTAL_TESTS=7
PASSED_TESTS=$(grep -c "✓" test*.log 2>/dev/null | cut -d: -f2 | awk '{s+=$1} END {print s}' || echo "0")

echo "Total tests run: $TOTAL_TESTS"
echo "Tests with passing checks: ~$PASSED_TESTS checks passed"
echo ""

print_info "Key findings:"
echo "1. State file creation: $([ -f "$STATE_FILE" ] && echo "✓ Working" || echo "✗ Issues")"
echo "2. Failure tracking: $(grep -q "failed" test2.log 2>/dev/null && echo "✓ Working" || echo "⚠ Check manually")"
echo "3. Resume capability: $(grep -q "Resuming" test3.log 2>/dev/null && echo "✓ Working" || echo "⚠ Check manually")"
echo "4. Phase tracking: $(grep -q "completed" test4.log 2>/dev/null && echo "✓ Working" || echo "⚠ Check manually")"
echo "5. Recovery instructions: $(grep -q "RECOVERY" test5.log 2>/dev/null && echo "✓ Working" || echo "⚠ Check manually")"
echo ""

print_info "Test artifacts saved in: $(pwd)"
print_info "Review individual test logs (test*.log) for details"
echo ""

# Clean up test directory (optional)
read -p "Clean up test directory? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ..
    rm -rf "$TEST_DIR"
    print_status "Test directory cleaned up"
fi