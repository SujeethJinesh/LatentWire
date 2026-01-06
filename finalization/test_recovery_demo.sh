#!/bin/bash
# =============================================================================
# DEMONSTRATION OF FAILURE RECOVERY MECHANISM
# =============================================================================
# This script demonstrates the failure recovery mechanism with a real scenario
# where training fails partway through and can be resumed.
# =============================================================================

set +e  # Don't exit on error - we want to test recovery

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}=============================================================="
    echo -e "$1"
    echo -e "==============================================================${NC}"
    echo ""
}

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

# Create demo directory
DEMO_DIR="recovery_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

print_header "FAILURE RECOVERY DEMONSTRATION"
print_info "Demo directory: $(pwd)"
echo ""

# =============================================================================
# STEP 1: Create a mock experiment that will fail
# =============================================================================

print_header "STEP 1: Setting up experiment that will fail at Phase 2"

# Create mock experiment script with intentional failure
cat > mock_experiment.sh << 'EOF'
#!/bin/bash
set +e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# State management
STATE_FILE=".experiment_state"
CURRENT_PHASE=""

save_state() {
    local key="$1"
    local value="$2"

    if grep -q "^${key}=" "$STATE_FILE" 2>/dev/null; then
        sed -i.bak "s|^${key}=.*|${key}=${value}|" "$STATE_FILE"
    else
        echo "${key}=${value}" >> "$STATE_FILE"
    fi
}

get_state() {
    local key="$1"
    grep "^${key}=" "$STATE_FILE" 2>/dev/null | cut -d'=' -f2-
}

check_completed() {
    local phase="$1"
    [[ "$(get_state "$phase")" == "completed" ]]
}

# Initialize state file if not exists
if [[ ! -f "$STATE_FILE" ]]; then
    cat > "$STATE_FILE" << 'STATE'
# Experiment State
experiment_name=mock_experiment
start_time=$(date)

# Phases
phase1_training=pending
phase2_evaluation=pending
phase3_analysis=pending

# Checkpoints
checkpoint_path=
failure_count=0
STATE
fi

echo -e "${BLUE}Starting experiment...${NC}"
echo "State file: $STATE_FILE"
echo ""

# Phase 1: Training
if ! check_completed "phase1_training"; then
    echo -e "${YELLOW}Running Phase 1: Training${NC}"
    save_state "phase1_training" "in_progress"

    # Simulate training
    for i in {1..3}; do
        echo "  Training epoch $i/3..."
        sleep 0.5
    done

    # Save checkpoint
    mkdir -p checkpoints
    echo "model_weights" > checkpoints/final.pt
    save_state "checkpoint_path" "checkpoints/final.pt"
    save_state "phase1_training" "completed"
    echo -e "${GREEN}  ✓ Training completed${NC}"
else
    echo -e "${GREEN}Skipping completed Phase 1: Training${NC}"
fi

echo ""

# Phase 2: Evaluation (will fail)
if ! check_completed "phase2_evaluation"; then
    echo -e "${YELLOW}Running Phase 2: Evaluation${NC}"
    save_state "phase2_evaluation" "in_progress"

    # Check if this is a retry
    failure_count=$(get_state "failure_count")

    if [[ "$failure_count" -eq "0" ]]; then
        # First attempt - simulate failure
        echo "  Loading checkpoint..."
        sleep 0.5
        echo "  Starting evaluation..."
        sleep 0.5
        echo -e "${RED}  ERROR: Out of memory during evaluation!${NC}"

        save_state "phase2_evaluation" "failed"
        save_state "failure_count" "1"
        save_state "last_failure" "phase2_evaluation"
        save_state "last_failure_time" "$(date)"

        echo ""
        echo -e "${RED}EXPERIMENT FAILED${NC}"
        echo ""
        echo "To resume from this point, run:"
        echo "  ./mock_experiment.sh"
        echo ""
        echo "The experiment will:"
        echo "  - Skip completed Phase 1 (training)"
        echo "  - Retry failed Phase 2 (evaluation)"
        echo "  - Continue with Phase 3 (analysis)"

        exit 1
    else
        # Retry attempt - succeed this time
        echo "  [RETRY] Adjusting batch size to avoid OOM..."
        echo "  Loading checkpoint from: $(get_state checkpoint_path)"
        sleep 0.5
        echo "  Running evaluation with smaller batch..."
        sleep 1

        # Save results
        mkdir -p results
        echo "accuracy: 0.92" > results/eval.txt

        save_state "phase2_evaluation" "completed"
        echo -e "${GREEN}  ✓ Evaluation completed (on retry)${NC}"
    fi
else
    echo -e "${GREEN}Skipping completed Phase 2: Evaluation${NC}"
fi

echo ""

# Phase 3: Analysis
if ! check_completed "phase3_analysis"; then
    echo -e "${YELLOW}Running Phase 3: Analysis${NC}"
    save_state "phase3_analysis" "in_progress"

    echo "  Analyzing results..."
    sleep 0.5
    echo "  Generating plots..."
    sleep 0.5

    # Save analysis
    mkdir -p analysis
    echo "Statistical significance: p<0.05" > analysis/stats.txt

    save_state "phase3_analysis" "completed"
    echo -e "${GREEN}  ✓ Analysis completed${NC}"
else
    echo -e "${GREEN}Skipping completed Phase 3: Analysis${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}EXPERIMENT COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results:"
echo "  - Training checkpoint: $(get_state checkpoint_path)"
echo "  - Evaluation results: results/eval.txt"
echo "  - Analysis: analysis/stats.txt"
echo ""
echo "State history:"
grep "=" "$STATE_FILE" | grep -E "(phase|failure)" | while read line; do
    echo "  $line"
done
EOF

chmod +x mock_experiment.sh

print_status "Created mock_experiment.sh with intentional Phase 2 failure"
echo ""

# =============================================================================
# STEP 2: Run experiment (will fail)
# =============================================================================

print_header "STEP 2: Running experiment (will fail at Phase 2)"

./mock_experiment.sh || FAILED=$?

if [[ $FAILED -eq 1 ]]; then
    print_warning "Experiment failed as expected!"
    echo ""

    # Show state file
    print_info "Current state file contents:"
    cat .experiment_state | grep -E "(phase|checkpoint|failure)" | while read line; do
        echo "    $line"
    done
    echo ""
else
    print_error "Experiment should have failed but didn't!"
fi

# =============================================================================
# STEP 3: Demonstrate recovery
# =============================================================================

print_header "STEP 3: Resuming failed experiment"

print_info "The experiment will now resume from where it failed..."
print_info "Notice how it:"
echo "  1. Skips the completed training phase"
echo "  2. Retries the failed evaluation phase"
echo "  3. Continues with remaining phases"
echo ""

read -p "Press Enter to resume the experiment..." -n 1 -r
echo ""

./mock_experiment.sh

print_status "Experiment completed successfully after recovery!"
echo ""

# =============================================================================
# STEP 4: Show final state
# =============================================================================

print_header "STEP 4: Final State Analysis"

print_info "Final state file:"
cat .experiment_state | grep -E "(phase|checkpoint|failure)" | while read line; do
    echo "    $line"
done
echo ""

print_info "Generated artifacts:"
find . -type f -not -name "*.sh" -not -name ".*" | while read f; do
    echo "    $f"
done
echo ""

# =============================================================================
# DEMONSTRATION WITH RUN_ALL.sh
# =============================================================================

print_header "DEMONSTRATION WITH RUN_ALL.sh"

print_info "The same recovery mechanism is built into RUN_ALL.sh"
echo ""
echo "Example workflow:"
echo ""
echo -e "${BOLD}1. Start an experiment:${NC}"
echo "   bash RUN_ALL.sh experiment"
echo ""
echo -e "${BOLD}2. If it fails (e.g., OOM, error), you'll see:${NC}"
echo -e "   ${YELLOW}FAILURE RECOVERY INSTRUCTIONS${NC}"
echo "   To resume from where it failed, run:"
echo -e "   ${GREEN}bash RUN_ALL.sh experiment --resume runs/experiment/.experiment_state${NC}"
echo ""
echo -e "${BOLD}3. Resume with the provided command:${NC}"
echo "   - Completed phases will be skipped"
echo "   - Failed phase will be retried"
echo "   - Remaining phases will continue"
echo ""
echo -e "${BOLD}Key features:${NC}"
echo "   ✓ Automatic state tracking"
echo "   ✓ Phase-level granularity"
echo "   ✓ Checkpoint preservation"
echo "   ✓ Failure count tracking"
echo "   ✓ Multiple retry support"
echo ""

# =============================================================================
# CLEANUP
# =============================================================================

print_header "CLEANUP"

echo "Demo artifacts are in: $(pwd)"
echo ""
read -p "Remove demo directory? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ..
    rm -rf "$DEMO_DIR"
    print_status "Demo directory removed"
else
    print_info "Demo artifacts kept in: $DEMO_DIR"
fi

echo ""
print_status "Recovery demonstration complete!"