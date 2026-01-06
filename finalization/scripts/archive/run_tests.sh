#!/usr/bin/env bash
# Run tests locally with the same configuration as CI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running LatentWire Test Suite${NC}"
echo "================================"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    if [[ -f .venv/bin/activate ]]; then
        echo "Activating .venv..."
        source .venv/bin/activate
    else
        echo -e "${RED}No virtual environment found. Please create one first.${NC}"
        exit 1
    fi
fi

# Set environment variables
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Parse arguments
COVERAGE=false
VERBOSE=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --quick|-q)
            QUICK=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --coverage    Generate coverage report"
            echo "  --verbose,-v  Verbose output"
            echo "  --quick,-q    Run only quick tests"
            echo "  --help,-h     Show this help"
            exit 0
            ;;
    esac
done

# Install test dependencies if needed
echo "Checking test dependencies..."
pip install -q pytest pytest-cov pytest-xdist pytest-timeout || {
    echo -e "${RED}Failed to install test dependencies${NC}"
    exit 1
}

# Build pytest command
PYTEST_CMD="python -m pytest"

if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=latentwire --cov-report=term-missing --cov-report=html"
fi

if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [[ "$QUICK" == true ]]; then
    # Run only quick unit tests
    PYTEST_CMD="$PYTEST_CMD tests/cli tests/train tests/features -x"
else
    # Run full test suite
    PYTEST_CMD="$PYTEST_CMD tests/ --ignore=tests/integration/test_checkpoint_roundtrip.py"
fi

# Add common options
PYTEST_CMD="$PYTEST_CMD --tb=short --color=yes --maxfail=10 --timeout=300"

# Run tests
echo -e "${GREEN}Running: $PYTEST_CMD${NC}"
echo "================================"

if $PYTEST_CMD; then
    echo "================================"
    echo -e "${GREEN}✅ All tests passed!${NC}"

    if [[ "$COVERAGE" == true ]]; then
        echo -e "${GREEN}Coverage report saved to htmlcov/index.html${NC}"
    fi
else
    echo "================================"
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi