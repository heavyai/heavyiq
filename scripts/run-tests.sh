#!/bin/bash
# Run HeavyIQ test suite
#
# Usage:
#   ./scripts/run-tests.sh              # Run unit tests only (default, in container)
#   ./scripts/run-tests.sh --e2e        # Run e2e tests only (default, in container)
#   ./scripts/run-tests.sh --all        # Run all tests (unit + integration)
#   ./scripts/run-tests.sh --integration # Run integration tests only
#   ./scripts/run-tests.sh --coverage   # Run with coverage report
#   ./scripts/run-tests.sh --local      # Run on host instead of container
#   ./scripts/run-tests.sh <path>       # Run specific test file/directory

set -e

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default: run in container, unit + e2e tests (skip integration only)
USE_CONTAINER=true
PYTEST_MARKER="not integration"
COVERAGE_FLAG=""
TEST_PATH=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_CONTAINER=false
            shift
            ;;
        --container)
            USE_CONTAINER=true
            shift
            ;;
        --all)
            PYTEST_MARKER=""
            echo "Running ALL tests (unit + integration + e2e)..."
            shift
            ;;
        --integration)
            PYTEST_MARKER="integration"
            echo "Running INTEGRATION tests only..."
            shift
            ;;
        --e2e)
            PYTEST_MARKER="e2e"
            echo "Running E2E (API endpoint) tests only..."
            shift
            ;;
        --unit)
            PYTEST_MARKER="not integration and not e2e"
            echo "Running UNIT tests only (no e2e)..."
            shift
            ;;
        --coverage)
            COVERAGE_FLAG="--cov=heavyiq --cov=heavyrag --cov-report=html --cov-report=term"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [TEST_PATH]"
            echo ""
            echo "Options:"
            echo "  --container     Run tests in Docker container (default)"
            echo "  --local         Run tests on host machine"
            echo "  --unit          Run unit tests only (skips both integration & e2e)"
            echo "  --integration   Run integration tests only (ChromaDB, embedding server)"
            echo "  --e2e           Run e2e API endpoint tests only (full app stack)"
            echo "  --all           Run all tests (unit + integration + e2e)"
            echo "  --coverage      Generate coverage report"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Default: runs unit + e2e tests (skips integration)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run unit + e2e tests (default)"
            echo "  $0 --e2e                              # Run only e2e API tests"
            echo "  $0 --integration                      # Run integration tests"
            echo "  $0 --all                              # Run everything"
            echo "  $0 tests/api/test_rag_router.py       # Run specific test file"
            exit 0
            ;;
        -*)
            # Pass other flags to pytest
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            # Assume it's a test path
            TEST_PATH="$TEST_PATH $1"
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="python -m pytest"

if [ -n "$PYTEST_MARKER" ]; then
    PYTEST_CMD="$PYTEST_CMD -m \"$PYTEST_MARKER\""
fi

if [ -n "$COVERAGE_FLAG" ]; then
    PYTEST_CMD="$PYTEST_CMD $COVERAGE_FLAG"
fi

# Default test path based on marker or default to tests/
if [ -n "$TEST_PATH" ]; then
    PYTEST_CMD="$PYTEST_CMD $TEST_PATH"
elif [ "$PYTEST_MARKER" = "e2e" ]; then
    # e2e tests are in tests/api/
    PYTEST_CMD="$PYTEST_CMD tests/api/"
else
    PYTEST_CMD="$PYTEST_CMD tests/"
fi

if [ -n "$EXTRA_ARGS" ]; then
    PYTEST_CMD="$PYTEST_CMD $EXTRA_ARGS"
fi

if [ "$USE_CONTAINER" = true ]; then
    # Use container config path
    CONFIG_PATH="config.container.toml"
    PYTEST_CMD="$PYTEST_CMD --config-path $CONFIG_PATH"
    
    echo "Running tests in Docker container..."
    echo "Config: $CONFIG_PATH"
    echo "Command: $PYTEST_CMD"
    echo ""
    
    # Build environment variable flags
    ENV_FLAGS="-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONPATH=/code"
    
    # Pass through OPENAI_API_KEY if set
    if [ -n "$OPENAI_API_KEY" ]; then
        ENV_FLAGS="$ENV_FLAGS -e OPENAI_API_KEY=$OPENAI_API_KEY"
    fi
    
    # Check if container is running
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^heavyiq-ubuntu$'; then
        # Container is running, exec into it
        docker exec $ENV_FLAGS heavyiq-ubuntu bash -c "$PYTEST_CMD"
    else
        echo "Container 'heavyiq-ubuntu' is not running."
        echo "Starting temporary container for tests..."
        
        # Start a temporary container for testing (use -T for non-TTY)
        docker compose -f docker/docker-compose.ubuntu.yml run --rm -T \
            $ENV_FLAGS \
            --entrypoint "" \
            heavyiq bash -c "$PYTEST_CMD"
    fi
else
    # Use local config path
    CONFIG_PATH="config.toml"
    PYTEST_CMD="$PYTEST_CMD --config-path $CONFIG_PATH"
    
    echo "Running tests locally on host..."
    echo "Config: $CONFIG_PATH"
    echo "Command: $PYTEST_CMD"
    echo ""
    
    # Activate virtual environment if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    elif [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi

    # Set environment variables for tests
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    # Run tests
    eval $PYTEST_CMD
fi
