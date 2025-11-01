#!/bin/bash
#
# Comprehensive evaluation runner script
#
# This script provides convenient wrappers for running evaluations.
# Usage examples:
#   ./run_evaluations.sh                    # Run everything (eval + plots + reports)
#   ./run_evaluations.sh --eval-only        # Only run model evaluation
#   ./run_evaluations.sh --plots-only       # Only generate plots
#   ./run_evaluations.sh --reports-only     # Only generate reports
#   ./run_evaluations.sh --skip-existing    # Skip models with existing eval logs
#   ./run_evaluations.sh --dry-run          # Show what would be done
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PYTHON=${PYTHON:-python3}
STORAGE_DIR=${STORAGE_DIR:-storage}

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if storage directory exists
if [ ! -d "$STORAGE_DIR" ]; then
    print_error "Storage directory '$STORAGE_DIR' not found!"
    print_info "Set STORAGE_DIR environment variable to override the default location."
    exit 1
fi

# Check if Python is available
if ! command -v "$PYTHON" &> /dev/null; then
    print_error "Python not found! Please install Python 3 or set PYTHON environment variable."
    exit 1
fi

# Run the comprehensive evaluation script
print_info "Running comprehensive evaluation..."
print_info "Storage directory: $STORAGE_DIR"
print_info "Python: $PYTHON"
echo ""

# Pass all arguments to the Python script
"$PYTHON" run_all_evaluations.py --storage "$STORAGE_DIR" "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    print_success "Evaluation completed successfully!"
else
    print_error "Evaluation failed with exit code $exit_code"
fi

exit $exit_code
