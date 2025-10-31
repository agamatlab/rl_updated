#!/bin/bash
#
# Bash wrapper for analyzing training and evaluation results
# This script provides a simple interface to the Python analysis script
#

set -e

# Default values
STORAGE_PATH="./storage"
OUTPUT_FILE="environment_classifications.json"
REPORT_FILE="environment_report.txt"
CONVERGENCE_THRESHOLD=0.05
HIGH_RESHAPED_RETURN=0.7
HIGH_EVAL_RETURN=0.9

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Analyze training and evaluation results for RL environments.

OPTIONS:
    -s, --storage PATH          Path to storage directory (default: ./storage)
    -o, --output FILE          Output JSON file (default: environment_classifications.json)
    -r, --report FILE          Summary report text file (default: environment_report.txt)
    -c, --convergence FLOAT    Convergence threshold (default: 0.05)
    --high-reshaped FLOAT      High reshaped return threshold (default: 0.7)
    --high-eval FLOAT          High evaluation return threshold (default: 0.9)
    --no-report                Don't generate text report
    -h, --help                 Show this help message

EXAMPLES:
    # Run with default settings
    $0

    # Specify custom storage path and output
    $0 -s ./my_storage -o results.json

    # Adjust thresholds for stricter convergence criteria
    $0 -c 0.03 --high-reshaped 0.8 --high-eval 0.95

EOF
    exit 1
}

# Parse arguments
GENERATE_REPORT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--storage)
            STORAGE_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -c|--convergence)
            CONVERGENCE_THRESHOLD="$2"
            shift 2
            ;;
        --high-reshaped)
            HIGH_RESHAPED_RETURN="$2"
            shift 2
            ;;
        --high-eval)
            HIGH_EVAL_RETURN="$2"
            shift 2
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if storage directory exists
if [ ! -d "$STORAGE_PATH" ]; then
    echo "Error: Storage directory '$STORAGE_PATH' does not exist"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if the analysis script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analyze_training_simple.py"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found at $ANALYSIS_SCRIPT"
    exit 1
fi

# Build command
CMD="python3 \"$ANALYSIS_SCRIPT\" --storage-path \"$STORAGE_PATH\" --output \"$OUTPUT_FILE\""
CMD="$CMD --convergence-threshold $CONVERGENCE_THRESHOLD"
CMD="$CMD --high-reshaped-return $HIGH_RESHAPED_RETURN"
CMD="$CMD --high-eval-return $HIGH_EVAL_RETURN"

if [ "$GENERATE_REPORT" = true ]; then
    CMD="$CMD --report \"$REPORT_FILE\""
fi

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Storage path: $STORAGE_PATH"
echo "  Output file: $OUTPUT_FILE"
if [ "$GENERATE_REPORT" = true ]; then
    echo "  Report file: $REPORT_FILE"
fi
echo "  Convergence threshold: $CONVERGENCE_THRESHOLD"
echo "  High reshaped return threshold: $HIGH_RESHAPED_RETURN"
echo "  High eval return threshold: $HIGH_EVAL_RETURN"
echo ""

# Run the analysis
eval $CMD

# Print results location
echo ""
echo -e "${GREEN}Analysis complete!${NC}"
echo "Results saved to: $OUTPUT_FILE"
if [ "$GENERATE_REPORT" = true ]; then
    echo "Report saved to: $REPORT_FILE"
fi