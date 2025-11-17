#!/bin/bash
# Quick comparison script - assumes Python outputs are in python_output/ directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_DIR="${1:-python_output}"
CPP_DIR="${2:-output}"
OUTPUT_DIR="${3:-comparison_output}"

echo "Comparing Python and C++ outputs..."
echo "  Python dir: $PYTHON_DIR"
echo "  C++ dir:    $CPP_DIR"
echo "  Output dir: $OUTPUT_DIR"

cd "$PROJECT_ROOT"
python3 scripts/compare_depths.py \
  --python_dir "$PYTHON_DIR" \
  --cpp_dir "$CPP_DIR" \
  --output "$OUTPUT_DIR"
