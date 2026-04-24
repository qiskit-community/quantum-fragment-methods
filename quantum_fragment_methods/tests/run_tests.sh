#!/bin/bash
# Test runner script for quantum-fragment-methods
# Run this inside the container where pytest, pyscf, and vayesta are installed

set -e

echo "==================================="
echo "Quantum Fragment Methods Test Suite"
echo "==================================="
echo ""

# Check if we're in the container
if [ ! -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "WARNING: Not running in container environment"
    echo "Tests require pyscf, vayesta, and pytest"
    echo ""
fi

# Activate conda environment if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate qfrag-env
    echo "Activated conda environment: qfrag-env"
    echo ""
fi

# Navigate to package root
cd "$(dirname "$0")/../../.."

echo "Running tests from: $(pwd)"
echo ""

# Run pytest with various options
echo "--- Running EWF Tests ---"
python -m pytest quantum_fragment_methods/tests/test_ewf.py -v --tb=short

echo ""
echo "==================================="
echo "Test run complete!"
echo "==================================="