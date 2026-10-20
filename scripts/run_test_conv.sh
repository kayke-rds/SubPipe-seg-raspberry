#!/bin/bash
# run_conv_test.sh <conv_idx>
set -e
IDX=${1:-44}
echo "=== Gerando teste para conv $IDX ==="
python python_files/gen_test_conv.py "$IDX"
echo
echo "=== Rodando kernel C ==="
./test_conv
echo
echo "=== Comparando ==="
python python_files/compare_conv.py
