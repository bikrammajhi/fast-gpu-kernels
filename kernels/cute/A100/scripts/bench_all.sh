#!/bin/bash
for v in v1 v2 v3 v4 v5 v6; do
    echo ""
    echo "========================================"
    echo "  matmul_${v}.cu"
    echo "========================================"
    modal run scripts/run.py --task "kernels/cute/A100/matmul_${v}.cu" 2>&1 | \
        grep -E '(CUTE_GEMM:|CORRECTNESS|TFlop|Error|FAIL)'
done
