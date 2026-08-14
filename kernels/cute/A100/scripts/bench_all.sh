#!/bin/bash
for v in v1 v2 v3 v4 v5 v6 v7 v8 ptx_gemm; do
    echo ""
    echo "========================================"
    echo "  ${v}.cu"
    echo "========================================"
    modal run scripts/run.py --task "kernels/cute/A100/${v}.cu" 2>&1 | \
        grep -E '(CUTE_GEMM:|CORRECTNESS|TFlop|Error|FAIL)'
done
